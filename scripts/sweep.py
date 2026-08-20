#!/usr/bin/env python
"""Balayage (graine x K) parallelise, reprenable, avec agregation.

Pourquoi ce script existe. Le plancher de bruit entre graines vaut 0.03 a 0.065
de GEV — plus que la plupart des ecarts que le projet cherche a etablir. Toute
conclusion exige donc plusieurs graines, ce qui multiplie le calcul. Trois
choix rendent cela tenable :

  * PARALLELISME. Le poste dominant est le back-fitting, du numpy mono-thread
    sur CPU ; l'entrainement GPU represente moins de 1 % du temps. Les cellules
    (graine, K) etant independantes, `--jobs 4` divise le temps d'attente par
    ~4 sur une machine a 16 coeurs, sans que le GPU devienne le goulot (les
    modeles font ~110 000 parametres).
  * REPRISE. Chaque cellule ecrit son propre JSON et les cellules deja
    presentes sont sautees. Une execution de plusieurs heures interrompue
    reprend ou elle s'est arretee, au lieu de tout refaire.
  * ALLEGEMENT. On ne calcule QUE ce qui sert au balayage : ni variantes
    `latentfit` (le poste dominant du pipeline complet, qui rend chaque
    echantillon en image), ni stabilite split-half, ni figures.

Exemples
--------
    # ancrage : VaDE retrouve-t-il K = 4 sur une cohorte ou la verite est 4 ?
    python scripts/sweep.py --dataset synthetic --effect transition --effect-t 0.35 \
        --n-subjects 120 --seeds 0 1 2 3 4 5 6 7 8 9 --k 2 3 4 5 6 7 \
        --arms vade --jobs 4 --out results/sweep_k_synth

    # courbe AUC contre K sur donnees cliniques, toutes methodes
    python scripts/sweep.py --dataset ds004504 --seeds 0 1 2 --k 3 4 5 6 \
        --jobs 4 --out results/sweep_k_ds004504
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALL_ARMS = ("pycrostates", "pca8", "conv", "dense", "token", "vade")


# ------------------------------------------------------------------ une cellule
def run_cell(a) -> dict:
    """Calcule une cellule (graine, K) : cartes de chaque bras, GEV, AUC."""
    import torch  # noqa: F401  (initialise CUDA dans le sous-processus)

    from msvae import baseline, classify, cluster, microstates, vade
    from msvae.models import VAEConfig, match_dense_to_conv, match_token_to_conv
    from msvae.pipeline import (ExperimentConfig, _electrode_positions,
                                build_bank, load_records)
    from msvae.train import TrainConfig, train_vae

    eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.065, trans_boost=2.5)
    if a.effect == "duration":
        eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.085 - a.effect_t * 0.020,
                   trans_boost=1.0)
    elif a.effect == "transition":
        eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.085,
                   trans_boost=1.0 + a.effect_t * 1.5)

    cfg = ExperimentConfig(dataset=a.dataset, n_subjects=a.n_subjects,
                           duration=a.duration, k=a.k, latent_dim=a.latent_dim,
                           seed=a.seed, n_per_subject=a.n_per_subject, **eff)
    records, gt = load_records(cfg)
    bank, info = build_bank(records, cfg.image_size)
    _, val_bank = bank.split_subjects(cfg.val_frac, seed=a.seed)
    full_bal = bank.balanced(cfg.n_per_subject, cfg.balance_percentile, seed=a.seed)
    val_bal = val_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                seed=a.seed)
    n_ch = bank.topo.shape[1]
    mask = bank.projector.mask
    arms = set(a.arms)

    conv_cfg = VAEConfig(kind="conv", latent_dim=a.latent_dim,
                         image_size=cfg.image_size, beta=1e-3,
                         n_channels_eeg=n_ch, loss_space="topo")
    readout = bank.projector.readout_matrix()
    base_tc = dict(epochs=a.epochs, seed=a.seed, device="auto",
                   batch_size=a.batch_size, verbose=False)

    maps, extra = {}, {}
    models = {}

    if "conv" in arms:
        r = train_vae(conv_cfg, full_bal.images, val_bal.images, mask,
                      TrainConfig(**base_tc), readout=readout)
        models["conv"] = (r.model, conv_cfg)
    if "dense" in arms or "vade" in arms:
        d = match_dense_to_conv(conv_cfg, n_ch)
        r = train_vae(d, full_bal.topo, val_bal.topo, None, TrainConfig(**base_tc))
        models["dense"] = (r.model, d)
    if "token" in arms:
        t = match_token_to_conv(conv_cfg, n_ch,
                                elec_pos=_electrode_positions(records[0], n_ch))
        # regime propre au bras token : a 2e-3 il plateaute ~30 epoques et
        # l'early stopping l'arrete avant qu'il n'ait rien appris. 200 epoques
        # et non 80 : sur les 19 canaux de ds004504 l'optimum tombe a l'epoque
        # 261, donc 80 le laissait sous-appris — la meme panne, un cran plus
        # loin.
        r = train_vae(t, full_bal.topo, val_bal.topo, None,
                      TrainConfig(**{**base_tc, "lr": 5e-4, "patience": 10 ** 6,
                                     "epochs": max(a.epochs, 200)}))
        models["token"] = (r.model, t)
        rec = [h["val_recon"] for h in r.history]
        extra["token_recon"] = float(min(rec))
        # `best_epoch` proche de la fin = entrainement coupe en cours : la
        # cellule est alors a jeter, et sans cette trace rien ne le dirait
        extra["token_best_epoch"] = int(np.argmin(rec))
        extra["token_n_epochs"] = len(rec)
        extra["token_locality"] = r.model.learned_locality().mean().item()

    for kind, (model, mcfg) in models.items():
        if kind == "dense" and "dense" not in arms:
            continue
        for mode in ("two_stage", "weighted"):
            maps[f"vae_{kind}_{mode}"] = cluster.decoded_maps_from_bank(
                model, bank, a.k, mode=mode, seed=a.seed, kind=kind)[0]

    if "vade" in arms:
        vm, hist = vade.fit_vade(models["dense"][0], full_bal.topo, val_bal.topo,
                                 n_components=a.k, epochs=a.vade_epochs,
                                 lr=1e-3, lambda_balance=a.vade_lambda_balance,
                                 batch_size=a.batch_size, seed=a.seed,
                                 verbose=False)
        maps["vade_dense"] = vade.component_maps(vm, full_bal.topo)
        import torch as _t
        w = _t.softmax(vm.prior.pi_logits.detach(), 0).cpu().numpy()
        extra["vade"] = dict(val_elbo=hist[-1].get("val_elbo"),
                             weights=w.tolist(),
                             components_alive=int((w > 0.01).sum()),
                             balance=hist[-1].get("train_balance"),
                             balance_max=hist[-1].get("train_balance_max"))

    if "pycrostates" in arms:
        try:
            maps["pycrostates"] = baseline.modkmeans_group(bank, info, a.k,
                                                           seed=a.seed)
        except Exception as exc:
            extra["pycrostates_error"] = str(exc)
    if "pca8" in arms:
        try:
            m, meta = baseline.pca_modkmeans_group(bank, info, a.k,
                                                   n_components=a.latent_dim,
                                                   seed=a.seed)
            maps[f"pca{a.latent_dim}_modkmeans"] = m
        except Exception as exc:
            extra["pca_error"] = str(exc)
    if gt is not None:
        maps["ground_truth"] = gt

    # ------------------------------------------------- GEV et pouvoir discriminant
    groups = np.array([r.group for r in records])
    subjects = np.array([r.subject for r in records])
    uniq = sorted(set(groups))
    pos = a.positive or uniq[-1]
    y = (groups == pos).astype(int)

    gev, auc = {}, {}
    for name, mp in maps.items():
        params, gevs = [], []
        for rec in records:
            seg = microstates.backfit(rec.data.astype(np.float64), mp, rec.sfreq,
                                      min_segment_ms=cfg.min_segment_ms)
            params.append(microstates.microstate_parameters(seg, rec.boundaries))
            gevs.append(params[-1]["gev_total"])
        gev[name] = float(np.mean(gevs))
        X, _ = classify.features_from_params(params)
        auc[name] = classify.evaluate_arm(X, y, subjects, n_splits=a.n_splits,
                                          n_repeats=a.n_repeats,
                                          seed=a.seed)["auc"]

    Xs, _ = classify.spectral_features(records, scope="global")
    auc["spectral_global"] = classify.evaluate_arm(
        Xs, y, subjects, n_splits=a.n_splits, n_repeats=a.n_repeats,
        seed=a.seed)["auc"]

    return dict(seed=a.seed, k=a.k, dataset=a.dataset, positive=pos,
                n_subjects=len(records), gev=gev, auc=auc, extra=extra)


# ---------------------------------------------------------------------- pilote
def cell_path(out: Path, seed: int, k: int) -> Path:
    return out / "cells" / f"seed{seed:02d}_k{k}.json"


def drive(a):
    out = Path(a.out)
    (out / "cells").mkdir(parents=True, exist_ok=True)
    todo = [(s, k) for s in a.seeds for k in a.k
            if not cell_path(out, s, k).exists()]
    done = len(a.seeds) * len(a.k) - len(todo)
    print(f"{len(todo)} cellules a calculer, {done} deja presentes "
          f"({a.jobs} en parallele)", flush=True)

    base = [sys.executable, "-u", str(Path(__file__).resolve()), "--cell",
            "--dataset", a.dataset, "--n-subjects", str(a.n_subjects),
            "--duration", str(a.duration), "--latent-dim", str(a.latent_dim),
            "--epochs", str(a.epochs), "--vade-epochs", str(a.vade_epochs),
            "--batch-size", str(a.batch_size), "--n-splits", str(a.n_splits),
            "--n-repeats", str(a.n_repeats), "--effect", a.effect,
            "--effect-t", str(a.effect_t), "--out", str(out),
            "--vade-lambda-balance", str(a.vade_lambda_balance),
            "--arms", *a.arms]
    if a.n_per_subject:
        base += ["--n-per-subject", str(a.n_per_subject)]
    if a.positive:
        base += ["--positive", a.positive]

    t0 = time.time()
    lock = {"n": 0}

    def work(cell):
        s, k = cell
        cmd = base + ["--seed", str(s), "--k", str(k)]
        t_cell = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        lock["n"] += 1
        ok = p.returncode == 0 and cell_path(out, s, k).exists()
        tag = "ok " if ok else "ECHEC"
        # duree de LA CELLULE, et non temps ecoule depuis le demarrage du
        # pilote : les cellules paralleles se terminant presque ensemble,
        # afficher le cumul donnait l'illusion que chacune avait pris tout ce
        # temps. On affiche aussi le debit, seul chiffre utile pour planifier.
        dt = time.time() - t_cell
        deb = (time.time() - t0) / max(lock["n"], 1)
        print(f"  [{lock['n']}/{len(todo)}] {tag} graine {s} K={k} — "
              f"cellule {dt / 3600:.1f} h, debit {deb / 3600:.1f} h/cellule",
              flush=True)
        if not ok:
            print((p.stderr or "")[-800:], flush=True)

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(work, todo))

    aggregate(out, a)


def aggregate(out: Path, a):
    cells = [json.loads(p.read_text()) for p in sorted((out / "cells").glob("*.json"))]
    if not cells:
        print("aucune cellule")
        return
    ks = sorted({c["k"] for c in cells})
    names = sorted({n for c in cells for n in c["auc"]})

    n_seeds = len({c["seed"] for c in cells})
    lines = [f"# Balayage — {cells[0]['dataset']}", "",
             f"{n_seeds} graines, "
             f"{cells[0]['n_subjects']} sujets, positif = `{cells[0]['positive']}`.",
             "", "## AUC (moyenne ± écart-type entre graines)", "",
             "| Méthode | " + " | ".join(f"K={k}" for k in ks) + " |",
             "|---|" + "---:|" * len(ks)]
    summary = {"auc": {}, "gev": {}, "vade": {}}
    incomplets = []
    for n in names:
        row = []
        for k in ks:
            v = np.array([c["auc"][n] for c in cells
                          if c["k"] == k and n in c["auc"]])
            n_att = sum(1 for c in cells if c["k"] == k)
            summary["auc"].setdefault(n, {})[k] = dict(
                mean=float(v.mean()) if len(v) else None,
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                n=int(len(v)), n_attendu=int(n_att))
            if len(v) == 0:
                row.append("—")
                continue
            cell = (f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1
                    else f"{v.mean():.3f}")
            # un bras qui echoue sur une partie des cellules (pycrostates ne
            # converge pas toujours) donnerait sinon une moyenne d'apparence
            # normale, calculee sur moins de graines que les autres, sans que
            # rien ne le signale
            if len(v) < n_att:
                cell += f" ⚠{len(v)}/{n_att}"
                incomplets.append(f"`{n}` a K={k} : {len(v)}/{n_att} graines")
            row.append(cell)
        lines.append(f"| `{n}` | " + " | ".join(row) + " |")
    if incomplets:
        lines += ["", "> **Moyennes incompletes** — ces bras ont echoue sur une "
                  "partie des cellules, leur moyenne porte donc sur moins de "
                  "graines que les autres et n'est pas comparable :", ""]
        lines += [f"> - {s}" for s in incomplets]

    ve = [(c["k"], c["extra"]["vade"]) for c in cells if "vade" in c.get("extra", {})]
    if ve:
        lines += ["", "## VaDE : sélection de K", "",
                  "| K | −ELBO validation | composantes vivantes |",
                  "|---:|---:|---:|"]
        for k in ks:
            e = [v for kk, v in ve if kk == k]
            if not e:
                continue
            el = np.array([x["val_elbo"] for x in e if x["val_elbo"] is not None])
            al = np.array([x["components_alive"] for x in e])
            summary["vade"][k] = dict(elbo_mean=float(el.mean()) if len(el) else None,
                                      elbo_sd=float(el.std(ddof=1)) if len(el) > 1 else 0.0,
                                      alive_mean=float(al.mean()))
            lines.append(f"| {k} | {el.mean():.4f} ± "
                         f"{el.std(ddof=1) if len(el) > 1 else 0:.4f} | "
                         f"{al.mean():.1f} / {k} |")
        lines += ["", "*Rappel : l'ELBO tend à surestimer K. Il doit être "
                  "confronté à un critère externe, pas lu seul.*"]

    (out / "SWEEP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"ok -> {out}/SWEEP.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="store_true", help="usage interne")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "ds004504", "eegbci"])
    p.add_argument("--n-subjects", type=int, default=120)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--vade-epochs", type=int, default=40)
    # ATTENTION pour la selection de K. Le terme d'equilibrage empeche une
    # composante de mourir — c'est ce qu'on veut quand K est FIXE au bon
    # nombre, mais c'est exactement le signal qu'on cherche quand on selectionne
    # K : une composante qui se vide a K=6 dit que 6 est trop. Le garder actif
    # pousse le modele a utiliser ses K composantes quel que soit K, donc a
    # mieux ajuster les grands K, donc a les surestimer. Mettre 0 pour la
    # selection de K ; garder 0.5 pour un run a K fixe.
    p.add_argument("--vade-lambda-balance", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-per-subject", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--effect", choices=["reference", "duration", "transition"],
                   default="reference")
    p.add_argument("--effect-t", type=float, default=1.0)
    p.add_argument("--positive", default=None)
    p.add_argument("--arms", nargs="+", default=list(ALL_ARMS), choices=ALL_ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--seed", type=int, default=0, help="cellule uniquement")
    p.add_argument("--k", type=int, nargs="+", default=[4])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--aggregate-only", action="store_true")
    a = p.parse_args()

    if a.cell:
        a.k = a.k[0] if isinstance(a.k, list) else a.k
        res = run_cell(a)
        cell_path(Path(a.out), a.seed, a.k).write_text(json.dumps(res, indent=2))
        return
    if a.aggregate_only:
        aggregate(Path(a.out), a)
        return
    drive(a)


if __name__ == "__main__":
    main()
