#!/usr/bin/env python
"""Banc de classification : compare le pouvoir discriminant des methodes.

Prend un run existant (`results/<run>/`), re-construit la cohorte a partir de
sa configuration, back-fitte les cartes de CHAQUE methode sur les memes
enregistrements, puis compare les methodes sur leur capacite a separer deux
groupes de sujets.

Toutes les methodes voient exactement les memes sujets, les memes plis de
validation croisee et le meme classifieur : seul le jeu de cartes change.

Deux bras de reference sont ajoutes systematiquement :
  - `spectral_global` / `spectral_channel` : puissance relative par bande +
    frequence du pic alpha. C'est le point zero. Si aucune methode de
    microstates ne les bat, la representation n'apporte rien.
  - `ground_truth` quand il existe (cohorte synthetique) : plafond atteignable.

Exemples
--------
    python scripts/run_classification.py --run results/synthetic
    python scripts/run_classification.py --run results/ds004504 --n-repeats 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import classify, microstates  # noqa: E402
from msvae.pipeline import ExperimentConfig, load_records  # noqa: E402


def _backfit_params(records, maps, min_segment_ms):
    """Cartes -> parametres de microstates par sujet (meme chemin que le pipeline)."""
    params = []
    for rec in records:
        seg = microstates.backfit(rec.data.astype(np.float64), maps, rec.sfreq,
                                  min_segment_ms=min_segment_ms)
        params.append(microstates.microstate_parameters(seg, rec.boundaries))
    return params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="repertoire d'un run (contient maps.npz)")
    p.add_argument("--out", default=None, help="defaut : <run>/classification")
    p.add_argument("--positive", default=None,
                   help="groupe code 1 (defaut : le second par ordre alphabetique)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=0,
                   help="controle par permutation des etiquettes (0 = ignore)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    run = Path(a.run)
    out = Path(a.out) if a.out else run / "classification"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cfg_dict = json.loads((run / "results.json").read_text())["config"]
    known = set(ExperimentConfig().to_dict())
    cfg = ExperimentConfig(**{k: v for k, v in cfg_dict.items() if k in known})
    print(f"cohorte : {cfg.dataset}, K={cfg.k}", flush=True)

    records, _ = load_records(cfg)
    maps_npz = np.load(run / "maps.npz")
    groups = np.array([r.group for r in records])
    subjects = np.array([r.subject for r in records])

    uniq = sorted(set(groups))
    if len(uniq) != 2:
        raise SystemExit(f"il faut exactement deux groupes, trouve : {uniq}")
    pos = a.positive or uniq[1]
    if pos not in uniq:
        raise SystemExit(f"groupe positif inconnu : {pos} (dispo : {uniq})")
    neg = next(g for g in uniq if g != pos)   # et non uniq[0], qui peut valoir pos
    y = (groups == pos).astype(int)
    print(f"groupes : {neg}={int((y == 0).sum())}, {pos}={int(y.sum())} "
          f"(positif = {pos})", flush=True)

    # ------------------------------------------------------------- features
    arms = {}
    for name in maps_npz.files:
        maps = maps_npz[name]
        if maps.shape[1] != records[0].data.shape[0]:
            print(f"  {name}: ignore ({maps.shape[1]} canaux != "
                  f"{records[0].data.shape[0]})")
            continue
        params = _backfit_params(records, maps, cfg.min_segment_ms)
        X, names = classify.features_from_params(params)
        arms[name] = X
        print(f"  {name}: {X.shape[1]} features", flush=True)

    for scope in ("global", "channel"):
        X, names = classify.spectral_features(records, scope=scope)
        arms[f"spectral_{scope}"] = X
        print(f"  spectral_{scope}: {X.shape[1]} features", flush=True)

    # ----------------------------------------------------------- evaluation
    print("evaluation (validation croisee repetee)", flush=True)
    results = {}
    for name, X in arms.items():
        r = classify.evaluate_arm(X, y, subjects, n_splits=a.n_splits,
                                  n_repeats=a.n_repeats, seed=a.seed)
        results[name] = r
        print(f"  {name:28s} AUC = {r['auc']:.3f} "
              f"(ecart-type entre repetitions {r['auc_sd_repeats']:.3f})", flush=True)

    comp = classify.compare_arms(results, y, n_boot=a.n_boot, seed=a.seed)

    perm = {}
    if a.n_perm:
        print(f"controle par permutation ({a.n_perm} tirages)", flush=True)
        for name, X in arms.items():
            perm[name] = classify.permutation_control(
                X, y, subjects, n_perm=a.n_perm, seed=a.seed,
                n_splits=a.n_splits, n_repeats=1)
            print(f"  {name:28s} AUC permutee = {perm[name]['mean']:.3f}", flush=True)

    payload = dict(
        run=str(run), dataset=cfg.dataset, k=cfg.k, positive=pos,
        n_subjects=len(records), n_positive=int(y.sum()),
        n_splits=a.n_splits, n_repeats=a.n_repeats,
        auc={n: r["auc"] for n, r in results.items()},
        auc_sd_repeats={n: r["auc_sd_repeats"] for n, r in results.items()},
        auc_ci=comp["auc_ci"], pairwise=comp["pairwise"],
        permutation=perm,
    )
    (out / "classification.json").write_text(json.dumps(payload, indent=2))
    _plot(results, comp, y, out / "figures" / "classification.png")
    _markdown(payload, out / "CLASSIFICATION.md")
    print(f"ok -> {out}")


def _plot(results, comp, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    order = sorted(results, key=lambda n: results[n]["auc"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    for i, name in enumerate(order):
        lo, hi = comp["auc_ci"][name]
        auc = results[name]["auc"]
        color = "#7f8c8d" if name.startswith("spectral") else (
            "#27ae60" if name == "ground_truth" else "#2c6fbb")
        ax.plot([lo, hi], [i, i], color=color, lw=2.4, alpha=0.75)
        ax.plot([auc], [i], "o", color=color, ms=7)
    ax.axvline(0.5, color="#c0392b", ls="--", lw=1, label="hasard")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("AUC (IC 95 %, bootstrap sur les sujets)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    for name in order:
        fpr, tpr, _ = roc_curve(y, results[name]["prob"])
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} ({results[name]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#c0392b", lw=1)
    ax.set_xlabel("taux de faux positifs")
    ax.set_ylabel("taux de vrais positifs")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Pouvoir discriminant par methode", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _markdown(payload, path):
    order = sorted(payload["auc"], key=lambda n: -payload["auc"][n])
    lines = [f"# Classification — {payload['dataset']} (K = {payload['k']})", "",
             f"{payload['n_subjects']} sujets, positif = `{payload['positive']}` "
             f"({payload['n_positive']} sujets). Validation croisee "
             f"{payload['n_splits']} plis x {payload['n_repeats']} repetitions.", "",
             "| Methode | AUC | IC 95 % |", "|---|---:|---|"]
    for n in order:
        lo, hi = payload["auc_ci"][n]
        lines.append(f"| `{n}` | {payload['auc'][n]:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "## Comparaisons appariees (bootstrap sur les sujets)", "",
              "| Paire | delta AUC | IC 95 % | p |", "|---|---:|---|---:|"]
    for pair, d in sorted(payload["pairwise"].items(),
                          key=lambda kv: -abs(kv[1]["delta"])):
        lo, hi = d["ci"]
        lines.append(f"| {pair} | {d['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] "
                     f"| {d['p_two_sided']:.3f} |")
    if payload["permutation"]:
        lines += ["", "## Controle par permutation (doit valoir ~0.5)", "",
                  "| Methode | AUC permutee |", "|---|---:|"]
        for n, v in payload["permutation"].items():
            lines.append(f"| `{n}` | {v['mean']:.3f} ± {v['sd']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
