#!/usr/bin/env python
"""Diagnostics du bras token : plateau, identifiabilite de gamma, balayage.

Ces trois mesures sont citees dans `HYPOTHESES.md` (H11 et le piege P3) mais
n'avaient aucun script dans le depot : elles avaient ete produites au fil de
l'investigation. Personne ne pouvait donc les reproduire ni les contester, ce
qui est le pire statut possible pour des chiffres qui portent des conclusions.

Trois modes, correspondant aux trois questions posees dans l'ordre :

  --mode plateau    Le bras token apprend-il ? A lr=2e-3, celui des bras conv
                    et dense, sa reconstruction reste bloquee ~30 epoques avant
                    de decrocher ; `patience=10` l'arrete en plein plateau et il
                    est alors evalue sans avoir rien appris. C'est ce qui a
                    produit un premier verdict ou il finissait dernier partout.

  --mode identif    gamma est-il identifie ? On entraine depuis des
                    initialisations eloignees. Si chacune reste ou on l'a posee,
                    le gradient ne porte aucune information sur la localite et
                    lire gamma apres entrainement ne renseigne que sur son
                    initialisation.

  --mode sweep      gamma etant non identifiable, on le BALAIE. Avec plusieurs
                    graines : a une seule, la courbe semblait decroissante puis
                    remontait au dernier point, ce qui trahissait du bruit. Ce
                    mode mesure aussi, en passant, le PLANCHER DE BRUIT entre
                    graines — la quantite qui disqualifie la plupart des effets
                    discutes dans HYPOTHESES.md.

    python scripts/diagnose_token.py --mode plateau
    python scripts/diagnose_token.py --mode sweep --seeds 0 1 2 3
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import cluster, evaluate  # noqa: E402
from msvae.models import VAEConfig, match_token_to_conv  # noqa: E402
from msvae.pipeline import (ExperimentConfig, _electrode_positions,  # noqa: E402
                            build_bank, load_records, make_downstream_score)
from msvae.train import TrainConfig, train_vae  # noqa: E402


def setup(a):
    eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.085,
               trans_boost=1.0 + a.effect_t * 1.5)
    cfg = ExperimentConfig(dataset="synthetic", n_subjects=a.n_subjects,
                           duration=a.duration, latent_dim=a.latent_dim, **eff)
    records, gt = load_records(cfg)
    bank, _ = build_bank(records, cfg.image_size)
    tr, va = bank.split_subjects(cfg.val_frac, seed=cfg.seed)
    tr_b = tr.balanced(a.n_per_subject, cfg.balance_percentile, seed=cfg.seed)
    va_b = va.balanced(a.n_per_subject, cfg.balance_percentile, seed=cfg.seed)
    n_ch = bank.topo.shape[1]
    conv_ref = VAEConfig(kind="conv", latent_dim=a.latent_dim, beta=1e-3,
                         image_size=cfg.image_size, n_channels_eeg=n_ch)
    base = match_token_to_conv(conv_ref, n_ch,
                               elec_pos=_electrode_positions(records[0], n_ch))
    val_records = [r for r in records if r.subject in set(va.subjects)]
    score = make_downstream_score(va_b, val_records, cfg.k, cfg.min_segment_ms,
                                  cfg.seed)
    return cfg, records, gt, bank, tr_b, va_b, base, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["plateau", "identif", "sweep"],
                   default="plateau")
    p.add_argument("--n-subjects", type=int, default=120)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--n-per-subject", type=int, default=None)
    p.add_argument("--effect-t", type=float, default=0.35)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gammas", type=float, nargs="+", default=[0.05, 0.693, 3.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", default="results/diagnostics_token")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, records, gt, bank, tr_b, va_b, base, score = setup(a)
    print(f"{len(tr_b)} pics, d_model={base.d_model}\n", flush=True)
    payload = {"mode": a.mode, "n_subjects": a.n_subjects, "rows": []}

    if a.mode == "plateau":
        print("lr     | reconstruction validation, une valeur par 10 epoques")
        for lr in (2e-3, 5e-4):
            r = train_vae(base, tr_b.topo, va_b.topo, None,
                          TrainConfig(epochs=a.epochs, seed=0, device="auto",
                                      batch_size=a.batch_size, verbose=False,
                                      patience=10 ** 6, lr=lr))
            rec = [h["val_recon"] for h in r.history]
            print(f"{lr:6g} | " + " ".join(f"{rec[i]:.3f}"
                                           for i in range(0, len(rec), 10)))
            print(f"       | min {min(rec):.4f} a l'epoque {int(np.argmin(rec))}",
                  flush=True)
            payload["rows"].append(dict(lr=lr, recon=rec,
                                        best_epoch=int(np.argmin(rec))))

    elif a.mode == "identif":
        print("init gamma | gamma final              | reconstruction")
        for g in a.gammas:
            r = train_vae(replace(base, gamma_init=g), tr_b.topo, va_b.topo, None,
                          TrainConfig(epochs=a.epochs, seed=0, device="auto",
                                      batch_size=a.batch_size, verbose=False,
                                      patience=10 ** 6, lr=5e-4))
            fin = r.model.learned_locality()
            rec = min(h["val_recon"] for h in r.history)
            print(f"{g:10.3f} | {fin.mean():.3f} "
                  f"[{fin.min():.3f}, {fin.max():.3f}] | {rec:.4f}", flush=True)
            payload["rows"].append(dict(gamma_init=g, gamma_final=fin.tolist(),
                                        recon_best=float(rec)))

    else:  # sweep
        print("gamma | GEV aval par graine | moyenne ± ecart-type | |r| au gt")
        for g in a.gammas:
            gevs, corrs = [], []
            for s in a.seeds:
                c = replace(base, gamma_init=g)
                r = train_vae(c, tr_b.topo, va_b.topo, None,
                              TrainConfig(epochs=a.epochs, seed=s, device="auto",
                                          batch_size=a.batch_size, verbose=False,
                                          patience=10 ** 6, lr=5e-4))
                gevs.append(-score(r.model, c))
                if gt is not None:
                    mp = cluster.decoded_maps_from_bank(r.model, bank, cfg.k,
                                                        mode="two_stage",
                                                        seed=cfg.seed,
                                                        kind="token")[0]
                    corrs.append(np.mean(np.abs(evaluate.align_maps(mp, gt)[2])))
            v = np.array(gevs)
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"{g:5.3f} | " + " ".join(f"{x:.4f}" for x in v)
                  + f" | {v.mean():.4f} ± {sd:.4f}"
                  + (f" | {np.mean(corrs):.3f}" if corrs else ""), flush=True)
            payload["rows"].append(dict(gamma=g, seeds=list(a.seeds),
                                        gev=[float(x) for x in v],
                                        gev_mean=float(v.mean()),
                                        gev_sd=float(sd),
                                        corr_gt=float(np.mean(corrs)) if corrs
                                        else None))
        allsd = [r["gev_sd"] for r in payload["rows"] if r["gev_sd"]]
        if allsd:
            print(f"\nPLANCHER DE BRUIT entre graines : {min(allsd):.3f} a "
                  f"{max(allsd):.3f} de GEV.\nAucun ecart inferieur a ~"
                  f"{2 * max(allsd):.2f} ne doit etre interprete a une graine.")
            payload["noise_floor"] = [float(min(allsd)), float(max(allsd))]

    (out / f"token_{a.mode}.json").write_text(json.dumps(payload, indent=2))
    print(f"ok -> {out}/token_{a.mode}.json")


if __name__ == "__main__":
    main()
