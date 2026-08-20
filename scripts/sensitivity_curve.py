#!/usr/bin/env python
"""Courbe de sensibilite : quel est le plus petit effet de groupe detectable ?

Au lieu de comparer des AUC ponctuelles sur un effet fixe — ce qui sature (le
premier passage a donne 0.94 a 1.00 pour tous les bras, IC entierement
recouvrants) — on fait varier l'amplitude de l'effet et on trace l'AUC en
fonction de cette amplitude, une courbe par methode.

Trois proprietes que la GEV n'a pas :

  * la metrique NE PEUT PAS saturer, puisqu'on choisit la plage d'effets ;
  * a effet nul, toutes les methodes doivent tomber a 0.5 : toute valeur
    superieure signale une fuite du protocole, et se voit immediatement ;
  * les methodes se classent par le plus petit effet qu'elles detectent, ce
    qui est la question que se pose reellement un clinicien.

Deux regimes, et la distinction est le coeur de l'affaire :

  --mode duration   : l'effet porte sur la duree moyenne des segments. C'est
                      une propriete temporelle du signal, donc la puissance
                      relative par bande la lit directement. Sert de CONTROLE :
                      le bras spectral doit y gagner, et une methode de
                      microstates qui n'y arrive pas est en defaut.

  --mode transition : l'effet porte uniquement sur la probabilite C -> D, a
                      durees APPARIEES. La composition frequentielle est alors
                      inchangee et le spectral n'a rien a lire ; seule une
                      methode qui recupere la sequence d'etats peut detecter
                      l'effet. C'est le seul regime ou l'apport propre des
                      microstates est demontrable.

Exemples
--------
    python scripts/sensitivity_curve.py --mode transition --n-subjects 120
    python scripts/sensitivity_curve.py --mode duration --levels 0 0.25 0.5 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import classify, microstates  # noqa: E402
from msvae.data import iter_synthetic  # noqa: E402

# amplitudes de reference (t = 1) : celles de la cohorte de reference
FULL_DURATION_GAP = 0.085 - 0.065      # 20 ms
FULL_TRANS_BOOST = 2.5


def cohort_params(mode: str, t: float) -> dict:
    """t dans [0, 1] : 0 = aucun effet, 1 = l'effet de la cohorte de reference."""
    if mode == "duration":
        return dict(mean_dur_g1=0.085, mean_dur_g2=0.085 - t * FULL_DURATION_GAP,
                    trans_boost=1.0)
    if mode == "transition":
        # durees APPARIEES : le spectral ne doit avoir aucune prise
        return dict(mean_dur_g1=0.085, mean_dur_g2=0.085,
                    trans_boost=1.0 + t * (FULL_TRANS_BOOST - 1.0))
    raise ValueError(f"mode inconnu : {mode}")


def arms_for_cohort(records, gt_maps, k, min_segment_ms, with_pycrostates, seed):
    """Matrices de features par bras, sur une cohorte donnee."""
    arms = {}

    maps_by_method = {"ground_truth": gt_maps}
    if with_pycrostates:
        from msvae.baseline import modkmeans_group
        from msvae.pipeline import build_bank, info_from_record
        bank, _ = build_bank(records, image_size=32)
        info = info_from_record(records[0])
        try:
            maps_by_method["pycrostates"] = modkmeans_group(bank, info, k, seed=seed)
        except Exception as exc:
            print(f"    pycrostates indisponible : {exc}", flush=True)

    for name, maps in maps_by_method.items():
        params = []
        for rec in records:
            seg = microstates.backfit(rec.data.astype(np.float64), maps, rec.sfreq,
                                      min_segment_ms=min_segment_ms)
            params.append(microstates.microstate_parameters(seg, rec.boundaries))
        arms[name], _ = classify.features_from_params(params)

    for scope in ("global", "channel"):
        arms[f"spectral_{scope}"], _ = classify.spectral_features(records, scope=scope)
    return arms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["duration", "transition"], default="transition")
    p.add_argument("--levels", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--n-subjects", type=int, default=120)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--min-segment-ms", type=float, default=30.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--no-pycrostates", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = Path(a.out or f"results/sensitivity_{a.mode}")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    print(f"mode {a.mode}, {a.n_subjects} sujets, niveaux {a.levels}", flush=True)

    curves = {}
    for t in a.levels:
        cp = cohort_params(a.mode, t)
        print(f"[t={t:.2f}] {cp}", flush=True)
        t0 = time.time()
        records, gt_maps = iter_synthetic(n_subjects=a.n_subjects,
                                          duration=a.duration, seed=a.seed,
                                          verbose=False, **cp)
        y = np.array([r.group == "G2" for r in records], dtype=int)
        subjects = np.array([r.subject for r in records])
        arms = arms_for_cohort(records, gt_maps, a.k, a.min_segment_ms,
                               not a.no_pycrostates, a.seed)
        for name, X in arms.items():
            r = classify.evaluate_arm(X, y, subjects, n_splits=a.n_splits,
                                      n_repeats=a.n_repeats, seed=a.seed)
            curves.setdefault(name, []).append(r["auc"])
            print(f"    {name:20s} AUC = {r['auc']:.3f}", flush=True)
        print(f"    ({time.time() - t0:.0f}s)", flush=True)

    payload = dict(mode=a.mode, levels=a.levels, n_subjects=a.n_subjects,
                   duration=a.duration, curves=curves)
    (out / "sensitivity.json").write_text(json.dumps(payload, indent=2))
    _plot(payload, out / "figures" / f"sensitivity_{a.mode}.png")
    print(f"ok -> {out}")


def _plot(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    colors = {"ground_truth": "#27ae60", "pycrostates": "#2c6fbb",
              "spectral_global": "#7f8c8d", "spectral_channel": "#bdc3c7"}
    for name, aucs in payload["curves"].items():
        ax.plot(payload["levels"], aucs, marker="o", ms=5, label=name,
                color=colors.get(name), lw=2 if "spectral" not in name else 1.5,
                ls="-" if "spectral" not in name else "--")
    ax.axhline(0.5, color="#c0392b", ls=":", lw=1)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel(f"amplitude de l'effet de groupe ({payload['mode']}) "
                  f"— 1.0 = cohorte de reference")
    ax.set_ylabel("AUC")
    ax.set_title(f"Sensibilite, effet « {payload['mode']} », "
                 f"{payload['n_subjects']} sujets", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
