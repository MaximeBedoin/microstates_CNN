#!/usr/bin/env python
"""Trace la GEV aval en fonction de l'epoque, pour tous les bras d'un run.

C'est la figure du test de H2 en version controlee : trois architectures, meme
budget d'epoques, meme objectif (`loss_space=topo` met les trois encodeurs en
espace capteur), et une seule variable qui bouge — la duree d'entrainement.

Le panneau de gauche montre la reconstruction, qui s'ameliore pour tous ; celui
de droite la qualite aval, qui ne suit pas. L'ecart entre les deux panneaux EST
le desalignement d'objectif.

    python scripts/plot_downstream_curves.py --run results/synth_t035_transition
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLORS = {"conv": "#c0392b", "dense": "#2c6fbb", "token": "#27ae60"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    # les sidecars font foi quand ils existent : results.json a pu etre ecrase
    # par un autre bras (cf. la condition de course documentee dans
    # retrain_arm_in_run.py)
    models = dict(res["models"])
    for sc in run.glob("retrain_*.json"):
        models[sc.stem.replace("retrain_", "")] = json.loads(sc.read_text())

    curves = {k: v["downstream_curve"] for k, v in models.items()
              if v.get("downstream_curve")}
    if not curves:
        raise SystemExit("aucune courbe : relancer avec --score-every")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for arm, c in sorted(curves.items()):
        ep = [x[0] for x in c]
        gev = [x[1] for x in c]
        col = COLORS.get(arm, "#7f8c8d")
        r = models[arm].get("retrained", {})
        axes[1].plot(ep, gev, marker="o", ms=3.5, color=col, lw=1.8,
                     label=f"{arm} ({gev[0]:.3f} → {gev[-1]:.3f})")
        axes[1].axhline(max(gev), color=col, ls=":", lw=0.8, alpha=0.5)
        axes[0].plot([0, ep[-1]],
                     [models[arm]["final"].get("val_recon", np.nan)] * 2,
                     color=col, ls="--", lw=0.9, alpha=0.5)
        axes[0].scatter([r.get("best_epoch", ep[-1])], [r.get("recon_best")],
                        color=col, s=45, zorder=3,
                        label=f"{arm} : {r.get('recon_best', float('nan')):.4f}")

    axes[0].set_xlabel("époque")
    axes[0].set_ylabel("meilleure erreur de reconstruction")
    axes[0].set_title("reconstruction : tous s'améliorent", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("époque")
    axes[1].set_ylabel("GEV aval (sujets de validation)")
    axes[1].set_title("qualité aval : un seul bras se dégrade", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Désalignement d'objectif à budget et objectif identiques",
                 fontsize=11)
    fig.tight_layout()
    out = Path(a.out) if a.out else run / "figures" / "downstream_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    for arm, c in sorted(curves.items()):
        g = [x[1] for x in c]
        print(f"  {arm:6s} max {max(g):.4f} -> final {g[-1]:.4f}  "
              f"(chute {max(g) - g[-1]:+.4f})")
    print(f"ok -> {out}")


if __name__ == "__main__":
    main()
