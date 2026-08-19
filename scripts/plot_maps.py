#!/usr/bin/env python
"""Regenere la figure comparative des cartes a partir du maps.npz COURANT.

La figure produite par le pipeline date de son execution : elle ne contient
donc ni les bras ajoutes ensuite (VaDE), ni les bras re-entraines depuis (le
token). Ce script la reconstruit a partir du fichier de cartes tel qu'il est,
ce qui evite de comparer visuellement des cartes qui ne sont plus celles des
chiffres rapportes.

Les cartes de chaque methode sont reordonnees et re-signees par appariement
hongrois sur une reference (le ground truth s'il existe, sinon `pycrostates`) :
sans cela les colonnes ne seraient pas comparables d'une rangee a l'autre,
l'ordre des classes et le signe etant arbitraires dans tout clustering de
microstates.

    python scripts/plot_maps.py --run results/synth_t035_transition
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import evaluate, plotting  # noqa: E402
from msvae.pipeline import (ExperimentConfig, info_from_record,  # noqa: E402
                            load_records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--out", default=None, help="defaut : <run>/figures/maps_courantes.png")
    p.add_argument("--reference", default=None,
                   help="methode servant de reference d'alignement")
    p.add_argument("--exclude", nargs="*", default=["weighted"],
                   help="motifs a exclure des rangees (defaut : variantes weighted)")
    a = p.parse_args()

    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    known = set(ExperimentConfig().to_dict())
    cfg = ExperimentConfig(**{k: v for k, v in res["config"].items() if k in known})
    records, _ = load_records(cfg)
    info = info_from_record(records[0])

    npz = np.load(run / "maps.npz")
    names = [n for n in npz.files
             if not any(pat in n for pat in a.exclude)]
    ref_name = a.reference or ("ground_truth" if "ground_truth" in names
                               else "pycrostates")
    if ref_name not in names:
        raise SystemExit(f"reference {ref_name} absente ({names})")
    reference = npz[ref_name]

    # la reference en premier, le reste par ordre alphabetique
    order = [ref_name] + sorted(n for n in names if n != ref_name)
    aligned = {}
    for n in order:
        m, _, corrs = evaluate.align_maps(npz[n], reference)
        label = n if n == ref_name else f"{n}\n|r|={np.mean(np.abs(corrs)):.2f}"
        aligned[label] = m

    out = Path(a.out) if a.out else run / "figures" / "maps_courantes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plotting.plot_maps_grid(aligned, info, out,
                            titles=[f"état {i + 1}" for i in range(len(reference))])
    print(f"reference d'alignement : {ref_name}")
    for n in order[1:]:
        _, _, c = evaluate.align_maps(npz[n], reference)
        print(f"  {n:24s} |r| moyen a la reference = {np.mean(np.abs(c)):.3f}")
    print(f"ok -> {out}")


if __name__ == "__main__":
    main()
