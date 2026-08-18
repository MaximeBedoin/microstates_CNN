#!/usr/bin/env python
"""Telechargement de ds004504 (AD / FTD / temoins, 19 canaux, repos yeux fermes).

Miltiadous et al., Data 8(6):95, 2023 - https://openneuro.org/datasets/ds004504

On ne recupere QUE les donnees brutes (`sub-*`), pas les derivees : le
prepretraitement du depot (filtrage, ASR, ICA) ferait double emploi avec
`msvae/preprocess.py` et rendrait l'effet du notre ininterpretable.
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cache/ds004504")
    p.add_argument("--include-derivatives", action="store_true",
                   help="recupere aussi derivatives/ (non recommande, voir docstring)")
    a = p.parse_args()

    import openneuro

    # `sub-*` seul remonterait AUSSI derivatives/sub-* (le motif est applique
    # recursivement), soit 5.4 Go au lieu de 2.7 : on ancre donc a la racine.
    include = ["participants.tsv", "dataset_description.json", "/sub-*"]
    if a.include_derivatives:
        include.append("derivatives")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    openneuro.download(dataset="ds004504", target_dir=out, include=include)
    print(f"ok -> {out}")


if __name__ == "__main__":
    main()
