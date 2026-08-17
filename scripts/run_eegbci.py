#!/usr/bin/env python
"""Run bout-en-bout sur EEGBCI (PhysioNet eegmmidb, miroir S3).

Chaque sujet fournit deux enregistrements de repos : yeux ouverts (run 1,
groupe EO) et yeux fermes (run 2, groupe EC). Le contraste EO/EC sert de
substitut au contraste clinique pour le volet "pouvoir discriminant".
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae.pipeline import ExperimentConfig, run_experiment  # noqa: E402
from msvae.train import TrainConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=40)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--no-arch-search", action="store_true")
    p.add_argument("--no-pycrostates", action="store_true")
    p.add_argument("--stability-repeats", type=int, default=10)
    p.add_argument("--n-per-subject", type=int, default=None)
    p.add_argument("--loss-space", choices=["image", "topo"], default="image",
                   help="espace de la loss de reconstruction du VAE conv")
    p.add_argument("--select-epoch-by-score", action="store_true",
                   help="selectionne l'epoque sur la GEV aval, pas sur la loss")
    p.add_argument("--grid", choices=["default", "extended"], default="default")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/eegbci")
    a = p.parse_args()

    cfg = ExperimentConfig(
        name="eegbci", dataset="eegbci", n_subjects=a.n_subjects, k=a.k,
        epochs=a.epochs, latent_dim=a.latent_dim, image_size=a.image_size,
        arch_search=not a.no_arch_search, run_pycrostates=not a.no_pycrostates,
        stability_repeats=a.stability_repeats, n_per_subject=a.n_per_subject,
        loss_space=a.loss_space, select_epoch_by_score=a.select_epoch_by_score,
        grid=a.grid,
        seed=a.seed, out_dir=a.out)
    run_experiment(cfg, TrainConfig(epochs=a.epochs, seed=a.seed,
                                   num_threads=a.threads))


if __name__ == "__main__":
    main()
