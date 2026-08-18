#!/usr/bin/env python
"""Run bout-en-bout sur ds004504 (Alzheimer / demence fronto-temporale / temoins).

Miltiadous et al., Data 8(6):95, 2023. Repos yeux fermes, 19 canaux 10-20,
500 Hz, 36 AD / 23 FTD / 29 temoins.

Contraste par defaut : AD vs CTR, le plus simple a detecter. `--groups AD FTD`
donne le contraste difficile, utile comme second niveau de difficulte sur le
meme jeu de donnees.

Mises en garde, a garder en tete en lisant les resultats :

  * 19 electrodes contre 64 pour la cohorte synthetique. Rendre une image 32x32
    a partir de 19 capteurs rend la quasi-totalite des pixels interpolee : le
    bras conv est penalise pour une raison etrangere a l'hypothese testee, et
    les chiffres ne sont PAS comparables terme a terme a results/synthetic.
  * AD vs CTR, c'est 65 sujets non apparies. L'erreur-type d'une AUC y vaut
    ~0.054, celle d'une difference appariee ~0.03 : ce jeu de donnees dit si
    les microstates fonctionnent, pas quelle methode est la meilleure. Le
    classement doit venir de la courbe de sensibilite sur synthetique.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae.pipeline import ExperimentConfig, run_experiment  # noqa: E402
from msvae.train import TrainConfig, resolve_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs=2, default=["AD", "CTR"],
                   choices=["AD", "FTD", "CTR"])
    p.add_argument("--n-subjects", type=int, default=0,
                   help="0 = tous les sujets des groupes retenus")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--no-arch-search", action="store_true")
    p.add_argument("--no-pycrostates", action="store_true")
    p.add_argument("--no-token", action="store_true")
    p.add_argument("--stability-repeats", type=int, default=10)
    p.add_argument("--n-per-subject", type=int, default=1000)
    p.add_argument("--loss-space", choices=["image", "topo"], default="topo")
    p.add_argument("--select-epoch-by-score", action="store_true")
    p.add_argument("--grid", choices=["default", "extended"], default="default")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = a.out or f"results/ds004504_{a.groups[0]}_vs_{a.groups[1]}".lower()
    cfg = ExperimentConfig(
        name="ds004504", dataset="ds004504", ds_groups=tuple(a.groups),
        n_subjects=a.n_subjects, k=a.k, epochs=a.epochs,
        latent_dim=a.latent_dim, image_size=a.image_size,
        arch_search=not a.no_arch_search, run_pycrostates=not a.no_pycrostates,
        run_token=not a.no_token, stability_repeats=a.stability_repeats,
        n_per_subject=a.n_per_subject, loss_space=a.loss_space,
        select_epoch_by_score=a.select_epoch_by_score, grid=a.grid,
        seed=a.seed, out_dir=out)
    tcfg = TrainConfig(epochs=a.epochs, seed=a.seed, num_threads=a.threads,
                       device=a.device, batch_size=a.batch_size)
    print(f"peripherique : {resolve_device(tcfg.device)}  "
          f"(batch={tcfg.batch_size})  groupes : {a.groups}")
    run_experiment(cfg, tcfg)


if __name__ == "__main__":
    main()
