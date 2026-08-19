#!/usr/bin/env python
"""Run bout-en-bout sur la cohorte synthetique (ground truth connu)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae.pipeline import ExperimentConfig, run_experiment  # noqa: E402
from msvae.train import TrainConfig, resolve_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=20)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--snr", type=float, default=1.0)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--no-arch-search", action="store_true")
    p.add_argument("--no-pycrostates", action="store_true")
    p.add_argument("--stability-refit", action="store_true")
    p.add_argument("--stability-repeats", type=int, default=10)
    p.add_argument("--loss-space", choices=["image", "topo"], default="image",
                   help="espace de la loss de reconstruction du VAE conv")
    p.add_argument("--select-epoch-by-score", action="store_true",
                   help="selectionne l'epoque sur la GEV aval, pas sur la loss")
    p.add_argument("--grid", choices=["default", "extended"], default="default")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="'auto' utilise le GPU s'il est disponible")
    p.add_argument("--batch-size", type=int, default=256,
                   help="256 convient au CPU ; 1024-2048 exploite mieux un GPU")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/synthetic")
    p.add_argument("--no-token", action="store_true",
                   help="desactive le bras attention sur electrodes")
    p.add_argument("--vade", action="store_true",
                   help="ajoute le bras VaDE (prior en melange de gaussiennes)")
    p.add_argument("--vade-kind", choices=["conv", "dense", "token"],
                   default="dense", help="encodeur de depart du bras VaDE")
    p.add_argument("--vade-epochs", type=int, default=40)
    # amplitude de l'effet de groupe. `--effect transition --effect-t 0.35`
    # place la cohorte dans la zone informative de la courbe de sensibilite :
    # ni au hasard, ni saturee, donc la seule ou les methodes se departagent.
    p.add_argument("--effect", choices=["reference", "duration", "transition"],
                   default="reference",
                   help="'transition' = durees appariees, le bras spectral n'a "
                        "aucune prise ; 'duration' = controle")
    p.add_argument("--effect-t", type=float, default=1.0,
                   help="amplitude dans [0, 1] ; 1.0 = effet de la cohorte de "
                        "reference, 0.0 = aucun effet")
    a = p.parse_args()

    # 1.0 = amplitude de la cohorte de reference (85 -> 65 ms, C->D x2.5)
    eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.065, trans_boost=2.5)
    if a.effect == "duration":
        eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.085 - a.effect_t * 0.020,
                   trans_boost=1.0)
    elif a.effect == "transition":
        eff = dict(mean_dur_g1=0.085, mean_dur_g2=0.085,
                   trans_boost=1.0 + a.effect_t * 1.5)
    if a.effect != "reference":
        print(f"effet de groupe « {a.effect} » a t={a.effect_t} : {eff}")

    cfg = ExperimentConfig(
        name="synthetic", dataset="synthetic", n_subjects=a.n_subjects,
        duration=a.duration, snr=a.snr, k=a.k, epochs=a.epochs,
        latent_dim=a.latent_dim, image_size=a.image_size,
        arch_search=not a.no_arch_search, run_pycrostates=not a.no_pycrostates,
        run_token=not a.no_token, run_vade=a.vade, vade_kind=a.vade_kind,
        vade_epochs=a.vade_epochs,
        stability_refit=a.stability_refit, stability_repeats=a.stability_repeats,
        loss_space=a.loss_space, select_epoch_by_score=a.select_epoch_by_score,
        grid=a.grid, seed=a.seed, out_dir=a.out, **eff)
    tcfg = TrainConfig(epochs=a.epochs, seed=a.seed, num_threads=a.threads,
                       device=a.device, batch_size=a.batch_size)
    print(f"peripherique : {resolve_device(tcfg.device)}  "
          f"(batch={tcfg.batch_size})")
    run_experiment(cfg, tcfg)


if __name__ == "__main__":
    main()
