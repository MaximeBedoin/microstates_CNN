#!/usr/bin/env python
"""Re-entraine le bras token d'un run avec le regime d'optimisation correct.

Les runs anterieurs au correctif ont evalue un bras token qui n'avait rien
appris : au taux d'apprentissage des bras conv et dense (2e-3), sa
reconstruction reste bloquee a 0.754 pendant ~30 epoques, et `patience=10`
declenche l'arret en plein plateau. Signature reconnaissable : une stabilite
split-half de 1.000 +/- 0.000, un modele fige etant parfaitement reproductible.

Ce script ne recalcule QUE le bras token et remplace ses cartes dans le
maps.npz du run. Tout le reste — cohorte, graine, pics, cartes des autres
methodes — demeure identique, ce qui preserve la comparabilite. Un run complet
reintroduirait du bruit d'initialisation (0.03 a 0.065 de GEV entre graines).

    python scripts/retrain_token_in_run.py --run results/synth_t035_transition
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import cluster  # noqa: E402
from msvae.models import VAEConfig, match_token_to_conv  # noqa: E402
from msvae.pipeline import (ExperimentConfig, _electrode_positions,  # noqa: E402
                            build_bank, load_records)
from msvae.train import TrainConfig, train_vae  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma-init", type=float, default=0.693)
    p.add_argument("--batch-size", type=int, default=1024)
    a = p.parse_args()

    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    known = set(ExperimentConfig().to_dict())
    cfg = ExperimentConfig(**{k: v for k, v in res["config"].items() if k in known})
    print(f"run {run} | graine {cfg.seed} | lr={a.lr:g}, {a.epochs} epoques, "
          f"sans early stopping", flush=True)

    records, _ = load_records(cfg)
    bank, _ = build_bank(records, cfg.image_size)
    _, val_bank = bank.split_subjects(cfg.val_frac, seed=cfg.seed)
    full_bal = bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                             seed=cfg.seed)
    val_bal = val_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                seed=cfg.seed)
    n_ch = bank.topo.shape[1]

    conv_ref = VAEConfig(kind="conv", latent_dim=cfg.latent_dim,
                         image_size=cfg.image_size, beta=1e-3,
                         n_channels_eeg=n_ch)
    tcfg_model = match_token_to_conv(
        conv_ref, n_ch, elec_pos=_electrode_positions(records[0], n_ch),
        n_heads=cfg.token_heads, n_layers=cfg.token_layers)
    tcfg_model = VAEConfig(**{**tcfg_model.to_dict(), "gamma_init": a.gamma_init})

    tcfg = TrainConfig(epochs=a.epochs, seed=cfg.seed, device="auto",
                       batch_size=a.batch_size, verbose=False,
                       patience=10 ** 6, lr=a.lr)
    r = train_vae(tcfg_model, full_bal.topo, val_bal.topo, None, tcfg)
    rec = [h["val_recon"] for h in r.history]
    print(f"  reconstruction : {rec[0]:.4f} -> {min(rec):.4f} "
          f"(epoque {int(np.argmin(rec))})")
    if min(rec) > 0.5:
        print("  ATTENTION : le modele n'a pas quitte son plateau, "
              "resultats a ne pas interpreter", flush=True)
    print(f"  localite finale softplus(gamma) : "
          f"{r.model.learned_locality().mean():.3f}")

    npz = dict(np.load(run / "maps.npz"))
    for mode in ("two_stage", "weighted"):
        m = cluster.decoded_maps_from_bank(r.model, bank, cfg.k, mode=mode,
                                           seed=cfg.seed, kind="token")[0]
        npz[f"vae_token_{mode}"] = m
    np.savez_compressed(run / "maps.npz", **npz)
    torch.save(dict(state=r.model.state_dict(), cfg=tcfg_model.to_dict()),
               run / "model_token.pt")
    (run / "token_retrain.json").write_text(json.dumps(dict(
        lr=a.lr, epochs=a.epochs, gamma_init=a.gamma_init,
        recon_first=rec[0], recon_best=min(rec),
        best_epoch=int(np.argmin(rec)),
        learned_locality=r.model.learned_locality().tolist()), indent=2))
    print(f"ok -> vae_token_* remplaces dans {run}/maps.npz")


if __name__ == "__main__":
    main()
