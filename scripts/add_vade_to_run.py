#!/usr/bin/env python
"""Ajoute le bras VaDE a un run DEJA calcule, sans rien recalculer d'autre.

Pourquoi ne pas relancer le pipeline : on veut comparer VaDE aux autres bras,
donc tout ce qui n'est pas VaDE doit etre rigoureusement identique — meme
cohorte, meme graine, memes pics, memes cartes des autres methodes. Repartir du
modele deja entraine du run garantit cela par construction, la ou un nouveau
run introduirait du bruit d'initialisation (mesure : 0.03 a 0.065 de GEV entre
graines, soit plus que la plupart des effets recherches).

Le bras VaDE part de l'encodeur d'un bras existant et n'en change que le prior.
Le contraste `vade_<kind>` contre `vae_<kind>_*` isole donc l'effet de
l'objectif clustering-aware, et rien d'autre.

    python scripts/add_vade_to_run.py --run results/synth_t035_transition
    python scripts/run_classification.py --run results/synth_t035_transition
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import vade  # noqa: E402
from msvae.models import VAEConfig, build_model  # noqa: E402
from msvae.pipeline import (ExperimentConfig, build_bank, load_records)  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--kind", choices=["conv", "dense", "token"], default="dense")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-balance", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--k", type=int, default=None, help="defaut : le K du run")
    a = p.parse_args()

    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    known = set(ExperimentConfig().to_dict())
    cfg = ExperimentConfig(**{k: v for k, v in res["config"].items() if k in known})
    k = a.k or cfg.k
    print(f"run {run} | cohorte {cfg.dataset}, graine {cfg.seed} | K={k}", flush=True)

    records, _ = load_records(cfg)
    bank, _ = build_bank(records, cfg.image_size)
    _, val_bank = bank.split_subjects(cfg.val_frac, seed=cfg.seed)
    full_bal = bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                             seed=cfg.seed)
    val_bal = val_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                seed=cfg.seed)

    ckpt = torch.load(run / f"model_{a.kind}.pt", weights_only=False)
    mcfg = VAEConfig(**ckpt["cfg"])
    if isinstance(mcfg.elec_pos, list):          # JSON rend des listes
        mcfg = VAEConfig(**{**ckpt["cfg"],
                            "elec_pos": tuple(map(tuple, ckpt["cfg"]["elec_pos"]))})
    base = build_model(mcfg)
    base.load_state_dict(ckpt["state"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base.to(dev)
    print(f"  encodeur de depart : {a.kind} ({base.n_params()} parametres) "
          f"sur {dev}", flush=True)

    x = full_bal.images if a.kind == "conv" else full_bal.topo
    xv = val_bal.images if a.kind == "conv" else val_bal.topo
    mask = bank.projector.mask if a.kind == "conv" else None

    model, hist = vade.fit_vade(base, x, xv, n_components=k, epochs=a.epochs,
                                lr=a.lr, lambda_balance=a.lambda_balance,
                                batch_size=a.batch_size, mask=mask, seed=cfg.seed)

    maps = vade.component_maps(model, x)
    if maps.ndim > 2:
        maps = bank.projector.to_topo(maps[:, 0] * bank.image_scale, method="ridge")
        maps = maps - maps.mean(axis=1, keepdims=True)
        maps = maps / np.maximum(np.linalg.norm(maps, axis=1, keepdims=True), 1e-12)

    name = f"vade_{a.kind}"
    npz = dict(np.load(run / "maps.npz"))
    npz[name] = maps
    np.savez_compressed(run / "maps.npz", **npz)

    weights = torch.softmax(model.prior.pi_logits.detach(), 0).cpu().numpy()
    alive = int((weights > 0.01).sum())
    payload = dict(base_kind=a.kind, n_components=k, epochs=a.epochs, lr=a.lr,
                   lambda_balance=a.lambda_balance,
                   weights=weights.tolist(), components_alive=alive,
                   val_elbo=hist[-1].get("val_elbo"),
                   balance=hist[-1].get("train_balance"),
                   balance_max=hist[-1].get("train_balance_max"))
    (run / "vade.json").write_text(json.dumps(payload, indent=2))
    torch.save(dict(state=model.state_dict(), cfg=ckpt["cfg"], n_components=k),
               run / "model_vade.pt")

    print(f"\n  poids des composantes : {np.round(weights, 3).tolist()}")
    print(f"  composantes vivantes (poids > 1%) : {alive}/{k}")
    print(f"  -ELBO validation : {payload['val_elbo']:.4f}")
    print(f"ok -> {name} ajoute a {run}/maps.npz")


if __name__ == "__main__":
    main()
