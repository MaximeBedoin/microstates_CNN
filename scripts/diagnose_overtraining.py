#!/usr/bin/env python
"""La reconstruction et la qualite des microstates divergent-elles ?

Entraine les deux bras (conv sur images, dense sur vecteurs) avec la meme
configuration, en evaluant a intervalle regulier le critere AVAL (GEV apres
clustering latent, decodage et back-fitting sur les sujets de validation).

Motivation : la meme configuration conv obtenait GEV = 0.695 apres 25 epoques
et 0.622 apres 60 epoques, alors que son erreur de reconstruction continuait
de baisser. Ce script mesure directement les deux courbes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import plotting  # noqa: E402
from msvae.models import VAEConfig, match_dense_to_conv  # noqa: E402
from msvae.pipeline import (ExperimentConfig, build_bank, load_records,  # noqa: E402
                            make_downstream_score)
from msvae.train import TrainConfig, train_vae  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-subjects", type=int, default=20)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--beta", type=float, default=1e-3)
    p.add_argument("--score-every", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/diagnostics")
    a = p.parse_args()

    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cfg = ExperimentConfig(dataset="synthetic", n_subjects=a.n_subjects,
                           duration=a.duration, latent_dim=a.latent_dim)
    records, _ = load_records(cfg)
    bank, info = build_bank(records, cfg.image_size)
    train_bank, val_bank = bank.split_subjects(cfg.val_frac, seed=cfg.seed)
    train_bal = train_bank.balanced(seed=cfg.seed)
    val_bal = val_bank.balanced(seed=cfg.seed)
    val_records = [r for r in records if r.subject in set(val_bank.subjects)]
    score_fn = make_downstream_score(val_bal, val_records, cfg.k,
                                     cfg.min_segment_ms, cfg.seed)

    conv_cfg = VAEConfig(kind="conv", latent_dim=a.latent_dim, beta=a.beta,
                         image_size=cfg.image_size, n_channels_eeg=bank.topo.shape[1])
    dense_cfg = match_dense_to_conv(conv_cfg, bank.topo.shape[1])
    tcfg = TrainConfig(epochs=a.epochs, seed=cfg.seed, num_threads=a.threads,
                       score_every=a.score_every, patience=10 ** 6, verbose=True)

    history = {}
    for kind, mcfg, x, mask in (("conv", conv_cfg, train_bal.images, bank.projector.mask),
                                ("dense", dense_cfg, train_bal.topo, None)):
        xv = val_bal.images if kind == "conv" else val_bal.topo
        print(f"=== {kind} ===", flush=True)
        res = train_vae(mcfg, x, xv, mask, tcfg, score_fn=score_fn)
        history[kind] = res.history
        scored = [(h["epoch"], -h["down_score"], h["val_recon"])
                  for h in res.history if "down_score" in h]
        for ep, gev, rec in scored:
            print(f"  ep{ep:3d}  recon={rec:.4f}  GEV_aval={gev:.4f}", flush=True)
        best = max(scored, key=lambda t: t[1])
        print(f"  -> meilleure GEV aval {best[1]:.4f} a l'epoque {best[0]}"
              f" (derniere epoque : {scored[-1][1]:.4f})", flush=True)

    (out / "overtraining.json").write_text(json.dumps(history, indent=2))
    _plot(history, out / "figures" / "overtraining.png")
    print(f"ok -> {out}")


def _plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    colors = {"conv": "#c0392b", "dense": "#2c6fbb"}
    for kind, hist in history.items():
        ep = [h["epoch"] for h in hist]
        axes[0].plot(ep, [h["val_recon"] for h in hist], color=colors[kind],
                     label=f"{kind}")
        sc = [(h["epoch"], -h["down_score"]) for h in hist if "down_score" in h]
        axes[1].plot([s[0] for s in sc], [s[1] for s in sc], "o-",
                     color=colors[kind], label=f"{kind}", ms=4)
    axes[0].set_xlabel("epoque")
    axes[0].set_ylabel("erreur de reconstruction (validation)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("epoque")
    axes[1].set_ylabel("GEV aval (sujets de validation)")
    axes[1].legend(fontsize=9)
    fig.suptitle("Reconstruction vs qualite des microstates au fil de l'entrainement",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
