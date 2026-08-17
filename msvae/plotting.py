"""Figures : topographies, courbes d'entrainement, espace latent."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_maps(maps: np.ndarray, info, path, titles=None, row_label=None):
    """Une rangee de topographies."""
    import mne

    plt = _mpl()
    k = len(maps)
    fig, axes = plt.subplots(1, k, figsize=(2.1 * k, 2.6))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        mne.viz.plot_topomap(maps[i], info, axes=ax, show=False,
                             cmap="RdBu_r", contours=6, sensors=False)
        ax.set_title(titles[i] if titles else f"{i}", fontsize=10)
    if row_label:
        axes[0].set_ylabel(row_label)
        fig.suptitle(row_label, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_maps_grid(maps_by_method: dict, info, path, titles=None):
    """Une rangee par methode, colonnes appariees."""
    import mne

    plt = _mpl()
    methods = list(maps_by_method)
    k = len(next(iter(maps_by_method.values())))
    fig, axes = plt.subplots(len(methods), k,
                             figsize=(2.0 * k, 2.1 * len(methods)),
                             squeeze=False)
    for r, m in enumerate(methods):
        for c in range(k):
            ax = axes[r][c]
            mne.viz.plot_topomap(maps_by_method[m][c], info, axes=ax, show=False,
                                 cmap="RdBu_r", contours=6, sensors=False)
            if r == 0:
                ax.set_title(titles[c] if titles else f"K{c}", fontsize=10)
            if c == 0:
                ax.text(-0.35, 0.5, m, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_training(history, path, keys=("train_recon", "val_recon", "train_pol_rel")):
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ep = [h["epoch"] for h in history]
    for k in keys:
        if k in history[0]:
            ax[0].plot(ep, [h[k] for h in history], label=k)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("MSE / invariance")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    for k in ("train_kl", "val_kl"):
        if k in history[0]:
            ax[1].plot(ep, [h[k] for h in history], label=k)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("KL (nats)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latent(z: np.ndarray, labels: np.ndarray, path, title=""):
    """Projection PCA 2D de l'espace latent, colorees par cluster."""
    plt = _mpl()
    zc = z - z.mean(0)
    u, s, vt = np.linalg.svd(zc, full_matrices=False)
    p = zc @ vt[:2].T
    fig, ax = plt.subplots(figsize=(4.5, 4))
    n = min(len(p), 20000)
    sel = np.random.default_rng(0).choice(len(p), n, replace=False)
    ax.scatter(p[sel, 0], p[sel, 1], c=labels[sel], s=2, alpha=0.3,
               cmap="tab10", linewidths=0)
    ax.set_xlabel(f"PC1 ({100 * s[0] ** 2 / (s ** 2).sum():.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * s[1] ** 2 / (s ** 2).sum():.0f}%)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_images(images: np.ndarray, path, titles=None, cols=8):
    """Grille d'images topographiques brutes (verification visuelle)."""
    plt = _mpl()
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.4 * cols, 1.5 * rows),
                             squeeze=False)
    vmax = np.abs(images).max()
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      origin="lower")
            if titles:
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
