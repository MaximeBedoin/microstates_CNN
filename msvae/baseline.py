"""Pipeline Pycrostates classique (modified k-means), pour comparaison.

Le point important pour l'equite de la comparaison : la baseline recoit
EXACTEMENT les memes pics de GFP que le pipeline VAE (memes sujets, memes
epochs, memes indices temporels), et le meme K.
"""

from __future__ import annotations

import numpy as np


def _chdata(topo: np.ndarray, info):
    """(n_peaks, n_ch) -> ChData Pycrostates."""
    from pycrostates.io import ChData

    return ChData(np.ascontiguousarray(topo.T, dtype=np.float64), info)


def modkmeans_maps(topo: np.ndarray, info, k: int, seed: int = 0,
                   n_init: int = 100, max_iter: int = 300) -> np.ndarray:
    """Modified k-means sur un ensemble de topographies -> (k, n_ch) normalisees."""
    from pycrostates.cluster import ModKMeans

    km = ModKMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, tol=1e-6,
                   random_state=seed)
    km.fit(_chdata(topo, info), picks="eeg", verbose="error")
    maps = np.asarray(km.cluster_centers_, dtype=np.float64)
    maps -= maps.mean(axis=1, keepdims=True)
    return maps / np.maximum(np.linalg.norm(maps, axis=1, keepdims=True), 1e-12)


def modkmeans_group(bank, info, k: int, seed: int = 0, two_stage: bool = True,
                    n_init: int = 100) -> np.ndarray:
    """Cartes de groupe facon Pycrostates.

    two_stage=True : ModKMeans par sujet puis ModKMeans sur les cartes
    individuelles concatenees (workflow standard des tutoriels Pycrostates).
    two_stage=False : un seul ModKMeans sur tous les pics de tous les sujets.
    """
    if not two_stage:
        return modkmeans_maps(bank.topo, info, k, seed, n_init)

    indiv = []
    for s in bank.subjects:
        sel = bank.subject == s
        if sel.sum() < k:
            continue
        indiv.append(modkmeans_maps(bank.topo[sel], info, k, seed,
                                    n_init=max(10, n_init // 5)))
    indiv = np.concatenate(indiv, axis=0)
    return modkmeans_maps(indiv, info, k, seed, n_init)
