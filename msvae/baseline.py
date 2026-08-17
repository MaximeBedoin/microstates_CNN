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


def pca_compress(topo: np.ndarray, n_components: int) -> tuple[np.ndarray, object]:
    """Compression LINEAIRE de rang `n_components` des topographies.

    Bras de reference de l'ablation : meme dimension de goulot que le VAE,
    mais obtenue par une projection lineaire, et clustering inchange (modified
    k-means sur les topographies reconstruites). L'ecart avec le VAE dense
    mesure alors l'apport de la non-linearite, et l'ecart VAE dense / VAE conv
    celui du biais inductif convolutionnel.

    On utilise une SVD sans centrage : les topographies normalisees par la GFP
    contiennent deja x et -x de facon symetrique, leur moyenne est ~0, et
    centrer introduirait une composante artificielle liee a la polarite.
    """
    x = np.asarray(topo, dtype=np.float64)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:n_components]                      # (n_components, n_ch)
    recon = (x @ basis.T) @ basis
    explained = float((s[:n_components] ** 2).sum() / (s ** 2).sum())
    return recon, dict(basis=basis, explained_variance_ratio=explained)


def pca_modkmeans_group(bank, info, k: int, n_components: int = 8, seed: int = 0,
                        two_stage: bool = True, n_init: int = 100):
    """Pipeline classique applique aux topographies compressees lineairement."""
    recon, meta = pca_compress(bank.topo, n_components)
    compressed = bank.subset(np.arange(len(bank)))
    compressed.topo = recon.astype(np.float32)
    maps = modkmeans_group(compressed, info, k, seed=seed, two_stage=two_stage,
                           n_init=n_init)
    return maps, meta


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
