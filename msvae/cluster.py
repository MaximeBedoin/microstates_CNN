"""Clustering dans l'espace latent et decodage des centroides."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def _kmeans(x, k, seed, n_init=20, sample_weight=None):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    km.fit(x, sample_weight=sample_weight)
    return km


def latent_kmeans(z: np.ndarray, subject: np.ndarray, k: int,
                  mode: str = "two_stage", seed: int = 0, n_init: int = 20):
    """k-means dans l'espace latent.

    mode = 'two_stage'
        k-means par sujet (k prototypes chacun) puis k-means de groupe sur les
        prototypes individuels — la logique de Pycrostates, qui donne le meme
        poids a chaque sujet et amortit le bruit individuel.
    mode = 'weighted'
        un seul k-means sur tous les pics, chaque pic pondere par 1/n_pics du
        sujet — meme equilibrage entre sujets, mais garde la densite fine de
        l'espace latent.

    Returns
    -------
    centroids : (k, d)
    labels : (n,) affectation de chaque pic
    info : dict
    """
    z = np.asarray(z, dtype=np.float64)
    subs = np.unique(subject)

    if mode == "two_stage":
        protos, owner = [], []
        for s in subs:
            zi = z[subject == s]
            if len(zi) < k:
                continue
            km = _kmeans(zi, k, seed, n_init=max(5, n_init // 2))
            protos.append(km.cluster_centers_)
            owner += [s] * k
        protos = np.concatenate(protos, axis=0)
        km_group = _kmeans(protos, k, seed, n_init=n_init)
        centroids = km_group.cluster_centers_
        info = dict(mode=mode, n_protos=len(protos),
                    proto_labels=km_group.labels_, proto_subject=np.array(owner),
                    inertia=km_group.inertia_)
    elif mode == "weighted":
        counts = {s: int((subject == s).sum()) for s in subs}
        w = np.array([1.0 / counts[s] for s in subject], dtype=np.float64)
        w *= len(z) / w.sum()
        km = _kmeans(z, k, seed, n_init=n_init, sample_weight=w)
        centroids = km.cluster_centers_
        info = dict(mode=mode, inertia=km.inertia_)
    else:
        raise ValueError(f"mode inconnu : {mode}")

    labels = np.argmin(
        ((z[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
    return centroids, labels, info


def decode_centroids(model, centroids: np.ndarray, projector=None,
                     kind: str = "conv", image_scale: float = 1.0) -> np.ndarray:
    """Decode les centroides latents en topographies (k, n_ch).

    Les cartes sont re-referencees en moyenne et normalisees (norme 1), comme
    les cartes de microstates de la litterature.
    """
    out = model.decode_numpy(np.asarray(centroids, dtype=np.float32))
    if kind == "conv":
        imgs = out[:, 0] * image_scale
        maps = projector.to_topo(imgs, method="ridge")
    else:
        maps = out
    maps = maps - maps.mean(axis=1, keepdims=True)
    return maps / np.maximum(np.linalg.norm(maps, axis=1, keepdims=True), 1e-12)


def decoded_maps_from_bank(model, bank, k, mode="two_stage", seed=0,
                           kind="conv", n_init=20):
    """Encode la banque de pics, clusterise, decode : renvoie (maps, z, labels, info)."""
    x = bank.get(kind)
    z = model.encode_numpy(x)
    centroids, labels, info = latent_kmeans(z, bank.subject, k, mode=mode,
                                            seed=seed, n_init=n_init)
    maps = decode_centroids(model, centroids, bank.projector, kind,
                            bank.image_scale)
    return maps, z, labels, info
