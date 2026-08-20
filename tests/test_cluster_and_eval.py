import numpy as np

from msvae.cluster import latent_kmeans
from msvae.evaluate import (abs_corr_matrix, align_maps, canonical_templates,
                            match_maps)


def _latent_with_clusters(n_subj=6, per_cluster=40, k=4, d=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d)) * 4
    z, subj = [], []
    for s in range(n_subj):
        n = per_cluster * (1 + s)  # sujets tres desequilibres
        lab = rng.integers(0, k, n)
        z.append(centers[lab] + rng.standard_normal((n, d)) * 0.4)
        subj += [f"sub{s}"] * n
    return np.concatenate(z), np.array(subj), centers


def test_latent_kmeans_two_stage_recovers_centers():
    z, subj, centers = _latent_with_clusters()
    got, labels, info = latent_kmeans(z, subj, 4, mode="two_stage", seed=0)
    d = np.linalg.norm(got[:, None] - centers[None], axis=-1)
    assert d.min(axis=1).max() < 1.0
    assert info["mode"] == "two_stage"


def test_latent_kmeans_weighted_recovers_centers():
    z, subj, centers = _latent_with_clusters()
    got, labels, _ = latent_kmeans(z, subj, 4, mode="weighted", seed=0)
    d = np.linalg.norm(got[:, None] - centers[None], axis=-1)
    assert d.min(axis=1).max() < 1.0
    assert len(labels) == len(z)


def test_match_maps_is_permutation_and_sign_invariant():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 30))
    a -= a.mean(1, keepdims=True)
    perm = [2, 0, 3, 1]
    b = a[perm] * np.array([1, -1, 1, -1])[:, None]
    idx, corrs, mean_corr = match_maps(a, b)
    assert mean_corr > 0.999
    assert list(np.argsort(idx)) == list(np.argsort(np.argsort(perm)))


def test_align_maps_fixes_sign():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((3, 20))
    ref -= ref.mean(1, keepdims=True)
    shuffled = ref[[1, 2, 0]] * np.array([-1, 1, -1])[:, None]
    aligned, _, corrs = align_maps(shuffled, ref)
    # apres alignement sur la reference, correlations positives et ~1
    for i in range(3):
        assert np.corrcoef(ref[i], aligned[i])[0, 1] > 0.99


def test_canonical_templates_are_distinct():
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 1.2, 40)
    phi = rng.uniform(0, 2 * np.pi, 40)
    pos = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)], axis=1)
    maps, names = canonical_templates(pos)
    assert names == ["A", "B", "C", "D"]
    c = abs_corr_matrix(maps, maps)
    np.fill_diagonal(c, 0)
    assert c.max() < 0.85


def test_select_subjects_est_stratifie():
    """ds004504 est ordonné par groupe : tronquer la liste globale rendait une
    cohorte mono-classe, et l'échec n'apparaissait qu'après l'entraînement."""
    from msvae.data import select_subjects

    meta = {f"sub-{i:03d}": {"group": "AD"} for i in range(1, 37)}
    meta |= {f"sub-{i:03d}": {"group": "FTD"} for i in range(37, 60)}
    meta |= {f"sub-{i:03d}": {"group": "CTR"} for i in range(60, 89)}

    sel = select_subjects(meta, ("AD", "CTR"), max_subjects=20)
    groupes = [meta[s]["group"] for s in sel]
    assert groupes.count("AD") == 10 and groupes.count("CTR") == 10

    # sans troncature, tous les sujets des groupes demandés
    assert len(select_subjects(meta, ("AD", "CTR"))) == 65
    # les groupes non demandés ne sont jamais inclus
    assert all(meta[s]["group"] != "FTD"
               for s in select_subjects(meta, ("AD", "CTR"), 40))
