"""Appariement de cartes, stabilite, et cartes canoniques de reference."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------- appariements
def abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|correlation spatiale| entre deux jeux de cartes, (na, nb)."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-20)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-20)
    return np.abs(a @ b.T)


def match_maps(a: np.ndarray, b: np.ndarray):
    """Apparie deux jeux de cartes (Hungarian sur |r|).

    Returns
    -------
    perm : array (na,)   indice de la carte de `b` appariee a chaque carte de `a`
    corrs : array (na,)  |r| de chaque paire
    mean_corr : float
    """
    c = abs_corr_matrix(a, b)
    ri, ci = linear_sum_assignment(-c)
    perm = np.full(len(a), -1, dtype=int)
    corrs = np.zeros(len(a))
    for i, j in zip(ri, ci):
        perm[i] = j
        corrs[i] = c[i, j]
    return perm, corrs, float(corrs[corrs > 0].mean())


def align_maps(maps: np.ndarray, reference: np.ndarray):
    """Reordonne et re-signe `maps` pour coller a `reference`."""
    perm, corrs, _ = match_maps(reference, maps)
    out = maps[perm].copy()
    for i in range(len(out)):
        r = np.corrcoef(reference[i], out[i])[0, 1]
        if r < 0:
            out[i] *= -1
    return out, perm, corrs


# ------------------------------------------------- cartes canoniques (proxy)
def canonical_templates(pos3d: np.ndarray) -> tuple[np.ndarray, list]:
    """Approximations geometriques des microstates canoniques A/B/C/D.

    Les cartes publiees (Koenig et al. 2002) ne sont pas telechargeables ici ;
    on construit donc des proxys geometriques a partir des coordonnees des
    electrodes, fideles aux descriptions classiques :
      A : gradient diagonal gauche-post <-> droite-ant
      B : gradient diagonal droite-post <-> gauche-ant
      C : gradient posterieur <-> anterieur (midline)
      D : maximum fronto-central (radial)
    A utiliser comme sanity check grossier, pas comme verite terrain.
    """
    p = np.asarray(pos3d, dtype=float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    z = (z - z.mean()) / z.std()
    a = (x + y) / np.sqrt(2)
    b = (-x + y) / np.sqrt(2)
    c = y
    # bump fronto-central : distance a un point frontal-central sur le scalp
    target = np.array([0.0, 0.6, 1.2])
    d = -np.linalg.norm(np.stack([x, y, z], axis=1) - target, axis=1)
    maps = np.stack([a, b, c, d])
    maps -= maps.mean(axis=1, keepdims=True)
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)
    return maps, ["A", "B", "C", "D"]


def sanity_check_canonical(maps: np.ndarray, pos3d: np.ndarray) -> dict:
    """Apparie les cartes obtenues aux proxys canoniques A/B/C/D."""
    tmpl, names = canonical_templates(pos3d)
    perm, corrs, mean_corr = match_maps(tmpl, maps)
    return dict(names=names, perm=perm, corrs=corrs, mean_corr=mean_corr)


# ----------------------------------------------------------------- stabilite
def split_half_stability(fit_fn, subjects: np.ndarray, n_repeats: int = 10,
                         seed: int = 0) -> dict:
    """Stabilite des prototypes entre sous-echantillonnages de sujets.

    `fit_fn(subject_subset) -> maps (k, n_ch)` est appelee sur deux moities
    disjointes de la cohorte ; on rapporte le |r| moyen apres appariement.
    """
    rng = np.random.default_rng(seed)
    subs = np.unique(subjects)
    scores, per_class = [], []
    for _ in range(n_repeats):
        perm = rng.permutation(subs)
        half = len(subs) // 2
        m1 = fit_fn(perm[:half])
        m2 = fit_fn(perm[half:2 * half])
        _, corrs, mean_corr = match_maps(m1, m2)
        scores.append(mean_corr)
        per_class.append(np.sort(corrs)[::-1])
    return dict(mean=float(np.mean(scores)), std=float(np.std(scores)),
                scores=np.array(scores), per_class=np.array(per_class))


# ------------------------------------------------------- comparaison groupes
def group_comparison(params_by_subject: list[dict], groups: np.ndarray,
                     keys=("mean_duration_ms", "occurrence_per_s", "coverage"),
                     subjects: np.ndarray | None = None):
    """Comparaison de deux groupes, par classe et par parametre.

    Si `subjects` est fourni et que chaque sujet apparait exactement une fois
    dans chaque groupe (cas yeux ouverts / yeux fermes), le test est
    **apparie** (Wilcoxon signed-rank), nettement plus puissant. Sinon
    Mann-Whitney. La taille d'effet (rank-biserial) est plus informative que p
    sur de petits echantillons.
    """
    from scipy.stats import mannwhitneyu, wilcoxon

    groups = np.asarray(groups)
    g = np.unique(groups)
    if len(g) != 2:
        raise ValueError("group_comparison attend exactement deux groupes")

    paired_subs = None
    if subjects is not None:
        subjects = np.asarray(subjects)
        s0 = list(subjects[groups == g[0]])
        s1 = list(subjects[groups == g[1]])
        if len(s0) == len(set(s0)) == len(s1) == len(set(s1)) and set(s0) == set(s1):
            paired_subs = sorted(set(s0))

    out = {}
    for key in keys:
        vals = np.array([p[key] for p in params_by_subject], dtype=float)
        k = vals.shape[1]
        stats = []
        for c in range(k):
            if paired_subs is not None:
                i0 = {s: i for i, s in zip(np.flatnonzero(groups == g[0]),
                                           subjects[groups == g[0]])}
                i1 = {s: i for i, s in zip(np.flatnonzero(groups == g[1]),
                                           subjects[groups == g[1]])}
                a = np.array([vals[i0[s], c] for s in paired_subs])
                b = np.array([vals[i1[s], c] for s in paired_subs])
                ok = np.isfinite(a) & np.isfinite(b)
                a, b = a[ok], b[ok]
                if len(a) < 5 or np.allclose(a, b):
                    stats.append((np.nan, np.nan, np.nan))
                    continue
                stat, p = wilcoxon(a, b)
                eff = float(np.mean(np.sign(b - a)))  # proportion signee
                stats.append((float(stat), float(p), eff))
            else:
                a = vals[groups == g[0], c]
                b = vals[groups == g[1], c]
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if len(a) < 3 or len(b) < 3:
                    stats.append((np.nan, np.nan, np.nan))
                    continue
                u, p = mannwhitneyu(a, b, alternative="two-sided")
                stats.append((float(u), float(p), float(2 * u / (len(a) * len(b)) - 1)))
        out[key] = dict(groups=tuple(g), stats=np.array(stats),
                        paired=paired_subs is not None,
                        n=len(paired_subs) if paired_subs else None)
    return out
