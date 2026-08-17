"""Back-projection des cartes sur les donnees et parametres des microstates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------- backfit
def spatial_correlation(data: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """Correlation spatiale entre chaque echantillon et chaque carte.

    data : (n_ch, n_times) — suppose re-reference en moyenne
    maps : (k, n_ch)
    -> (k, n_times)
    """
    d = data - data.mean(axis=0, keepdims=True)
    m = maps - maps.mean(axis=1, keepdims=True)
    d = d / np.maximum(np.linalg.norm(d, axis=0, keepdims=True), 1e-20)
    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-20)
    return m @ d


def _reject_short_segments(labels: np.ndarray, corr_abs: np.ndarray,
                           min_samples: int) -> np.ndarray:
    """Supprime les segments trop courts, facon Pycrostates.

    Les segments de duree < min_samples sont retires par ordre de duree
    croissante et leurs echantillons reaffectes a la classe voisine
    presentant la plus forte correlation.
    """
    if min_samples <= 1:
        return labels
    labels = labels.copy()
    n = len(labels)
    for _ in range(50):  # convergence en pratique en quelques passes
        bounds = np.flatnonzero(np.diff(labels)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [n]])
        lengths = ends - starts
        short = np.flatnonzero(lengths < min_samples)
        if len(short) == 0:
            break
        # traite le plus court d'abord
        i = short[np.argmin(lengths[short])]
        s, e = starts[i], ends[i]
        cand = []
        if i > 0:
            cand.append(labels[starts[i - 1]])
        if i < len(starts) - 1:
            cand.append(labels[starts[i + 1]])
        if not cand:
            break
        scores = [corr_abs[c, s:e].mean() for c in cand]
        labels[s:e] = cand[int(np.argmax(scores))]
    return labels


@dataclass
class Segmentation:
    labels: np.ndarray       # (n_times,) classe attribuee a chaque echantillon
    corr: np.ndarray         # (k, n_times) correlations spatiales signees
    gfp: np.ndarray          # (n_times,)
    sfreq: float
    n_maps: int


def backfit(data: np.ndarray, maps: np.ndarray, sfreq: float,
            min_segment_ms: float = 30.0,
            polarity_invariant: bool = True,
            reject_by_corr: float | None = None) -> Segmentation:
    """Attribue une classe a chaque echantillon (tous les echantillons).

    polarity_invariant : la classe est choisie sur |correlation| — c'est la
    convention standard des microstates, la polarite n'etant pas informative.
    """
    corr = spatial_correlation(data, maps)
    score = np.abs(corr) if polarity_invariant else corr
    labels = np.argmax(score, axis=0)
    if reject_by_corr is not None:
        # echantillons mal expliques : marques -1 (exclus des parametres)
        weak = score.max(axis=0) < reject_by_corr
    else:
        weak = None
    min_samples = int(round(min_segment_ms * sfreq / 1000.0))
    labels = _reject_short_segments(labels, score, min_samples)
    if weak is not None:
        labels = np.where(weak, -1, labels)
    gfp = data.std(axis=0)
    return Segmentation(labels=labels, corr=corr, gfp=gfp, sfreq=sfreq,
                        n_maps=len(maps))


# ---------------------------------------------------------------- parametres
def _segments(labels: np.ndarray):
    """Liste (classe, debut, fin) des segments consecutifs."""
    if len(labels) == 0:
        return []
    bounds = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def lempel_ziv_complexity(seq: np.ndarray, n_symbols: int | None = None,
                          normalize: bool = True) -> float:
    """Complexite de Lempel-Ziv (LZ76, historique exhaustif).

    Normalisation standard : c(n) * log_K(n) / n, ou K est le nombre de
    symboles. Une sequence aleatoire uniforme tend vers 1.
    """
    s = np.asarray(seq)
    n = len(s)
    if n == 0:
        return 0.0
    k = int(n_symbols) if n_symbols else int(s.max() + 1)

    # algorithme de Kaspar & Schuster (1987) pour la complexite LZ76
    i, ell, kk, k_max, c = 0, 1, 1, 1, 1
    while True:
        if s[i + kk - 1] == s[ell + kk - 1]:
            kk += 1
            if ell + kk > n:
                c += 1
                break
        else:
            if kk > k_max:
                k_max = kk
            i += 1
            if i == ell:
                c += 1
                ell += k_max
                if ell + 1 > n:
                    break
                i, kk, k_max = 0, 1, 1
            else:
                kk = 1
    if not normalize:
        return float(c)
    if k < 2 or n < 2:
        return 0.0  # normalisation indefinie (un seul symbole)
    # normalisation standard : c(n) / (n / log_K(n))
    return float(c * np.log(n) / np.log(k) / n)


def shannon_entropy(labels: np.ndarray, n_maps: int) -> float:
    """Entropie de Shannon (bits) de la distribution des classes."""
    lab = labels[labels >= 0]
    if len(lab) == 0:
        return 0.0
    p = np.bincount(lab, minlength=n_maps).astype(float)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_rate(labels: np.ndarray, n_maps: int) -> float:
    """Taux d'entropie markovien d'ordre 1 (bits/symbole), self-transitions incluses."""
    lab = labels[labels >= 0]
    if len(lab) < 2:
        return 0.0
    t = np.zeros((n_maps, n_maps))
    np.add.at(t, (lab[:-1], lab[1:]), 1.0)
    pi = t.sum(axis=1)
    pi = pi / pi.sum()
    rows = t.sum(axis=1, keepdims=True)
    p = np.divide(t, np.maximum(rows, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return float((pi * h).sum())


def transition_matrix(labels: np.ndarray, n_maps: int,
                      exclude_self: bool = True) -> np.ndarray:
    """Matrice de transition sur la sequence de SEGMENTS.

    On travaille sur les segments (et non les echantillons) : les
    self-transitions n'existent alors pas par construction, ce qui est la
    convention de la litterature microstates.
    """
    segs = [c for c, _, _ in _segments(labels) if c >= 0]
    t = np.zeros((n_maps, n_maps))
    for a, b in zip(segs[:-1], segs[1:]):
        if exclude_self and a == b:
            continue
        t[a, b] += 1
    rows = t.sum(axis=1, keepdims=True)
    return np.divide(t, np.maximum(rows, 1e-12))


def global_explained_variance(seg: Segmentation) -> tuple[float, np.ndarray]:
    """GEV globale et par classe.

    GEV_k = somme_{t in k} (GFP_t * corr_kt)^2 / somme_t GFP_t^2
    """
    gfp2 = (seg.gfp ** 2).sum()
    gev_k = np.zeros(seg.n_maps)
    for k in range(seg.n_maps):
        m = seg.labels == k
        if m.any():
            gev_k[k] = ((seg.gfp[m] * seg.corr[k, m]) ** 2).sum()
    gev_k /= max(gfp2, 1e-30)
    return float(gev_k.sum()), gev_k


def microstate_parameters(seg: Segmentation, boundaries: np.ndarray | None = None
                          ) -> dict:
    """Parametres standards des microstates.

    boundaries : indices de debut des blocs continus (bords d'epoch). Les
    segments a cheval sur un bord ne sont pas comptes dans les durees.
    """
    k = seg.n_maps
    labels = seg.labels
    n = len(labels)
    dur_s = 1.0 / seg.sfreq
    total_time = n * dur_s

    bset = set(int(b) for b in (boundaries if boundaries is not None else []))
    counts = np.zeros(k)
    durations = [[] for _ in range(k)]
    for cls, s, e in _segments(labels):
        if cls < 0:
            continue
        # segment tronque par un bord de bloc : compte dans la couverture,
        # pas dans les durees
        truncated = (s in bset and s != 0) or (e in bset) or s == 0 or e == n
        if not truncated:
            durations[cls].append((e - s) * dur_s)
        counts[cls] += 1

    coverage = np.array([(labels == c).mean() for c in range(k)])
    mean_dur = np.array([np.mean(d) if d else np.nan for d in durations])
    occurrence = counts / total_time
    gev_total, gev_k = global_explained_variance(seg)

    return dict(
        coverage=coverage,
        mean_duration_ms=mean_dur * 1000.0,
        occurrence_per_s=occurrence,
        gev=gev_k,
        gev_total=gev_total,
        transition=transition_matrix(labels, k, exclude_self=True),
        entropy_bits=shannon_entropy(labels, k),
        entropy_rate_bits=entropy_rate(labels, k),
        lzc=lempel_ziv_complexity(labels[labels >= 0], n_symbols=k),
        n_segments=int(counts.sum()),
    )
