import numpy as np

from msvae.microstates import (backfit, entropy_rate, lempel_ziv_complexity,
                               microstate_parameters, shannon_entropy,
                               transition_matrix, _segments)


def _maps(k=4, n_ch=32, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((k, n_ch))
    m -= m.mean(axis=1, keepdims=True)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_lzc_bounds():
    rng = np.random.default_rng(0)
    rand = lempel_ziv_complexity(rng.integers(0, 4, 20000), n_symbols=4)
    per = lempel_ziv_complexity(np.tile([0, 1, 2, 3], 5000), n_symbols=4)
    assert 0.9 < rand < 1.1
    assert per < 0.05


def test_shannon_entropy_uniform():
    lab = np.tile([0, 1, 2, 3], 1000)
    assert abs(shannon_entropy(lab, 4) - 2.0) < 1e-9
    assert shannon_entropy(np.zeros(100, int), 4) == 0.0


def test_entropy_rate_deterministic_sequence():
    lab = np.tile([0, 1, 2, 3], 1000)
    assert entropy_rate(lab, 4) < 1e-9


def test_transition_matrix_excludes_self():
    lab = np.array([0, 0, 0, 1, 1, 2, 2, 0, 0])
    t = transition_matrix(lab, 3)
    assert np.allclose(np.diag(t), 0)
    assert np.allclose(t.sum(axis=1)[t.sum(axis=1) > 0], 1.0)


def test_segments():
    lab = np.array([1, 1, 0, 0, 0, 2])
    assert _segments(lab) == [(1, 0, 2), (0, 2, 5), (2, 5, 6)]


def test_backfit_recovers_known_sequence():
    """Sur un signal construit a partir des cartes, le backfit doit retrouver
    la sequence — y compris quand la polarite est inversee."""
    maps = _maps()
    rng = np.random.default_rng(1)
    sfreq = 250.0
    labels = np.repeat(rng.integers(0, 4, 200), 25)  # segments de 100 ms
    sign = np.repeat(rng.choice([-1.0, 1.0], 200), 25)
    data = (maps[labels].T * sign) + 0.05 * rng.standard_normal((32, len(labels)))
    seg = backfit(data, maps, sfreq, min_segment_ms=30.0)
    assert (seg.labels == labels).mean() > 0.98


def test_parameters_are_coherent():
    maps = _maps()
    rng = np.random.default_rng(2)
    labels = np.repeat(rng.integers(0, 4, 400), 25)
    data = maps[labels].T + 0.1 * rng.standard_normal((32, len(labels)))
    seg = backfit(data, maps, 250.0, min_segment_ms=30.0)
    p = microstate_parameters(seg)
    assert abs(p["coverage"].sum() - 1.0) < 1e-9
    assert 0.0 <= p["gev_total"] <= 1.0
    assert np.nanmin(p["mean_duration_ms"]) >= 30.0
    assert p["transition"].shape == (4, 4)
    assert 0.0 <= p["lzc"] <= 2.0


def test_short_segments_are_removed():
    maps = _maps()
    rng = np.random.default_rng(3)
    data = rng.standard_normal((32, 2000))
    seg = backfit(data, maps, 250.0, min_segment_ms=40.0)
    lengths = [e - s for _, s, e in _segments(seg.labels)]
    # tolerance : le premier et le dernier segment peuvent etre tronques
    assert min(lengths[1:-1]) >= 10


def test_boundaries_exclude_truncated_segments_from_durations():
    """Les segments qui touchent un bord d'epoch ne doivent pas compter dans
    les durees moyennes (mais restent dans la couverture)."""
    maps = _maps(k=2, n_ch=16)
    # 4 blocs de 100 echantillons ; classe alternee toutes les 50 (200 ms)
    labels = np.tile(np.repeat([0, 1], 50), 4)
    data = maps[labels].T
    seg = backfit(data, maps, 250.0, min_segment_ms=0.0)
    bounds = np.arange(4) * 100
    p_free = microstate_parameters(seg)
    p_bnd = microstate_parameters(seg, bounds)
    assert abs(p_free["coverage"].sum() - p_bnd["coverage"].sum()) < 1e-12
    # sans bords : des segments de 200 ms sont mesures
    assert np.nanmax(p_free["mean_duration_ms"]) > 100
    # avec bords tous les 100 echantillons, tout segment touche un bord
    assert np.all(np.isnan(p_bnd["mean_duration_ms"]))
