import numpy as np
import pytest

from msvae import classify


def _params(k=4, rng=None, shift=0.0):
    rng = rng or np.random.default_rng(0)
    return dict(
        coverage=rng.random(k), mean_duration_ms=60 + 10 * rng.random(k) + shift,
        occurrence_per_s=rng.random(k), gev=rng.random(k),
        gev_total=float(rng.random()),
        transition=rng.random((k, k)) * (1 - np.eye(k)),
        entropy_bits=float(rng.random()), entropy_rate_bits=float(rng.random()),
        lzc=float(rng.random()), n_segments=100)


def test_features_shape_and_names():
    k = 4
    X, names = classify.features_from_params([_params(k), _params(k)])
    # 4 blocs par classe + hors-diagonale de la transition + 3 scalaires
    assert X.shape == (2, 4 * k + k * (k - 1) + 3)
    assert len(names) == X.shape[1]
    assert "gev_total" not in names           # exclu par defaut
    assert not any("trans[0->0]" == n for n in names)   # diagonale exclue


def test_features_gev_total_optionnel():
    X, names = classify.features_from_params([_params()], include_gev_total=True)
    assert names[-1] == "gev_total"


def test_features_nan_conserve():
    """mean_duration_ms peut valoir NaN : c'est au pipeline sklearn d'imputer,
    pas a l'extraction de features de masquer l'information."""
    p = _params()
    p["mean_duration_ms"][2] = np.nan
    X, _ = classify.features_from_params([p])
    assert np.isnan(X).sum() == 1


def test_evaluate_arm_separe_deux_groupes():
    rng = np.random.default_rng(0)
    n = 40
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, 5))
    X[:, 0] += 3.0 * y                     # signal franc sur une colonne
    subjects = np.array([f"s{i}" for i in range(n)])
    res = classify.evaluate_arm(X, y, subjects, n_splits=4, n_repeats=2)
    assert res["auc"] > 0.9
    assert res["prob"].shape == (n,)


def test_evaluate_arm_sans_signal_reste_au_hasard():
    rng = np.random.default_rng(1)
    n = 60
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, 5))            # aucune information
    subjects = np.array([f"s{i}" for i in range(n)])
    res = classify.evaluate_arm(X, y, subjects, n_splits=5, n_repeats=2)
    assert 0.25 < res["auc"] < 0.75


def test_groupes_par_sujet_quand_sujets_repetes():
    """Deux lignes par sujet (protocole apparie) : aucun pli ne doit contenir
    le meme sujet des deux cotes."""
    n_sub = 20
    subjects = np.repeat([f"s{i}" for i in range(n_sub)], 2)
    y = np.tile([0, 1], n_sub)
    splits = classify._cv_splits(y, subjects, n_splits=4, n_repeats=2, seed=0)
    assert len(splits) == 8
    for tr, te in splits:
        assert not (set(subjects[tr]) & set(subjects[te]))


def test_compare_arms_detecte_la_difference():
    y = np.repeat([0, 1], 30)
    rng = np.random.default_rng(2)
    bon = np.where(y == 1, rng.uniform(0.6, 1.0, 60), rng.uniform(0.0, 0.4, 60))
    nul = rng.uniform(0, 1, 60)
    comp = classify.compare_arms({"bon": dict(prob=bon, auc=1.0),
                                  "nul": dict(prob=nul, auc=0.5)},
                                 y, n_boot=200, seed=0)
    d = comp["pairwise"]["bon - nul"]
    assert d["delta"] > 0
    assert d["ci"][0] < d["ci"][1]
    assert "bon" in comp["auc_ci"]


def test_spectral_features_global_et_channel():
    class Rec:
        def __init__(self):
            rng = np.random.default_rng(3)
            self.sfreq = 250.0
            t = np.arange(0, 8, 1 / self.sfreq)
            # 10 Hz net sur 6 capteurs : le pic alpha doit etre retrouve
            self.data = (np.sin(2 * np.pi * 10 * t)[None] * np.ones((6, 1))
                         + 0.1 * rng.normal(size=(6, len(t))))

    recs = [Rec(), Rec()]
    Xg, ng = classify.spectral_features(recs, scope="global")
    Xc, nc = classify.spectral_features(recs, scope="channel")
    assert Xg.shape == (2, len(classify.DEFAULT_BANDS) + 1)
    assert Xc.shape == (2, 6 * len(classify.DEFAULT_BANDS) + 1)
    assert ng[-1] == nc[-1] == "peak_alpha_hz"
    assert 9.0 <= Xg[0, -1] <= 11.0
    # puissance relative : somme des bandes proche de 1
    assert Xg[0, :len(classify.DEFAULT_BANDS)].sum() == pytest.approx(1.0, abs=0.05)
