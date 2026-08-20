"""Banc de classification : quelle representation preserve l'information utile ?

Motivation. Les metriques disponibles jusqu'ici ne permettent pas de departager
les methodes :

  - la GEV est saturee (sur la cohorte synthetique, les cartes VRAIES sont
    battues par trois methodes sur quatre : 0.674 contre 0.688) ;
  - le sanity check canonique est anti-correle a la verite (le ground truth y
    score 0.746, moins bien que toutes les methodes) ;
  - la stabilite split-half mesure la reproductibilite, pas la qualite : une
    methode degeneree qui renvoie toujours les memes cartes est parfaitement
    stable ;
  - la correlation aux cartes vraies est valide mais n'existe pas sur du reel.

Le pouvoir discriminant, lui, a de la marge (0.5 -> 1.0), existe sur simule
comme sur reel, se tient HORS de la boucle de selection (donc regle le piege
P1 de HYPOTHESES.md), et se compare entre des methodes qui n'ont rien d'autre
en commun.

Point de comparaison obligatoire : le bras spectral. L'EEG des patients
Alzheimer est globalement ralenti, et la duree des microstates est couplee au
rythme dominant ; un classifieur sur parametres de microstates peut donc
reussir en lisant seulement "ce cerveau est ralenti". Sans ce point zero, une
AUC de 0.82 n'est pas interpretable.
"""

from __future__ import annotations

import numpy as np

DEFAULT_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                 "beta": (13.0, 30.0), "gamma": (30.0, 40.0)}


# ------------------------------------------------------- vecteurs de features
def features_from_params(params: list[dict], include_gev_total: bool = False,
                         include_gev: bool = True
                         ) -> tuple[np.ndarray, list[str]]:
    """Parametres de microstates -> matrice (n_sujets, n_features).

    ATTENTION a l'interaction entre les deux drapeaux. `gev_total` est exclu par
    defaut au motif que c'est une qualite d'ajustement et non un descripteur du
    sujet — mais `microstates.global_explained_variance` retourne
    `gev_total = gev_k.sum()`, donc garder les K valeurs par classe suffit a
    reconstituer le total par simple somme, ce qu'un classifieur lineaire fait
    sans effort. **Exclure `gev_total` en gardant `gev` ne retire donc rien.**

    Le drapeau `include_gev` rend le choix explicite :

      * True (defaut) : jeu de features conforme a la litterature, ou la GEV par
        classe est rapportee au meme titre que couverture, duree et occurrence.
        La qualite d'ajustement est alors dans les features, assumee.
      * False : aucune information de qualite d'ajustement. C'est la version a
        utiliser pour COMPARER des methodes, ou l'on ne veut pas qu'une methode
        tire son pouvoir discriminant du fait qu'elle ajuste mieux un groupe
        que l'autre.
    """
    rows, names = [], None
    keys = ("coverage", "mean_duration_ms", "occurrence_per_s")
    if include_gev:
        keys = keys + ("gev",)
    for p in params:
        vec, nm = [], []
        k = len(p["coverage"])
        for key in keys:
            vec.extend(np.asarray(p[key], dtype=float).ravel())
            nm.extend(f"{key}[{c}]" for c in range(k))
        # matrice de transition : la diagonale est nulle par construction
        # (exclude_self=True), on ne garde donc que les termes hors diagonale
        trans = np.asarray(p["transition"], dtype=float)
        for i in range(k):
            for j in range(k):
                if i != j:
                    vec.append(trans[i, j])
                    nm.append(f"trans[{i}->{j}]")
        for key in ("entropy_bits", "entropy_rate_bits", "lzc"):
            vec.append(float(p[key]))
            nm.append(key)
        if include_gev_total:
            vec.append(float(p["gev_total"]))
            nm.append("gev_total")
        rows.append(vec)
        names = nm if names is None else names
    return np.asarray(rows, dtype=float), names


def spectral_features(records, bands: dict | None = None, scope: str = "global"
                      ) -> tuple[np.ndarray, list[str]]:
    """Bras de reference : puissance relative par bande + frequence du pic alpha.

    scope : 'global' agrege sur les capteurs (n_bandes + 1 features, dimension
    comparable a celle des parametres de microstates) ; 'channel' garde le
    detail par capteur (n_ch * n_bandes + 1), plus riche mais de dimension tres
    superieure, ce qui n'est pas neutre a 65 sujets. Les deux sont rapportes :
    un point zero trop faible fausserait la comparaison autant qu'un point zero
    trop fort.
    """
    from scipy.signal import welch

    bands = bands or DEFAULT_BANDS
    rows, names = [], None
    for rec in records:
        data = np.asarray(rec.data, dtype=float)
        nperseg = int(min(4 * rec.sfreq, data.shape[1]))
        freqs, psd = welch(data, fs=rec.sfreq, nperseg=nperseg)
        total = psd[:, (freqs >= 1.0) & (freqs <= 40.0)].sum(axis=1)
        total = np.where(total > 0, total, 1.0)

        vec, nm = [], []
        for bname, (lo, hi) in bands.items():
            sel = (freqs >= lo) & (freqs < hi)
            rel = psd[:, sel].sum(axis=1) / total
            if scope == "global":
                vec.append(float(rel.mean()))
                nm.append(f"relpow_{bname}")
            else:
                vec.extend(rel.tolist())
                nm.extend(f"relpow_{bname}[{c}]" for c in range(data.shape[0]))

        # frequence du pic alpha, moyennee sur les capteurs
        asel = (freqs >= 7.0) & (freqs <= 13.0)
        peak = freqs[asel][np.argmax(psd[:, asel], axis=1)]
        vec.append(float(np.mean(peak)))
        nm.append("peak_alpha_hz")

        rows.append(vec)
        names = nm if names is None else names
    return np.asarray(rows, dtype=float), names


# ------------------------------------------------------------------ protocole
def _make_estimator(seed: int = 0):
    """Classifieur FIXE et identique pour tous les bras.

    Regression logistique L2, standardisee, avec choix de C par validation
    croisee interne : les bras n'ont pas la meme dimension (31 features pour
    les microstates a K=4, 96 pour le spectral par capteur), et imposer un C
    unique avantagerait mecaniquement le bras de plus faible dimension.
    L'imputation par la mediane est DANS le pipeline, donc ajustee sur le seul
    pli d'entrainement : `mean_duration_ms` vaut NaN si une classe n'a aucun
    segment non tronque.

    LIMITE CONNUE, sans effet sur les jeux de donnees utilises jusqu'ici. La
    validation croisee INTERNE de `LogisticRegressionCV` ignore les groupes :
    elle ne sait pas qu'un sujet peut apparaitre plusieurs fois. Le decoupage
    externe, lui, est groupe par sujet (`_cv_splits`). Sur un protocole APPARIE
    — un sujet fournissant deux conditions, comme yeux ouverts / yeux fermes —
    les deux lignes d'un meme sujet peuvent donc se retrouver de part et
    d'autre du pli interne, ce qui biaise optimistement le choix de C, sans
    toutefois contaminer l'estimation hors-pli externe. `synthetic` et
    `ds004504` n'ont qu'une ligne par sujet et ne sont pas concernes ; pour
    EEGBCI il faudrait remplacer LogisticRegressionCV par un GridSearchCV avec
    GroupKFold, sklearn ne propageant pas les groupes au CV interne.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        # l1_ratios=(0.0,) == penalite L2 pure (`penalty=` est deprecie
        # depuis scikit-learn 1.8 et disparait en 1.10)
        ("clf", LogisticRegressionCV(Cs=np.logspace(-3, 3, 13), cv=4,
                                     l1_ratios=(0.0,), scoring="roc_auc",
                                     use_legacy_attributes=False,
                                     max_iter=2000, random_state=seed)),
    ])


def _cv_splits(y, subjects, n_splits, n_repeats, seed):
    """Plis stratifies, groupes par sujet des qu'un sujet apparait plusieurs
    fois (cas apparie type yeux ouverts / yeux fermes) : sans cela le meme
    sujet se retrouve des deux cotes et toutes les AUC montent artificiellement.
    """
    from sklearn.model_selection import (RepeatedStratifiedKFold,
                                         StratifiedGroupKFold)

    subjects = np.asarray(subjects)
    if len(np.unique(subjects)) == len(subjects):
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                     random_state=seed)
        return [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y)]
    out = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=seed + r)
        out.extend(cv.split(np.zeros(len(y)), y, groups=subjects))
    return out


def evaluate_arm(X, y, subjects, n_splits: int = 5, n_repeats: int = 10,
                 seed: int = 0) -> dict:
    """Predictions hors-pli d'un bras, moyennees sur les repetitions.

    Retourne les probabilites OOF (n_sujets,) et l'AUC par repetition. Ce sont
    les probabilites par sujet qui servent ensuite aux comparaisons appariees :
    comparer deux AUC independantes gaspillerait la correlation entre methodes,
    qui est justement ce qui rend le test puissant a petit effectif.
    """
    from sklearn.metrics import roc_auc_score

    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    splits = _cv_splits(y, subjects, n_splits, n_repeats, seed)
    n_per_repeat = max(len(splits) // n_repeats, 1)

    oof = np.zeros((n_repeats, len(y)))
    counts = np.zeros((n_repeats, len(y)))
    for i, (tr, te) in enumerate(splits):
        r = min(i // n_per_repeat, n_repeats - 1)
        est = _make_estimator(seed)
        est.fit(X[tr], y[tr])
        oof[r, te] += est.predict_proba(X[te])[:, 1]
        counts[r, te] += 1

    oof = np.divide(oof, np.maximum(counts, 1))
    auc_per_repeat = np.array([roc_auc_score(y, oof[r]) for r in range(n_repeats)])
    return dict(prob=oof.mean(axis=0), auc_per_repeat=auc_per_repeat,
                auc=float(roc_auc_score(y, oof.mean(axis=0))),
                auc_sd_repeats=float(auc_per_repeat.std(ddof=1))
                if n_repeats > 1 else 0.0)


def compare_arms(results: dict, y, n_boot: int = 2000, seed: int = 0) -> dict:
    """Comparaisons appariees entre bras, par bootstrap sur les SUJETS.

    On re-echantillonne les sujets (pas les plis) : les plis d'une validation
    croisee repetee sont fortement correles, un test apparie sur les plis est
    donc anti-conservateur. Le bootstrap sujet respecte la seule unite
    d'observation independante dont on dispose.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    names = list(results)
    rng = np.random.default_rng(seed)
    n = len(y)

    boot = {nm: np.full(n_boot, np.nan) for nm in names}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:      # tirage degenere
            continue
        for nm in names:
            boot[nm][b] = roc_auc_score(y[idx], results[nm]["prob"][idx])

    out = {"auc_ci": {}, "pairwise": {}}
    for nm in names:
        v = boot[nm][np.isfinite(boot[nm])]
        out["auc_ci"][nm] = [float(np.percentile(v, 2.5)),
                             float(np.percentile(v, 97.5))]
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            d = boot[a] - boot[b_]
            d = d[np.isfinite(d)]
            out["pairwise"][f"{a} - {b_}"] = dict(
                delta=float(results[a]["auc"] - results[b_]["auc"]),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                # borne a 1 : les deux queues se recouvrent des qu'il y a des
                # ex aequo exacts a 0, frequents quand l'AUC est discrete
                # (petits effectifs), et 2*min(...) peut alors depasser 1
                p_two_sided=float(min(1.0, 2 * min((d <= 0).mean(),
                                                   (d >= 0).mean()))))
    return out


def _permute_labels(y, subjects, rng):
    """Permutation respectant la structure du protocole.

    Un sujet apparaissant plusieurs fois (protocole APPARIE, type yeux ouverts
    / yeux fermes), permuter ligne par ligne pourrait lui attribuer deux fois
    la meme condition, ce qui detruit l'appariement que `_cv_splits` respecte
    par ailleurs : la distribution nulle obtenue ne serait pas celle du
    protocole reel. On permute alors A L'INTERIEUR de chaque sujet. Quand
    chaque sujet n'a qu'une ligne, on retombe sur une permutation globale.
    """
    subjects = np.asarray(subjects)
    y = np.asarray(y, dtype=int)
    if len(np.unique(subjects)) == len(subjects):
        return rng.permutation(y)
    out = y.copy()
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        out[idx] = rng.permutation(y[idx])
    return out


def permutation_control(X, y, subjects, n_perm: int = 20, seed: int = 0,
                        **kw) -> dict:
    """Controle de calibration : sous permutation des etiquettes, l'AUC doit
    valoir 0.5. Toute valeur nettement superieure signale une fuite dans le
    protocole (decoupage, imputation ajustee hors pli, sujet des deux cotes).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    aucs = [evaluate_arm(X, _permute_labels(y, subjects, rng), subjects,
                         **kw)["auc"] for _ in range(n_perm)]
    return dict(mean=float(np.mean(aucs)),
                sd=float(np.std(aucs, ddof=1)) if n_perm > 1 else 0.0,
                aucs=[float(a) for a in aucs])
