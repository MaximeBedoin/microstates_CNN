"""Generateur de donnees EEG synthetiques avec dipoles connus.

Modele direct : sphere a 3 couches de MNE (`make_sphere_model`, approximation
de Berg), qui ne necessite aucun telechargement. Les "microstates" sont
produits par des dipoles fixes :

  A : dipole central tangentiel oriente selon la diagonale gauche-post -> droite-ant
  B : dipole central tangentiel selon la diagonale opposee
  C : dipole central tangentiel oriente posterieur -> anterieur
  D : dipole radial superficiel sous FCz

Un dipole central produit sur la sphere un champ quasi lineaire selon son
orientation (topographie "en gradient"), un dipole radial superficiel produit
un maximum local : c'est exactement la geometrie decrite pour les microstates
canoniques A/B/C/D.

Le decours temporel est semi-markovien (durees Gamma, pas de self-transition),
avec inversion aleatoire de polarite a chaque segment. Le bruit de fond est
genere par des dipoles aleatoires a spectre 1/f, donc spatialement correle
(un bruit blanc capteur serait trivialement rejete par l'interpolation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_MONTAGE = "biosemi64"


# --------------------------------------------------------------------- modele
def make_info(montage: str = DEFAULT_MONTAGE, sfreq: float = 250.0):
    import mne

    names = mne.channels.make_standard_montage(montage).ch_names
    info = mne.create_info(names, sfreq, "eeg")
    info.set_montage(montage)
    return info


def make_leadfield(info):
    """Leadfield EEG (n_ch, 3 * n_src) et positions des sources (n_src, 3)."""
    import mne

    sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09,
                                   relative_radii=(0.90, 0.92, 0.97, 1.0),
                                   sigmas=(0.33, 1.0, 0.004, 0.33))
    src = mne.setup_volume_source_space(sphere=sphere, pos=12.0, sphere_units="m")
    fwd = mne.make_forward_solution(info, trans=None, src=src, bem=sphere,
                                    eeg=True, meg=False)
    lead = fwd["sol"]["data"]
    rr = fwd["src"][0]["rr"][fwd["src"][0]["inuse"].astype(bool)]
    return lead, rr


def _project(lead, rr, pos, ori):
    """Topographie d'un dipole (pos, ori) : plus proche source de la grille."""
    idx = int(np.argmin(np.linalg.norm(rr - np.asarray(pos), axis=1)))
    ori = np.asarray(ori, dtype=float)
    ori = ori / np.linalg.norm(ori)
    v = lead[:, 3 * idx:3 * idx + 3] @ ori
    return v - v.mean()


# Configurations de dipoles : (position en metres, orientation).
# Choisies par recherche numerique pour minimiser la correlation absolue
# maximale entre topographies (max |r| = 0.34 ici), tout en restant fidele a
# la description des microstates canoniques. Avec quatre dipoles centraux
# purement tangentiels on ne depasse pas trois champs orthogonaux (les
# gradients vivent dans un espace de dimension 3), d'ou le recours a des
# dipoles radiaux excentres pour C et D.
CANONICAL_DIPOLES = {
    # tangentiel excentre gauche-posterieur : gradient diagonal
    "A": ((-0.055, -0.040, 0.050), (0.707, 0.707, 0.0)),
    # miroir droit
    "B": ((0.055, -0.040, 0.050), (-0.707, 0.707, 0.0)),
    # radial occipital (maximum posterieur)
    "C": ((0.0, -0.060, 0.055), (0.0, -0.5, 0.86)),
    # radial fronto-central
    "D": ((0.0, 0.035, 0.080), (0.0, 0.25, 0.97)),
    # deux classes supplementaires si K_true > 4 (E, F de la litterature)
    "E": ((0.0, 0.0, 0.090), (0.0, 0.0, 1.0)),
    "F": ((0.070, 0.0, 0.045), (1.0, 0.0, 0.2)),
}


def canonical_maps(info=None, lead=None, rr=None, n_states: int = 4,
                   jitter: float = 0.0, rng=None) -> np.ndarray:
    """Topographies de reference (n_states, n_ch), normalisees (norme 1).

    `jitter` (en metres) deplace aleatoirement les dipoles et perturbe leur
    orientation : c'est la variabilite inter-sujets.
    """
    if lead is None or rr is None:
        lead, rr = make_leadfield(info)
    rng = np.random.default_rng(rng)
    keys = list(CANONICAL_DIPOLES)[:n_states]
    maps = []
    for k in keys:
        pos, ori = CANONICAL_DIPOLES[k]
        pos = np.asarray(pos, float)
        ori = np.asarray(ori, float)
        if jitter > 0:
            pos = pos + rng.normal(0, jitter, 3)
            ori = ori + rng.normal(0, 0.15, 3)
        v = _project(lead, rr, pos, ori)
        maps.append(v / np.linalg.norm(v))
    return np.asarray(maps)


# ----------------------------------------------------------------- sequences
def _pink_noise(n_samples: int, n_series: int, rng, exponent: float = 1.0):
    """Bruit a spectre en 1/f^exponent, (n_series, n_samples)."""
    freqs = np.fft.rfftfreq(n_samples)
    scale = np.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-exponent / 2.0)
    spec = (rng.normal(size=(n_series, len(freqs)))
            + 1j * rng.normal(size=(n_series, len(freqs)))) * scale
    out = np.fft.irfft(spec, n=n_samples, axis=1)
    return out / out.std(axis=1, keepdims=True)


def _semi_markov_labels(n_samples: int, sfreq: float, n_states: int,
                        trans: np.ndarray, mean_dur: float, rng):
    """Sequence de labels, durees Gamma(shape=4), pas de self-transition."""
    labels = np.empty(n_samples, dtype=np.int64)
    pos, state = 0, rng.integers(n_states)
    shape = 4.0
    while pos < n_samples:
        dur_s = rng.gamma(shape, mean_dur / shape)
        n = max(int(round(dur_s * sfreq)), 2)
        labels[pos:pos + n] = state
        pos += n
        state = rng.choice(n_states, p=trans[state])
    return labels[:n_samples]


def default_transition_matrix(n_states: int, asym: float = 0.35, rng=None):
    """Matrice de transition sans self-transition, legerement asymetrique."""
    rng = np.random.default_rng(rng)
    t = np.ones((n_states, n_states)) + asym * rng.random((n_states, n_states))
    np.fill_diagonal(t, 0.0)
    return t / t.sum(axis=1, keepdims=True)


@dataclass
class SyntheticSubject:
    raw: object
    labels: np.ndarray           # (n_samples,) etat vrai
    maps: np.ndarray             # (n_states, n_ch) topographies vraies du sujet
    group: str
    subject: str
    meta: dict = field(default_factory=dict)


def simulate_subject(info, lead, rr, subject: str, group: str = "G1",
                     duration: float = 60.0, n_states: int = 4,
                     mean_dur: float = 0.080, snr: float = 1.0,
                     jitter: float = 0.006, trans: np.ndarray | None = None,
                     n_bg_dipoles: int = 60, rng=None) -> SyntheticSubject:
    """Simule un sujet.

    Parameters
    ----------
    snr : float
        Rapport ecart-type(signal microstates) / ecart-type(bruit de fond).
    jitter : float
        Deplacement aleatoire des dipoles (m) : variabilite inter-sujets.
    mean_dur : float
        Duree moyenne des segments (s).
    """
    import mne

    rng = np.random.default_rng(rng)
    sfreq = info["sfreq"]
    n_samples = int(duration * sfreq)

    maps = canonical_maps(lead=lead, rr=rr, n_states=n_states, jitter=jitter, rng=rng)
    if trans is None:
        trans = default_transition_matrix(n_states, rng=rng)
    labels = _semi_markov_labels(n_samples, sfreq, n_states, trans, mean_dur, rng)

    # enveloppe d'amplitude lente (1/f) + polarite aleatoire par segment
    env = np.abs(_pink_noise(n_samples, 1, rng, exponent=1.5)[0]) + 0.35
    switch = np.flatnonzero(np.diff(labels, prepend=labels[0] - 1) != 0)
    sign = np.ones(n_samples)
    seg_sign = rng.choice([-1.0, 1.0], size=len(switch))
    for i, s in enumerate(switch):
        e = switch[i + 1] if i + 1 < len(switch) else n_samples
        sign[s:e] = seg_sign[i]

    signal = (maps[labels].T * (env * sign))  # (n_ch, n_samples)

    # bruit de fond : dipoles aleatoires a spectre 1/f
    idx = rng.choice(len(rr), size=n_bg_dipoles, replace=False)
    oris = rng.normal(size=(n_bg_dipoles, 3))
    oris /= np.linalg.norm(oris, axis=1, keepdims=True)
    topo_bg = np.stack([lead[:, 3 * i:3 * i + 3] @ o for i, o in zip(idx, oris)], axis=1)
    topo_bg -= topo_bg.mean(axis=0, keepdims=True)
    topo_bg /= np.linalg.norm(topo_bg, axis=0, keepdims=True)
    bg = topo_bg @ _pink_noise(n_samples, n_bg_dipoles, rng, exponent=1.2)

    signal /= signal.std()
    bg /= bg.std()
    data = signal + bg / max(snr, 1e-6)
    data += 0.02 * rng.normal(size=data.shape)  # bruit capteur
    data *= 20e-6  # echelle physiologique (V)

    raw = mne.io.RawArray(data, info.copy(), verbose="error")
    raw.info["subject_info"] = {"his_id": subject}
    return SyntheticSubject(raw=raw, labels=labels, maps=maps, group=group,
                            subject=subject,
                            meta=dict(snr=snr, mean_dur=mean_dur, trans=trans))


def simulate_dataset(n_subjects: int = 20, duration: float = 60.0,
                     n_states: int = 4, sfreq: float = 250.0,
                     montage: str = DEFAULT_MONTAGE, snr: float = 1.0,
                     two_groups: bool = True, seed: int = 0):
    """Cohorte synthetique.

    Si `two_groups`, la moitie des sujets a une duree moyenne de segment plus
    courte pour la classe 0 (effet "clinique" connu, sert a valider le volet
    pouvoir discriminant).
    """
    rng = np.random.default_rng(seed)
    info = make_info(montage, sfreq)
    lead, rr = make_leadfield(info)
    group_maps = canonical_maps(lead=lead, rr=rr, n_states=n_states, jitter=0.0)

    subjects = []
    for i in range(n_subjects):
        group = "G2" if (two_groups and i % 2 == 1) else "G1"
        mean_dur = 0.065 if group == "G2" else 0.085
        trans = default_transition_matrix(n_states, rng=rng)
        if group == "G2":  # transition C->D renforcee dans le groupe 2
            trans[min(2, n_states - 1)] *= 1.0
            trans[min(2, n_states - 1), min(3, n_states - 1)] *= 2.5
            trans /= trans.sum(axis=1, keepdims=True)
        subjects.append(simulate_subject(
            info, lead, rr, subject=f"sub-{i:03d}", group=group,
            duration=duration, n_states=n_states, mean_dur=mean_dur,
            snr=snr, trans=trans, rng=rng))
    return subjects, group_maps
