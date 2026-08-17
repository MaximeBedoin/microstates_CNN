"""Preprocessing EEG et extraction des pics de GFP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeakSet:
    """Pics de GFP d'un sujet."""
    data: np.ndarray        # (n_peaks, n_ch) topographies aux pics
    gfp: np.ndarray         # (n_peaks,) valeur de GFP au pic
    times: np.ndarray       # (n_peaks,) indices d'echantillons
    subject: str
    group: str = ""
    ch_names: tuple = ()


def preprocess_raw(raw, l_freq: float = 1.0, h_freq: float = 40.0,
                   average_reference: bool = True, resample: float | None = None):
    """Filtrage passe-bande + reference moyenne.

    Le passe-haut a 1 Hz n'est pas dans la specification initiale mais est
    indispensable : sans lui les derives lentes dominent la GFP et les "pics"
    ne correspondent plus a des topographies stables (cf. CHOIX_METHODO).
    """
    raw = raw.copy().load_data(verbose="error")
    raw.pick("eeg")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, fir_design="firwin", verbose="error")
    if resample is not None and abs(raw.info["sfreq"] - resample) > 1e-6:
        raw.resample(resample, verbose="error")
    if average_reference:
        raw.set_eeg_reference("average", projection=False, verbose="error")
    return raw


def global_field_power(data: np.ndarray) -> np.ndarray:
    """GFP = ecart-type spatial. `data` : (n_ch, n_times)."""
    return data.std(axis=0)


def find_gfp_peaks(data: np.ndarray, min_distance: int = 3,
                   reject_percentile: float | None = 99.0):
    """Maxima locaux de la GFP.

    Parameters
    ----------
    data : array (n_ch, n_times), deja re-reference en moyenne.
    min_distance : int
        Distance minimale entre pics, en echantillons (3 par specification).
    reject_percentile : float or None
        Les pics dont la GFP depasse ce percentile sont ecartes (artefacts
        residuels). None pour desactiver.

    Returns
    -------
    idx : array (n_peaks,)
    gfp : array (n_times,)
    """
    from scipy.signal import find_peaks

    gfp = global_field_power(data)
    idx, _ = find_peaks(gfp, distance=min_distance)
    if reject_percentile is not None and len(idx):
        thr = np.percentile(gfp[idx], reject_percentile)
        idx = idx[gfp[idx] <= thr]
    return idx, gfp


def extract_peaks(raw, subject: str, group: str = "", min_distance: int = 3,
                  reject_percentile: float | None = 99.0,
                  normalize: bool = True) -> PeakSet:
    """Extrait les topographies aux pics de GFP d'un Raw deja pretraite.

    `normalize` divise chaque topographie par sa GFP : le VAE apprend alors
    des motifs spatiaux et non des amplitudes.
    """
    data = raw.get_data()
    idx, gfp = find_gfp_peaks(data, min_distance, reject_percentile)
    topo = data[:, idx].T.astype(np.float32)          # (n_peaks, n_ch)
    peak_gfp = gfp[idx].astype(np.float32)
    if normalize:
        topo = topo / np.maximum(peak_gfp[:, None], 1e-20)
    return PeakSet(data=topo, gfp=peak_gfp, times=idx, subject=subject,
                   group=group, ch_names=tuple(raw.ch_names))


def epochs_to_continuous(epochs) -> np.ndarray:
    """Concatene des Epochs en (n_ch, n_times) pour l'analyse temporelle.

    Attention : introduit des discontinuites aux bords d'epoch ; les segments
    a cheval sont ignores lors du calcul des durees (cf. `microstates`).
    """
    d = epochs.get_data(copy=False)
    return np.concatenate(list(d), axis=1)
