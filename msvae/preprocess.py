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


def epochs_to_continuous(epochs) -> tuple[np.ndarray, np.ndarray]:
    """Concatene des Epochs en (n_ch, n_times) + indices de debut d'epoch.

    Attention : introduit des discontinuites aux bords d'epoch ; les segments
    a cheval sont ignores lors du calcul des durees (cf. `microstates`).
    """
    d = epochs.get_data(copy=False)
    n_times = d.shape[2]
    boundaries = np.arange(len(d)) * n_times
    return np.concatenate(list(d), axis=1), boundaries


def clean_epochs(raw, duration: float = 2.0,
                 reject_ptp: float | str | None = "auto"):
    """Decoupe en epochs de longueur fixe et rejette les epochs artefactees.

    Le critere est le pic-a-pic maximal sur les canaux EEG : nettoyage
    grossier, suffisant pour du repos, mais qui ne remplace pas une inspection
    ICA/visuelle sur des donnees cliniques.

    reject_ptp : float | 'auto' | None
        Seuil absolu en volts, ou 'auto' pour un seuil robuste propre a
        l'enregistrement (mediane + 3 x MAD normalise). Un seuil absolu est
        difficile a fixer a l'avance : sur EEGBCI, 150 uV rejetait 58 % des
        epochs et faisait disparaitre la moitie des enregistrements.

    Returns
    -------
    data : (n_ch, n_times) concatene
    boundaries : indices de debut d'epoch (discontinuites)
    n_dropped : nombre d'epochs rejetees
    """
    import mne

    epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True,
                                          verbose="error")
    n_before = len(epochs)
    if reject_ptp is not None:
        ptp = np.ptp(epochs.get_data(copy=False), axis=2).max(axis=1)
        if isinstance(reject_ptp, str):
            if reject_ptp != "auto":
                raise ValueError("reject_ptp doit valoir un float, 'auto' ou None")
            mad = np.median(np.abs(ptp - np.median(ptp))) * 1.4826
            thr = np.median(ptp) + 3.0 * max(mad, 1e-12)
        else:
            thr = reject_ptp
        epochs = epochs[ptp <= thr]
    if len(epochs) == 0:
        raise ValueError("toutes les epochs ont ete rejetees")
    data, boundaries = epochs_to_continuous(epochs)
    return data, boundaries, n_before - len(epochs)
