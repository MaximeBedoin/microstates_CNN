"""Acces aux jeux de donnees et cache des donnees pretraitees.

Deux sources :
  * 'synthetic' : cohorte simulee (dipoles connus, ground truth disponible) ;
  * 'eegbci'    : EEG Motor Movement/Imagery Database (PhysioNet), repos yeux
    ouverts (R01) et yeux fermes (R02), 109 sujets, 64 electrodes, 160 Hz.

PhysioNet n'est pas joignable depuis cet environnement (proxy) mais son
miroir AWS Open Data l'est : on telecharge donc depuis
https://physionet-open.s3.amazonaws.com/ plutot que via `mne.datasets.eegbci`.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EEGBCI_BASE = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0"
DEFAULT_CACHE = Path(os.environ.get("MSVAE_CACHE", "cache"))


@dataclass
class SubjectRecord:
    subject: str
    group: str
    data: np.ndarray        # (n_ch, n_times) pretraite, float32
    sfreq: float
    ch_names: list
    boundaries: np.ndarray  # indices de debut des blocs continus
    extra: dict


# --------------------------------------------------------------------- cache
class SubjectCache:
    """Cache disque des donnees pretraitees (un .npz par sujet et condition)."""

    def __init__(self, root: Path | str = DEFAULT_CACHE, tag: str = "default"):
        self.root = Path(root) / tag
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def save(self, key: str, rec: SubjectRecord):
        np.savez_compressed(self.path(key), data=rec.data.astype(np.float32),
                            sfreq=rec.sfreq, ch_names=np.array(rec.ch_names),
                            boundaries=rec.boundaries, subject=rec.subject,
                            group=rec.group, extra=json.dumps(rec.extra))

    def load(self, key: str) -> SubjectRecord:
        z = np.load(self.path(key), allow_pickle=False)
        return SubjectRecord(subject=str(z["subject"]), group=str(z["group"]),
                             data=z["data"], sfreq=float(z["sfreq"]),
                             ch_names=[str(c) for c in z["ch_names"]],
                             boundaries=z["boundaries"],
                             extra=json.loads(str(z["extra"])))


# ------------------------------------------------------------------- eegbci
def eegbci_url(subject: int, run: int) -> str:
    return f"{EEGBCI_BASE}/S{subject:03d}/S{subject:03d}R{run:02d}.edf"


def download_eegbci(subject: int, run: int, dest: Path | str,
                    retries: int = 4) -> Path:
    """Telecharge un fichier EDF depuis le miroir S3 (idempotent)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"S{subject:03d}R{run:02d}.edf"
    if out.exists() and out.stat().st_size > 1000:
        return out
    url = eegbci_url(subject, run)
    delay = 2
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            out.write_bytes(data)
            return out
        except Exception:
            if attempt == retries - 1:
                raise
            import time
            time.sleep(delay)
            delay *= 2
    return out


def _standardize_eegbci(raw):
    """Nettoie les noms de canaux PhysioNet et applique le montage 10-05."""
    import mne

    mne.datasets.eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    on_missing="warn", verbose="error")
    return raw


# ------------------------------------------------------------------ sources
def iter_eegbci(n_subjects: int = 40, runs=(1, 2), cache: SubjectCache | None = None,
                raw_dir: Path | str = "cache/eegbci_raw", l_freq: float = 1.0,
                h_freq: float = 40.0, subjects: list[int] | None = None,
                epoch_length: float | None = 2.0,
                reject_ptp: float | str | None = "auto",
                verbose: bool = True):
    """Genere les SubjectRecord d'EEGBCI (une condition par run).

    group = 'EO' (run 1, yeux ouverts) ou 'EC' (run 2, yeux fermes).
    Les donnees sont decoupees en epochs de longueur fixe et les epochs
    artefactees sont rejetees (pic-a-pic) ; les bords d'epoch sont conserves
    dans `boundaries` pour ne pas fausser les durees de microstates.
    """
    import mne

    from .preprocess import clean_epochs, preprocess_raw

    mne.set_log_level("error")
    cache = cache or SubjectCache(tag="eegbci")
    subs = subjects if subjects is not None else list(range(1, n_subjects + 1))
    labels = {1: "EO", 2: "EC"}
    for s in subs:
        for run in runs:
            key = f"S{s:03d}R{run:02d}"
            if cache.has(key):
                yield cache.load(key)
                continue
            try:
                path = download_eegbci(s, run, raw_dir)
                raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
                _standardize_eegbci(raw)
                raw = preprocess_raw(raw, l_freq, h_freq)
                if epoch_length:
                    data, bounds, dropped = clean_epochs(raw, epoch_length, reject_ptp)
                else:
                    data, bounds, dropped = raw.get_data(), np.array([0], int), 0
                rec = SubjectRecord(subject=f"S{s:03d}", group=labels.get(run, str(run)),
                                    data=data.astype(np.float32),
                                    sfreq=float(raw.info["sfreq"]),
                                    ch_names=list(raw.ch_names),
                                    boundaries=bounds.astype(int),
                                    extra=dict(run=run, dropped_epochs=dropped))
                cache.save(key, rec)
                if verbose:
                    print(f"  {key}: {rec.data.shape} @ {rec.sfreq} Hz "
                          f"({dropped} epochs rejetees)")
                yield rec
            except Exception as exc:  # sujet indisponible : on continue
                if verbose:
                    print(f"  {key}: echec ({exc})")
                continue


def iter_synthetic(n_subjects: int = 20, duration: float = 60.0, snr: float = 1.0,
                   n_states: int = 4, seed: int = 0, sfreq: float = 250.0,
                   cache: SubjectCache | None = None, l_freq: float = 1.0,
                   h_freq: float = 40.0, verbose: bool = True,
                   mean_dur_g1: float = 0.085, mean_dur_g2: float = 0.065,
                   trans_boost: float = 2.5):
    """Genere les SubjectRecord de la cohorte synthetique (+ ground truth).

    `mean_dur_g2` / `trans_boost` : amplitude des deux composantes de l'effet
    de groupe (cf. `synthetic.simulate_dataset`). Elles entrent dans la cle de
    cache, sans quoi deux points d'une courbe de sensibilite se recouvriraient
    silencieusement.
    """
    import mne

    from .preprocess import preprocess_raw
    from .synthetic import simulate_dataset

    mne.set_log_level("error")
    tag = (f"synth_n{n_subjects}_d{int(duration)}_snr{snr}_k{n_states}_s{seed}"
           f"_d1{mean_dur_g1:g}_d2{mean_dur_g2:g}_tb{trans_boost:g}")
    cache = cache or SubjectCache(tag=tag)
    subs, group_maps = simulate_dataset(n_subjects=n_subjects, duration=duration,
                                        n_states=n_states, sfreq=sfreq, snr=snr,
                                        seed=seed, mean_dur_g1=mean_dur_g1,
                                        mean_dur_g2=mean_dur_g2,
                                        trans_boost=trans_boost)
    np.save(cache.root / "ground_truth_maps.npy", group_maps)
    out = []
    for s in subs:
        raw = preprocess_raw(s.raw, l_freq, h_freq)
        rec = SubjectRecord(subject=s.subject, group=s.group,
                            data=raw.get_data().astype(np.float32),
                            sfreq=float(raw.info["sfreq"]),
                            ch_names=list(raw.ch_names),
                            boundaries=np.array([0], dtype=int),
                            extra=dict(true_labels=s.labels.tolist()[:0]))
        rec.extra["true_maps"] = s.maps.tolist()
        rec.extra["true_labels_path"] = str(cache.root / f"{s.subject}_labels.npy")
        np.save(rec.extra["true_labels_path"], s.labels)
        out.append(rec)
        if verbose:
            print(f"  {s.subject} ({s.group}): {rec.data.shape}")
    return out, group_maps


# ------------------------------------------------------------------- ds004504
# Nomenclature ancienne du 10-20 clinique -> noms du montage MNE standard.
_DS004504_RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
DS004504_GROUPS = {"A": "AD", "F": "FTD", "C": "CTR"}


def read_participants(root: Path | str) -> dict:
    """participants.tsv -> {sub-XXX: {group, age, gender, mmse}}."""
    root = Path(root)
    lines = (root / "participants.tsv").read_text().splitlines()
    header = lines[0].split("\t")
    idx = {name: i for i, name in enumerate(header)}
    out = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split("\t")
        sid = f[idx["participant_id"]].strip()
        out[sid] = dict(group=DS004504_GROUPS.get(f[idx["Group"]].strip(),
                                                  f[idx["Group"]].strip()),
                        age=float(f[idx["Age"]]), gender=f[idx["Gender"]].strip(),
                        mmse=float(f[idx["MMSE"]]))
    return out


def iter_ds004504(root: Path | str = "cache/ds004504", groups=("AD", "CTR"),
                  cache: SubjectCache | None = None, l_freq: float = 1.0,
                  h_freq: float = 40.0, epoch_length: float | None = 2.0,
                  reject_ptp: float | str | None = "auto", max_subjects=None,
                  verbose: bool = True):
    """Genere les SubjectRecord de ds004504 (AD / FTD / temoins, 19 canaux).

    Miltiadous et al., Data 8(6):95, 2023. Repos yeux fermes, 500 Hz, ~10 min
    par sujet, 36 AD / 23 FTD / 29 temoins.

    On lit les donnees BRUTES (`sub-*`), pas les derivees : celles-ci ont deja
    subi filtrage, ASR et ICA, ce qui ferait double emploi avec `preprocess_raw`
    et rendrait l'effet de notre propre pretraitement ininterpretable.

    Attention : 19 electrodes en 10-20, contre 64 pour la cohorte synthetique et
    EEGBCI. Les resultats ne sont donc PAS comparables terme a terme a ceux de
    `results/synthetic/` — rendre une image 32x32 a partir de 19 capteurs rend
    la quasi-totalite des pixels interpolee.
    """
    import mne

    from .preprocess import clean_epochs, preprocess_raw

    mne.set_log_level("error")
    root = Path(root)
    cache = cache or SubjectCache(tag="ds004504")
    meta = read_participants(root)
    keep = [s for s, m in sorted(meta.items()) if m["group"] in groups]
    if max_subjects:
        keep = keep[:max_subjects]

    for sid in keep:
        m = meta[sid]
        key = f"{sid}_{m['group']}"
        if cache.has(key):
            yield cache.load(key)
            continue
        path = root / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        try:
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="error")
            raw.rename_channels({k: v for k, v in _DS004504_RENAME.items()
                                 if k in raw.ch_names})
            raw.set_montage("standard_1020", on_missing="warn")
            raw = preprocess_raw(raw, l_freq, h_freq)
            if epoch_length:
                data, bounds, dropped = clean_epochs(raw, epoch_length, reject_ptp)
            else:
                data, bounds, dropped = raw.get_data(), np.array([0], int), 0
            rec = SubjectRecord(subject=sid, group=m["group"],
                                data=data.astype(np.float32),
                                sfreq=float(raw.info["sfreq"]),
                                ch_names=list(raw.ch_names),
                                boundaries=bounds.astype(int),
                                extra=dict(age=m["age"], gender=m["gender"],
                                           mmse=m["mmse"], dropped_epochs=dropped))
            cache.save(key, rec)
            if verbose:
                print(f"  {sid} ({m['group']}): {rec.data.shape} @ {rec.sfreq} Hz "
                      f"({dropped} epochs rejetees)", flush=True)
            yield rec
        except Exception as exc:
            if verbose:
                print(f"  {sid}: echec ({exc})", flush=True)
            continue
