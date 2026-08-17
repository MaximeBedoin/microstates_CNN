"""Assemblage des pics de GFP de plusieurs sujets en jeux d'entrainement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import PeakSet
from .topo import TopoProjector


@dataclass
class PeakBank:
    """Pics de GFP de tous les sujets, dans les deux representations.

    topo    : (n, n_ch)  topographies normalisees par la GFP
    images  : (n, 1, S, S) images interpolees (normalisees par un scalaire
              global pour que la variance moyenne par pixel soit ~1)
    subject : (n,) identifiants sujet
    group   : (n,) etiquette de groupe
    """

    topo: np.ndarray
    images: np.ndarray
    subject: np.ndarray
    group: np.ndarray
    times: np.ndarray
    projector: TopoProjector
    image_scale: float = 1.0

    # ------------------------------------------------------------------ build
    @classmethod
    def from_peaksets(cls, peaksets: list[PeakSet], projector: TopoProjector,
                      image_scale: float | None = None) -> "PeakBank":
        topo = np.concatenate([p.data for p in peaksets], axis=0).astype(np.float32)
        subject = np.concatenate([[p.subject] * len(p.data) for p in peaksets])
        group = np.concatenate([[p.group] * len(p.data) for p in peaksets])
        times = np.concatenate([p.times for p in peaksets])
        images = projector.to_image(topo)[:, None]  # (n, 1, S, S)
        if image_scale is None:
            image_scale = float(images[:, 0][:, projector.mask].std())
        images = images / image_scale
        return cls(topo=topo, images=images.astype(np.float32), subject=subject,
                   group=group, times=times, projector=projector,
                   image_scale=image_scale)

    # ------------------------------------------------------------------ views
    def __len__(self):
        return len(self.topo)

    @property
    def subjects(self) -> np.ndarray:
        return np.unique(self.subject)

    def get(self, kind: str) -> np.ndarray:
        return self.images if kind == "conv" else self.topo

    def subset(self, idx: np.ndarray) -> "PeakBank":
        return PeakBank(topo=self.topo[idx], images=self.images[idx],
                        subject=self.subject[idx], group=self.group[idx],
                        times=self.times[idx], projector=self.projector,
                        image_scale=self.image_scale)

    # ------------------------------------------------------------- protocoles
    def split_subjects(self, val_frac: float = 0.2, seed: int = 0):
        """Split train/val PAR SUJET (jamais par image : fuite massive sinon)."""
        subs = self.subjects.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(subs)
        n_val = max(1, int(round(val_frac * len(subs))))
        val_subs = set(subs[:n_val])
        is_val = np.isin(self.subject, list(val_subs))
        return self.subset(~is_val), self.subset(is_val)

    def balanced_indices(self, n_per_subject: int | None = None,
                         percentile: float = 10.0, seed: int = 0):
        """Tire au plus `n_per_subject` pics par sujet.

        Si `n_per_subject` est None, il est fixe au percentile bas de la
        distribution du nombre de pics par sujet (par defaut le 10e), ce qui
        evite qu'un sujet tres long domine l'entrainement sans jeter trop de
        donnees.
        """
        rng = np.random.default_rng(seed)
        counts = {s: int((self.subject == s).sum()) for s in self.subjects}
        if n_per_subject is None:
            n_per_subject = int(np.percentile(list(counts.values()), percentile))
        idx = []
        for s in self.subjects:
            si = np.flatnonzero(self.subject == s)
            if len(si) > n_per_subject:
                si = rng.choice(si, n_per_subject, replace=False)
            idx.append(si)
        return np.sort(np.concatenate(idx)), n_per_subject

    def balanced(self, n_per_subject: int | None = None, percentile: float = 10.0,
                 seed: int = 0) -> "PeakBank":
        idx, _ = self.balanced_indices(n_per_subject, percentile, seed)
        return self.subset(idx)
