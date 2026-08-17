"""Topographie -> image 2D.

Chaine : positions 3D des electrodes (montage) -> projection azimuthale
equidistante sur le disque unite -> interpolation par splines spheriques
(Perrin et al. 1989) sur une grille reguliere -> image (H, W).

Choix documentes dans CHOIX_METHODO.md :
  * projection azimuthale equidistante (celle des topomaps MNE) ;
  * interpolation spline SPHERIQUE (et non 2D) : l'interpolation est faite
    sur la sphere, seule la grille de sortie est plane. C'est plus
    physiologique qu'une triangulation 2D type Clough-Tocher ;
  * l'exterieur du disque est mis a zero et masque dans la loss.

Toute l'operation est lineaire : image = W @ v ou v est le vecteur des
n_channels valeurs. W est precalculee une fois pour toutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ordre du polynome de Legendre et rigidite de la spline spherique.
# MNE utilise m=4 et ~50 termes ; on garde les memes valeurs.
_LEGENDRE_TERMS = 50
_STIFFNESS = 4

# Regularisation de l'inversion image -> electrodes, en fraction de
# trace(W^T W)/n_ch. Calibree empiriquement (plateau entre 3e-2 et 3e-1) sur
# des images reconstruites par le VAE ; voir CHOIX_METHODO.md.
REG_RIDGE = 0.1


def _legendre_poly_values(x: np.ndarray, n_terms: int) -> np.ndarray:
    """Valeurs P_n(x) pour n = 1..n_terms, via la recurrence de Bonnet.

    Parameters
    ----------
    x : array, shape (...)
        Cosinus des angles, dans [-1, 1].

    Returns
    -------
    array, shape (n_terms, ...)
        P_1(x) ... P_n_terms(x).
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((n_terms,) + x.shape, dtype=np.float64)
    p_prev = np.ones_like(x)  # P_0
    p_cur = x.copy()  # P_1
    out[0] = p_cur
    for n in range(1, n_terms):
        # (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}
        p_next = ((2 * n + 1) * x * p_cur - n * p_prev) / (n + 1)
        out[n] = p_next
        p_prev, p_cur = p_cur, p_next
    return out


def _calc_g(cos_angles: np.ndarray, stiffness: int = _STIFFNESS,
            n_terms: int = _LEGENDRE_TERMS) -> np.ndarray:
    """Fonction de Green g(cos gamma) des splines spheriques (Perrin 1989)."""
    n = np.arange(1, n_terms + 1, dtype=np.float64)
    coefs = (2 * n + 1) / (n ** stiffness * (n + 1) ** stiffness)
    polys = _legendre_poly_values(cos_angles, n_terms)
    return np.tensordot(coefs, polys, axes=(0, 0)) / (4.0 * np.pi)


def spherical_spline_matrix(pos_from: np.ndarray, pos_to: np.ndarray,
                            stiffness: int = _STIFFNESS,
                            reg: float = 1e-5) -> np.ndarray:
    """Matrice d'interpolation spline spherique.

    Parameters
    ----------
    pos_from : array, shape (n_from, 3)
        Positions des electrodes (seront normalisees sur la sphere unite).
    pos_to : array, shape (n_to, 3)
        Positions cibles.
    reg : float
        Regularisation (Tikhonov) ajoutee a la diagonale de G. Sans elle le
        systeme est mal conditionne des que deux electrodes sont proches.

    Returns
    -------
    array, shape (n_to, n_from)
        W telle que v_to = W @ v_from.
    """
    u_from = pos_from / np.linalg.norm(pos_from, axis=1, keepdims=True)
    u_to = pos_to / np.linalg.norm(pos_to, axis=1, keepdims=True)

    n_from = len(u_from)
    cos_ff = np.clip(u_from @ u_from.T, -1.0, 1.0)
    cos_tf = np.clip(u_to @ u_from.T, -1.0, 1.0)

    g_ff = _calc_g(cos_ff, stiffness)
    g_tf = _calc_g(cos_tf, stiffness)

    # Systeme augmente : [[G, 1], [1^T, 0]] [c; c0] = [v; 0]
    a = np.zeros((n_from + 1, n_from + 1), dtype=np.float64)
    a[:n_from, :n_from] = g_ff + reg * np.eye(n_from)
    a[:n_from, n_from] = 1.0
    a[n_from, :n_from] = 1.0
    a_inv = np.linalg.pinv(a)

    # v_to = c0 + g_tf @ c  ou [c; c0] = a_inv @ [v; 0]
    c_from_v = a_inv[:n_from, :n_from]  # (n_from, n_from)
    c0_from_v = a_inv[n_from, :n_from]  # (n_from,)
    return g_tf @ c_from_v + c0_from_v[None, :]


def azimuthal_equidistant(pos3d: np.ndarray, center: np.ndarray | None = None,
                          scale: float | None = None) -> tuple[np.ndarray, float, np.ndarray]:
    """Projection azimuthale equidistante des positions 3D sur le disque unite.

    L'angle polaire theta (depuis le vertex +z) devient le rayon 2D, l'azimut
    est conserve. Le rayon est normalise pour que l'electrode la plus basse
    tombe sur `radius_max` (0.95 par defaut via `scale`).

    Returns
    -------
    xy : array, shape (n, 2)
    scale : float
        Facteur theta -> rayon (reutilise pour l'inversion).
    center : array, shape (3,)
    """
    pos3d = np.asarray(pos3d, dtype=np.float64)
    if center is None:
        center = pos3d.mean(axis=0)
        center[2] = pos3d[:, 2].min()  # centre approx. sous la calotte
    p = pos3d - center
    r = np.linalg.norm(p, axis=1)
    theta = np.arccos(np.clip(p[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(p[:, 1], p[:, 0])
    if scale is None:
        scale = 0.95 / theta.max()
    rad = theta * scale
    xy = np.stack([rad * np.cos(phi), rad * np.sin(phi)], axis=1)
    return xy, scale, center


def _inverse_azimuthal(xy: np.ndarray, scale: float) -> np.ndarray:
    """Disque 2D -> vecteurs unitaires sur la sphere (inverse de la projection)."""
    rad = np.linalg.norm(xy, axis=1)
    theta = rad / scale
    phi = np.arctan2(xy[:, 1], xy[:, 0])
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=1)


@dataclass
class TopoProjector:
    """Projette des vecteurs de canaux en images 2D (et inversement).

    Attributes
    ----------
    ch_names : list[str]
    size : int
        Cote de l'image (carree).
    xy : array (n_ch, 2)
        Positions 2D des electrodes dans le disque unite.
    mask : array (size, size) bool
        Pixels a l'interieur du disque.
    W : array (n_inside, n_ch)
        Matrice d'interpolation (pixels internes seulement).
    S : array (n_ch, n_inside)
        Matrice d'echantillonnage bilineaire image -> electrodes.
    P : array (n_ch, n_inside)
        Inversion ridge (lambda = REG_RIDGE * trace(W^T W)/n_ch), utilisee par
        defaut. W est mal conditionnee (cond ~ 4e3) : la pseudo-inverse exacte
        amplifie violemment le residu hors-variete des images decodees par le
        VAE (|r| chute a 0.19). La ridge calibree ramene |r| a 0.87 tout en
        restant exacte a 0.997 sur les images reellement produites par
        `to_image`. Voir CHOIX_METHODO.md.
    P_exact : array (n_ch, n_inside)
        Pseudo-inverse quasi exacte : a n'utiliser que sur des images issues
        de `to_image` (verification de la fidelite de la representation).
    """

    ch_names: list
    size: int
    xy: np.ndarray
    mask: np.ndarray
    W: np.ndarray
    S: np.ndarray
    P: np.ndarray
    P_exact: np.ndarray
    scale: float
    center: np.ndarray

    # ------------------------------------------------------------------ build
    @classmethod
    def from_positions(cls, pos3d: np.ndarray, ch_names: list, size: int = 32,
                       radius: float = 1.0) -> "TopoProjector":
        xy, scale, center = azimuthal_equidistant(pos3d)

        lin = np.linspace(-radius, radius, size)
        gx, gy = np.meshgrid(lin, lin)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        inside = np.linalg.norm(grid, axis=1) <= radius
        mask = inside.reshape(size, size)

        u_to = _inverse_azimuthal(grid[inside], scale)
        u_from = _inverse_azimuthal(xy, scale)
        w = spherical_spline_matrix(u_from, u_to)

        s = cls._bilinear_sampler(xy, size, radius, inside)
        # inversions ridge : (W^T W + lambda I)^-1 W^T
        base = np.trace(w.T @ w) / w.shape[1]
        gram = w.T @ w
        eye = np.eye(w.shape[1])
        p = np.linalg.solve(gram + REG_RIDGE * base * eye, w.T)
        p_exact = np.linalg.solve(gram + 1e-6 * base * eye, w.T)
        return cls(ch_names=list(ch_names), size=size, xy=xy, mask=mask,
                   W=w.astype(np.float32), S=s.astype(np.float32),
                   P=p.astype(np.float32), P_exact=p_exact.astype(np.float32),
                   scale=scale, center=center)

    @classmethod
    def from_info(cls, info, size: int = 32) -> "TopoProjector":
        """Construit le projecteur a partir d'un `mne.Info` (canaux EEG)."""
        import mne

        picks = mne.pick_types(info, eeg=True, exclude=())
        pos = np.array([info["chs"][p]["loc"][:3] for p in picks], dtype=np.float64)
        names = [info["ch_names"][p] for p in picks]
        if not np.isfinite(pos).all() or np.allclose(pos, 0):
            raise ValueError("Positions d'electrodes absentes : montage non defini.")
        return cls.from_positions(pos, names, size=size)

    @staticmethod
    def _bilinear_sampler(xy: np.ndarray, size: int, radius: float,
                          inside: np.ndarray) -> np.ndarray:
        """Poids bilineaires pour relire l'image aux positions des electrodes."""
        n_ch = len(xy)
        s_full = np.zeros((n_ch, size * size), dtype=np.float64)
        # coordonnee continue en pixels
        step = (2 * radius) / (size - 1)
        for i, (x, y) in enumerate(xy):
            cx = (x + radius) / step
            cy = (y + radius) / step
            x0, y0 = int(np.floor(cx)), int(np.floor(cy))
            fx, fy = cx - x0, cy - y0
            for dy in (0, 1):
                for dx in (0, 1):
                    xi, yi = np.clip(x0 + dx, 0, size - 1), np.clip(y0 + dy, 0, size - 1)
                    wgt = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                    s_full[i, yi * size + xi] += wgt  # ligne = y, colonne = x
        s = s_full[:, inside]
        # Renormalise si un poids tombait hors du masque
        row = s.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return s / row

    # ------------------------------------------------------------- transforms
    @property
    def n_inside(self) -> int:
        return int(self.mask.sum())

    def to_image(self, data: np.ndarray) -> np.ndarray:
        """(n, n_ch) -> (n, size, size). Zeros hors du disque."""
        data = np.asarray(data, dtype=np.float32)
        single = data.ndim == 1
        if single:
            data = data[None]
        vals = data @ self.W.T  # (n, n_inside)
        out = np.zeros((len(data), self.size, self.size), dtype=np.float32)
        out[:, self.mask] = vals
        return out[0] if single else out

    def to_topo(self, images: np.ndarray, method: str = "ridge") -> np.ndarray:
        """(n, size, size) -> (n, n_ch).

        method : {'ridge', 'pinv', 'bilinear'}
            'ridge'    inversion regularisee (defaut, robuste aux images
                       decodees par le VAE) ;
            'pinv'     inversion quasi exacte, reservee aux images issues de
                       `to_image` ;
            'bilinear' lecture du pixel a l'aplomb de chaque electrode.
        """
        images = np.asarray(images, dtype=np.float32)
        single = images.ndim == 2
        if single:
            images = images[None]
        vals = images.reshape(len(images), -1)[:, self.mask.ravel()]
        mat = {"ridge": self.P, "pinv": self.P_exact, "bilinear": self.S}[method]
        out = vals @ mat.T
        return out[0] if single else out

    def readout_matrix(self, method: str = "ridge") -> np.ndarray:
        """Matrice (n_ch, size*size) ramenant une image entiere aux electrodes.

        Contrairement a `to_topo`, elle opere sur l'image aplatie complete
        (zeros hors masque), ce qui en fait une couche lineaire fixe utilisable
        directement dans la loss du VAE (cf. VAEConfig.loss_space == 'topo').
        """
        mat = {"ridge": self.P, "pinv": self.P_exact, "bilinear": self.S}[method]
        full = np.zeros((len(self.ch_names), self.size ** 2), dtype=np.float32)
        full[:, self.mask.ravel()] = mat
        return full

    def as_dict(self) -> dict:
        return dict(ch_names=self.ch_names, size=self.size, xy=self.xy,
                    mask=self.mask, W=self.W, S=self.S, P=self.P,
                    P_exact=self.P_exact, scale=self.scale, center=self.center)
