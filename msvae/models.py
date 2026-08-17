"""VAE convolutionnel (images topographiques) et VAE dense (vecteurs).

Le VAE dense est la brique de l'ablation centrale : meme dimension latente,
meme protocole, meme budget de parametres — seule change la representation
d'entree (64 canaux bruts vs image 32x32 interpolee) et donc le biais
inductif de l'architecture.

Invariance de polarite (deux mecanismes complementaires) :
  1. loss de reconstruction invariante  L = min(||D(E(x))-x||^2, ||D(E(x))+x||^2)
  2. loss de consistance latente        ||mu(x) - mu(-x)||^2
Le point 2 est indispensable : le point 1 seul n'impose rien a l'encodeur, or
tout le pipeline aval (k-means dans le latent) suppose que x et -x tombent au
meme endroit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- config
@dataclass
class VAEConfig:
    kind: str = "conv"          # 'conv' | 'dense'
    latent_dim: int = 8
    image_size: int = 32
    n_channels_eeg: int = 64    # utilise par le VAE dense
    base_width: int = 16        # conv : largeur du premier bloc
    n_blocks: int = 3           # conv : nombre de blocs stride-2
    kernel_size: int = 4        # conv : taille de noyau
    hidden_width: int = 256     # dense : largeur des couches cachees
    n_hidden: int = 2           # dense : nombre de couches cachees
    beta: float = 1e-3          # poids de la KL
    lambda_pol: float = 1.0     # poids de la consistance latente
    loss_space: str = "image"   # conv : 'image' | 'topo'
    # 'image' : MSE sur les pixels du disque. Attention, cela pondere le scalp
    #   par l'AIRE en pixels et non par les electrodes : le modele optimise
    #   alors la fidelite a des valeurs interpolees, donc inventees.
    # 'topo'  : la sortie du decodeur est ramenee aux n_ch electrodes par
    #   l'operateur lineaire fixe (matrice de lecture) et la MSE y est
    #   calculee. L'objectif devient identique a celui du VAE dense, ce qui
    #   rend l'ablation strictement comparable.

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------- base class
class BaseVAE(nn.Module):
    """API commune : encode -> (mu, logvar), decode, forward."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def encode_numpy(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Encode en mode deterministe (retourne mu)."""
        self.eval()
        out = []
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32).to(self.device)
            out.append(self.encode(xb)[0].cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def decode_numpy(self, z: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(z), batch_size):
            zb = torch.as_tensor(z[i:i + batch_size], dtype=torch.float32).to(self.device)
            out.append(self.decode(zb).cpu().numpy())
        return np.concatenate(out, axis=0)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ conv VAE
class ConvVAE(BaseVAE):
    """VAE convolutionnel sur images (1, S, S)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__(cfg)
        w, nb, s = cfg.base_width, cfg.n_blocks, cfg.image_size
        k = cfg.kernel_size
        pad = (k - 2) // 2  # conserve le facteur 2 exact du stride
        chans = [1] + [w * (2 ** i) for i in range(nb)]
        enc = []
        for i in range(nb):
            enc += [nn.Conv2d(chans[i], chans[i + 1], k, stride=2, padding=pad),
                    nn.GroupNorm(min(8, chans[i + 1]), chans[i + 1]),
                    nn.SiLU()]
        self.enc = nn.Sequential(*enc)
        self.spatial = s // (2 ** nb)
        flat = chans[-1] * self.spatial ** 2
        self.fc_mu = nn.Linear(flat, cfg.latent_dim)
        self.fc_logvar = nn.Linear(flat, cfg.latent_dim)

        self.fc_dec = nn.Linear(cfg.latent_dim, flat)
        dec = []
        rch = chans[::-1]
        for i in range(nb):
            out_c = rch[i + 1]
            dec += [nn.ConvTranspose2d(rch[i], out_c, k, stride=2, padding=pad)]
            if i < nb - 1:
                dec += [nn.GroupNorm(min(8, out_c), out_c), nn.SiLU()]
        self.dec = nn.Sequential(*dec)
        self._flat_shape = (chans[-1], self.spatial, self.spatial)

    def encode(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, *self._flat_shape)
        return self.dec(h)


# ----------------------------------------------------------------- dense VAE
class DenseVAE(BaseVAE):
    """VAE dense sur le vecteur des n_ch electrodes (ablation)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__(cfg)
        d, h, nh = cfg.n_channels_eeg, cfg.hidden_width, cfg.n_hidden
        layers, prev = [], d
        for _ in range(nh):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        self.enc = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, cfg.latent_dim)
        self.fc_logvar = nn.Linear(prev, cfg.latent_dim)

        layers, prev = [], cfg.latent_dim
        for _ in range(nh):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, d)]
        self.dec = nn.Sequential(*layers)

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def decode(self, z):
        return self.dec(z)


def build_model(cfg: VAEConfig) -> BaseVAE:
    return ConvVAE(cfg) if cfg.kind == "conv" else DenseVAE(cfg)


def match_dense_to_conv(conv_cfg: VAEConfig, n_channels_eeg: int,
                        n_hidden: int = 2, tol: float = 0.05) -> VAEConfig:
    """Choisit `hidden_width` pour que le VAE dense ait ~autant de parametres
    que le VAE conv de reference (equite de l'ablation)."""
    target = ConvVAE(conv_cfg).n_params()
    lo, hi = 8, 4096
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        cfg = VAEConfig(kind="dense", latent_dim=conv_cfg.latent_dim,
                        image_size=conv_cfg.image_size,
                        n_channels_eeg=n_channels_eeg, hidden_width=mid,
                        n_hidden=n_hidden, beta=conv_cfg.beta,
                        lambda_pol=conv_cfg.lambda_pol)
        n = DenseVAE(cfg).n_params()
        best = cfg if best is None or abs(n - target) < abs(
            DenseVAE(best).n_params() - target) else best
        if abs(n - target) / target < tol:
            return cfg
        if n < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ---------------------------------------------------------------------- loss
def polarity_invariant_recon(x_hat, x, mask=None, reduction="mean"):
    """min(||xhat - x||^2, ||xhat + x||^2) par echantillon (MSE par dimension).

    `mask` : tenseur booleen broadcastable sur x, pour ne compter que les
    pixels a l'interieur du scalp.
    """
    diff_p = (x_hat - x) ** 2
    diff_m = (x_hat + x) ** 2
    if mask is not None:
        m = mask.to(diff_p.dtype).expand_as(x)
        denom = m.flatten(1).sum(1).clamp_min(1.0)
        dp = (diff_p * m).flatten(1).sum(1) / denom
        dm = (diff_m * m).flatten(1).sum(1) / denom
    else:
        dp = diff_p.flatten(1).mean(1)
        dm = diff_m.flatten(1).mean(1)
    per_sample = torch.minimum(dp, dm)
    if reduction == "mean":
        return per_sample.mean()
    return per_sample


def kl_divergence(mu, logvar):
    """KL(q||N(0,I)) sommee sur les dimensions latentes, moyennee sur le batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()


def vae_loss(model: BaseVAE, x, mask=None, beta: float | None = None,
             lambda_pol: float | None = None, readout=None):
    """Loss complete + diagnostics.

    Un seul passage encodeur sur x et un sur -x (cout ~2x l'encodeur) :
    le second sert a la fois a la consistance latente et a mesurer
    l'invariance residuelle.

    `readout` : matrice fixe (n_ch, S*S) ramenant une image aux electrodes.
    Si elle est fournie (cfg.loss_space == 'topo'), la reconstruction est
    evaluee sur les n_ch mesures reelles et non sur les pixels interpoles.
    """
    cfg = model.cfg
    beta = cfg.beta if beta is None else beta
    lambda_pol = cfg.lambda_pol if lambda_pol is None else lambda_pol

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    x_hat = model.decode(z)
    if x_hat.shape != x.shape:
        x_hat = x_hat.view(x.shape)

    if readout is not None and x.dim() == 4:
        recon = polarity_invariant_recon(x_hat.flatten(1) @ readout.T,
                                         x.flatten(1) @ readout.T)
    else:
        recon = polarity_invariant_recon(x_hat, x, mask)
    kl = kl_divergence(mu, logvar)

    mu_neg, _ = model.encode(-x)
    pol_gap = ((mu - mu_neg) ** 2).sum(1)
    scale = (mu ** 2).sum(1).detach() + (mu_neg ** 2).sum(1).detach() + 1e-8
    consistency = pol_gap.mean()
    rel_gap = (pol_gap.detach() / scale).mean()

    loss = recon + beta * kl + lambda_pol * consistency
    diag = dict(loss=float(loss.detach()), recon=float(recon.detach()),
                kl=float(kl.detach()), pol=float(consistency.detach()),
                pol_rel=float(rel_gap))
    return loss, diag
