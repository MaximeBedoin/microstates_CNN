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

import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- config
@dataclass
class VAEConfig:
    kind: str = "conv"          # 'conv' | 'dense' | 'token'
    latent_dim: int = 8
    image_size: int = 32
    n_channels_eeg: int = 64    # utilise par les VAE dense et token
    base_width: int = 16        # conv : largeur du premier bloc
    n_blocks: int = 3           # conv : nombre de blocs stride-2
    kernel_size: int = 4        # conv : taille de noyau
    hidden_width: int = 256     # dense : largeur des couches cachees
    n_hidden: int = 2           # dense : nombre de couches cachees
    d_model: int = 32           # token : dimension des tokens
    n_heads: int = 4            # token : nombre de tetes d'attention
    n_layers: int = 2           # token : nombre de blocs d'attention
    elec_pos: tuple = ()        # token : positions 3D des electrodes (n_ch, 3)
    gamma_init: float = 0.693   # token : valeur INITIALE de softplus(gamma),
    # c'est-a-dire de la localite. 0.693 = softplus(0). A faire varier pour
    # verifier que gamma est identifie : si des initialisations eloignees ne
    # convergent pas vers la meme valeur, gamma n'est pas appris et sa lecture
    # ne mesure rien (voir H11).
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


# ----------------------------------------------------------------- token VAE
class _DistanceBiasedAttention(nn.Module):
    """Attention multi-tetes biaisee par la distance inter-electrodes.

    logit_ij = q_i . k_j / sqrt(d) - softplus(gamma_h) * d_ij

    C'est le coeur de l'argument. La localite n'est plus une hypothese
    d'architecture (comme dans une convolution, ou elle est cablee et non
    negociable) mais un PARAMETRE APPRIS, lisible apres entrainement :

      gamma grand  -> le modele a choisi un traitement local ;
      gamma  ~ 0   -> attention uniforme, melange global, la localite ne sert
                      a rien.

    Si l'argument des basses frequences spatiales est correct (une topographie
    est dominee par des harmoniques spheriques de bas ordre), gamma doit tendre
    vers 0. On obtient donc une mesure la ou la demarche par elimination ne
    donnait qu'une conclusion par defaut.
    """

    def __init__(self, d_model: int, n_heads: int, gamma_init: float = 0.693):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model doit etre divisible par n_heads")
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        # `gamma_init` est la valeur voulue de softplus(gamma), donc de la
        # localite de depart ; on inverse le softplus pour la poser.
        g0 = math.log(math.expm1(max(float(gamma_init), 1e-6)))
        self.gamma = nn.Parameter(torch.full((n_heads,), g0))

    def forward(self, h, dist):
        b, n, d = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) / self.dh ** 0.5
        bias = nn.functional.softplus(self.gamma).view(1, self.h, 1, 1) * dist
        att = torch.softmax(logits - bias, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class _TokenBlock(nn.Module):
    """Bloc pre-norm : attention biaisee par la distance, puis MLP."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 2,
                 gamma_init: float = 0.693):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.att = _DistanceBiasedAttention(d_model, n_heads, gamma_init)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_ratio * d_model), nn.SiLU(),
                                 nn.Linear(mlp_ratio * d_model, d_model))

    def forward(self, h, dist):
        h = h + self.att(self.n1(h), dist)
        return h + self.mlp(self.n2(h))


class TokenVAE(BaseVAE):
    """VAE a attention sur les ELECTRODES, sans passer par une image.

    Chaque electrode est un token : (valeur mesuree, position 3D). Il n'y a
    donc aucune interpolation, aucun masque, aucun bord a padding zero, et
    aucune inversion image -> electrodes a calibrer. L'objectif est nativement
    en espace capteur, donc identique a celui du bras dense : l'ablation porte
    exactement sur l'encodeur, ce que `loss_space='topo'` cherchait a approcher.

    La position est absolue et encodee a partir des coordonnees, donc aucune
    equivariance par translation n'est imposee — deplacer un motif sur le scalp
    change bien son sens, contrairement a ce que suppose une convolution. Et
    comme un token est un couple (valeur, position), le modele est agnostique au
    montage : rien n'est cable pour 64 ou 19 electrodes en particulier.

    Le decodeur est celui du bras dense (MLP latent -> n_ch), volontairement :
    on ne veut pas confondre l'effet de l'encodeur avec celui du decodeur.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__(cfg)
        pos = torch.as_tensor(np.asarray(cfg.elec_pos, dtype=np.float32))
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("cfg.elec_pos doit etre de forme (n_ch, 3)")
        n_ch, d = pos.shape[0], cfg.d_model

        # distances normalisees par la mediane : gamma devient sans dimension,
        # donc comparable entre montages et entre jeux de donnees
        dist = torch.cdist(pos[None], pos[None])[0]
        scale = dist[dist > 0].median().clamp_min(1e-6)
        self.register_buffer("dist", dist / scale)
        self.register_buffer("pos", pos / scale)

        self.value_emb = nn.Linear(1, d)
        self.pos_emb = nn.Sequential(nn.Linear(3, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([_TokenBlock(d, cfg.n_heads,
                                                 gamma_init=cfg.gamma_init)
                                     for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.fc_mu = nn.Linear(d, cfg.latent_dim)
        self.fc_logvar = nn.Linear(d, cfg.latent_dim)

        h, nh = cfg.hidden_width, cfg.n_hidden
        layers, prev = [], cfg.latent_dim
        for _ in range(nh):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, n_ch)]
        self.dec = nn.Sequential(*layers)

    def encode(self, x):
        # x : (B, n_ch) -> tokens (B, n_ch, d)
        h = self.value_emb(x.unsqueeze(-1)) + self.pos_emb(self.pos).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h, self.dist)
        # moyenne sur les tokens : invariante par permutation, donc l'ordre
        # des electrodes ne peut pas devenir une variable cachee
        h = self.norm(h).mean(dim=1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def decode(self, z):
        return self.dec(z)

    @torch.no_grad()
    def learned_locality(self) -> np.ndarray:
        """softplus(gamma) par tete et par couche : la localite effectivement
        apprise. C'est la quantite a rapporter pour trancher H9."""
        return np.array([[float(v) for v in
                          nn.functional.softplus(blk.att.gamma).cpu().numpy()]
                         for blk in self.blocks])


def build_model(cfg: VAEConfig) -> BaseVAE:
    if cfg.kind == "conv":
        return ConvVAE(cfg)
    if cfg.kind == "token":
        return TokenVAE(cfg)
    return DenseVAE(cfg)


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


def match_token_to_conv(conv_cfg: VAEConfig, n_channels_eeg: int, elec_pos,
                        n_heads: int = 4, n_layers: int = 2, n_hidden: int = 2,
                        hidden_width: int = 256) -> VAEConfig:
    """Choisit `d_model` pour que le VAE token ait ~autant de parametres que le
    VAE conv de reference (meme exigence d'equite que `match_dense_to_conv`).

    `d_model` est contraint a un multiple de `n_heads`, donc on balaie la
    grille au lieu de dichotomiser : elle est courte.
    """
    target = ConvVAE(conv_cfg).n_params()
    best, best_gap = None, None
    for d in range(n_heads, 16 * n_heads + 1, n_heads):
        cfg = VAEConfig(kind="token", latent_dim=conv_cfg.latent_dim,
                        image_size=conv_cfg.image_size,
                        n_channels_eeg=n_channels_eeg, d_model=d,
                        n_heads=n_heads, n_layers=n_layers,
                        hidden_width=hidden_width, n_hidden=n_hidden,
                        beta=conv_cfg.beta, lambda_pol=conv_cfg.lambda_pol,
                        elec_pos=elec_pos)
        gap = abs(TokenVAE(cfg).n_params() - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = cfg, gap
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
