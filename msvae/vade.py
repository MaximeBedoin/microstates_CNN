"""VaDE : prior en melange de gaussiennes, a la place de N(0, I).

Pourquoi. Arreter l'entrainement sur la GEV aval (`--select-epoch-by-score`)
est un pansement : on reconnait que l'objectif entraine n'est pas celui qui
nous interesse, et on compense a la sortie. VaDE traite la cause — le modele
optimise une vraisemblance qui CONTIENT deja la structure en K classes, et le
k-means aval devient une lecture du modele au lieu d'un post-traitement.

Le vrai gain n'est pas la, cependant. C'est que **K devient un parametre du
modele, selectionnable par vraisemblance**. Le choix de K est un probleme
ouvert de la litterature microstates — il se fait au coude de la GEV ou par
convention a 4, et ni pycrostates ni le pipeline actuel ne le tranchent :
ils recoivent K en entree. C'est la seule question du projet ou une reponse
positive est encore accessible, toutes les autres metriques etant saturees.

Composition. Le prior est independant de l'encodeur : il se pose aussi bien
sur ConvVAE, DenseVAE que TokenVAE. Encodeur a tokens + prior en melange est
le modele final coherent.

Reference : Jiang et al., "Variational Deep Embedding", IJCAI 2017.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .models import polarity_invariant_recon


class GMMPrior(nn.Module):
    """Melange de K gaussiennes diagonales dans l'espace latent.

    Les parametres (poids, moyennes, variances) sont appris conjointement avec
    l'encodeur et le decodeur.
    """

    def __init__(self, n_components: int, latent_dim: int):
        super().__init__()
        self.k, self.d = n_components, latent_dim
        self.pi_logits = nn.Parameter(torch.zeros(n_components))
        self.mu_c = nn.Parameter(torch.randn(n_components, latent_dim) * 0.5)
        self.logvar_c = nn.Parameter(torch.zeros(n_components, latent_dim))

    def log_pi(self):
        return torch.log_softmax(self.pi_logits, dim=0)

    def log_p_z_given_c(self, z):
        """log N(z ; mu_c, diag(var_c)) -> (B, K)."""
        z = z.unsqueeze(1)                       # (B, 1, D)
        mu = self.mu_c.unsqueeze(0)              # (1, K, D)
        logvar = self.logvar_c.clamp(-10, 10).unsqueeze(0)
        return -0.5 * (self.d * math.log(2 * math.pi)
                       + (logvar + (z - mu) ** 2 / logvar.exp()).sum(-1))

    def posterior(self, z):
        """gamma_c = q(c | z), calcule par log-sum-exp -> (B, K)."""
        logits = self.log_pi().unsqueeze(0) + self.log_p_z_given_c(z)
        return torch.softmax(logits, dim=1)

    def kl(self, mu, logvar, z):
        """Terme a MINIMISER, equivalent de KL(q||prior) pour le melange.

        Somme des trois contributions de l'ELBO de VaDE, avec le signe inverse
        pour rester homogene a `kl_divergence` du prior gaussien : c'est une
        quantite positive que la loss ajoute.
        """
        gamma = self.posterior(z)                        # (B, K)
        logvar = logvar.clamp(-10, 10)
        var = logvar.exp().unsqueeze(1)                  # (B, 1, D)
        mu_ = mu.unsqueeze(1)                            # (B, 1, D)
        mu_c = self.mu_c.unsqueeze(0)                    # (1, K, D)
        logvar_c = self.logvar_c.clamp(-10, 10).unsqueeze(0)

        # E_q(c) [ KL( q(z|x) || p(z|c) ) ], hors constante
        per_c = 0.5 * (logvar_c + var / logvar_c.exp()
                       + (mu_ - mu_c) ** 2 / logvar_c.exp()).sum(-1)   # (B, K)
        t1 = (gamma * per_c).sum(1)
        # KL( q(c|x) || p(c) )
        t2 = (gamma * (torch.log(gamma + 1e-10) - self.log_pi().unsqueeze(0))).sum(1)
        # entropie de q(z|x)
        t3 = -0.5 * (1 + logvar).sum(1)
        return (t1 + t2 + t3).mean()


def attach_gmm_prior(model, n_components: int) -> None:
    """Ajoute un prior en melange a un BaseVAE deja construit (in-place)."""
    model.prior = GMMPrior(n_components, model.cfg.latent_dim).to(model.device)


def batch_balance(gamma) -> torch.Tensor:
    """Entropie de la responsabilite MOYENNEE sur le lot, H(q_bar(c)).

    Garde-fou contre l'effondrement de composantes, le mode de defaillance
    classique de VaDE : sans lui, une composante absorbe tout et les autres se
    vident, et l'echec est ensuite attribue a tort a l'architecture. Mesure sur
    donnees jouet a 4 prototypes connus : 3 composantes utilisees sur 4.

    On maximise l'entropie de la moyenne, PAS l'inverse : penaliser
    l'entropie individuelle de q(c|x) forcerait des affectations dures, et
    imposer q_bar uniforme forcerait des classes de tailles egales — ce qui
    serait faux pour des microstates, dont les couvertures sont inegales. Ce
    terme n'empeche que la mort complete d'une composante.
    """
    q_bar = gamma.mean(0)
    return -(q_bar * torch.log(q_bar + 1e-10)).sum()


def vade_loss(model, x, mask=None, beta: float | None = None,
              lambda_pol: float | None = None, readout=None,
              lambda_balance: float = 0.0):
    """Loss VaDE, avec la meme invariance de polarite que `models.vae_loss`.

    L'invariance de polarite n'est PAS optionnelle ici : elle est ce qui rend
    les cartes comparables a celles de la litterature, et le changement de
    prior ne doit pas la faire sauter. On reprend donc exactement la meme
    reconstruction invariante et le meme terme de consistance latente, seul le
    terme de prior change.
    """
    cfg = model.cfg
    beta = cfg.beta if beta is None else beta
    lambda_pol = cfg.lambda_pol if lambda_pol is None else lambda_pol

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    x_hat = model.decode(z)
    if readout is not None:
        recon = polarity_invariant_recon(x_hat.flatten(1) @ readout.T,
                                         x.flatten(1) @ readout.T)
    else:
        recon = polarity_invariant_recon(x_hat, x, mask)

    mu_neg, _ = model.encode(-x)
    consistency = ((mu - mu_neg) ** 2).sum(1).mean()
    denom = (mu ** 2).sum(1).mean() + (mu_neg ** 2).sum(1).mean() + 1e-12
    kl = model.prior.kl(mu, logvar, z)

    loss = recon + beta * kl + lambda_pol * consistency
    gamma = model.prior.posterior(z)
    balance = batch_balance(gamma)
    if lambda_balance:
        loss = loss - lambda_balance * balance
    diag = dict(loss=float(loss.detach()), recon=float(recon.detach()),
                kl=float(kl.detach()), pol=float(consistency.detach()),
                pol_rel=float((consistency / denom).detach()),
                # entropie max = log K : proche de log K, toutes les
                # composantes vivent ; qui s'effondre le fait voir ici
                balance=float(balance.detach()),
                balance_max=float(np.log(model.prior.k)))
    return loss, diag


@torch.no_grad()
def init_prior_from_latent(model, x: np.ndarray, seed: int = 0,
                           batch_size: int = 4096) -> None:
    """Initialise le melange par k-means sur le latent du modele pre-entraine.

    Etape non negociable en pratique : VaDE initialise au hasard converge vers
    des solutions degenerees (une composante absorbe tout, les autres se vident)
    et l'echec est alors attribue a tort a l'architecture. Le protocole standard
    est : pre-entrainer avec le prior gaussien, puis poser le melange sur les
    positions trouvees par k-means.
    """
    from sklearn.cluster import KMeans

    z = model.encode_numpy(np.asarray(x, dtype=np.float32), batch_size=batch_size)
    km = KMeans(n_clusters=model.prior.k, n_init=20, random_state=seed).fit(z)
    labels = km.labels_

    dev = model.prior.mu_c.device
    model.prior.mu_c.copy_(torch.as_tensor(km.cluster_centers_,
                                           dtype=torch.float32, device=dev))
    var = np.stack([z[labels == c].var(axis=0) if (labels == c).sum() > 1
                    else np.ones(z.shape[1]) for c in range(model.prior.k)])
    model.prior.logvar_c.copy_(torch.log(torch.as_tensor(
        np.maximum(var, 1e-4), dtype=torch.float32, device=dev)))
    counts = np.bincount(labels, minlength=model.prior.k).astype(float)
    weights = np.maximum(counts / counts.sum(), 1e-6)
    model.prior.pi_logits.copy_(torch.log(torch.as_tensor(
        weights, dtype=torch.float32, device=dev)))


@torch.no_grad()
def assign(model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Affectation aux composantes : argmax q(c | z). C'est la LECTURE du
    modele, la ou le pipeline actuel fait un k-means separe sur le latent."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(np.asarray(x[i:i + batch_size], dtype=np.float32),
                             device=model.device)
        mu, _ = model.encode(xb)
        out.append(model.prior.posterior(mu).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def component_maps(model, x: np.ndarray | None = None,
                   use_empirical: bool = True) -> np.ndarray:
    """Topographies des composantes.

    `x` fourni et `use_empirical` (defaut) : on decode la moyenne EMPIRIQUE des
    points assignes a chaque composante. Sinon on decode `mu_c`.

    La distinction n'est pas cosmetique. `mu_c` est un parametre libre du
    melange, optimise conjointement, que rien ne contraint a rester sur la
    variete des points encodes — mesure sur la cohorte t=0.35 : la derive vaut
    45 a 80 % de la norme du vecteur. Le decodeur n'a jamais vu de tels points,
    et les cartes obtenues se degradent sans qu'aucune erreur ne soit levee
    (|r| au ground truth : 0.770 pour mu_c contre 0.881 pour la moyenne
    empirique). C'est le mecanisme de H1bis transpose au latent.

    C'est aussi la seule version COMPARABLE aux autres bras, qui decodent des
    centroides de k-means, c'est-a-dire des moyennes empiriques.

    Cartes re-referencees en moyenne et normalisees. Pour un encodeur conv la
    sortie est une image : a l'appelant de la ramener aux electrodes.
    """
    model.eval()
    if x is not None and use_empirical:
        z = model.encode_numpy(np.asarray(x, dtype=np.float32))
        lab = assign(model, x)
        mu = model.prior.mu_c.detach().cpu().numpy()
        centers = np.stack([z[lab == c].mean(0) if (lab == c).any() else mu[c]
                            for c in range(model.prior.k)])
    else:
        centers = model.prior.mu_c.detach().cpu().numpy()
    maps = model.decode_numpy(centers.astype(np.float32))
    if maps.ndim > 2:
        return maps
    maps = maps - maps.mean(axis=1, keepdims=True)
    return maps / np.maximum(np.linalg.norm(maps, axis=1, keepdims=True), 1e-12)


@torch.no_grad()
def elbo(model, x: np.ndarray, mask=None, batch_size: int = 4096) -> float:
    """-ELBO moyen (plus bas = mieux). C'est le critere de selection de K.

    PIEGE, verrouille ici : cette fonction appelle `vade_loss` avec
    `lambda_balance=0`, et ce n'est pas un oubli. Le terme d'equilibrage n'est
    PAS un terme de l'ELBO ; l'entrainement peut (doit) l'utiliser, mais
    l'inclure dans la valeur rapportee rendrait les K non comparables, puisque
    son maximum vaut log K et croit donc mecaniquement avec K. On selectionnerait
    alors le plus grand K par pur artefact. Ne jamais passer `lambda_balance`
    ici.

    Autre reserve d'usage : l'ELBO tend a surestimer le nombre de composantes.
    Il doit etre confronte a un critere externe — stabilite split-half, ou
    pouvoir discriminant — et non lu seul.
    """
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb = torch.as_tensor(np.asarray(x[i:i + batch_size], dtype=np.float32),
                             device=model.device)
        loss, _ = vade_loss(model, xb, mask)
        tot += float(loss) * len(xb)
        n += len(xb)
    return tot / max(n, 1)


def fit_vade(base_model, x_train, x_val=None, n_components: int = 4,
             epochs: int = 40, lr: float = 1e-3, lambda_balance: float = 0.5,
             batch_size: int = 1024, mask=None, seed: int = 0,
             verbose: bool = True):
    """Pose un prior en melange sur un VAE DEJA entraine, puis affine.

    Le pre-entrainement n'est pas une commodite : VaDE initialise au hasard
    converge vers des solutions degenerees. Le protocole est donc toujours
    (1) VAE ordinaire, (2) k-means sur le latent pour poser le melange,
    (3) affinage avec l'objectif VaDE. On part ici d'un modele deja entraine
    par `train_vae`, ce qui garantit que le bras VaDE et le bras dont il derive
    partagent EXACTEMENT le meme encodeur de depart : le contraste entre eux
    isole l'effet du prior, et rien d'autre.
    """
    import copy

    torch.manual_seed(seed)
    model = copy.deepcopy(base_model)
    attach_gmm_prior(model, n_components)
    init_prior_from_latent(model, x_train, seed=seed)

    dev = model.device
    xt = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    m = None if mask is None else torch.as_tensor(mask).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch_size):
            xb = xt[perm[i:i + batch_size]].to(dev)
            opt.zero_grad()
            loss, diag = vade_loss(model, xb, m, lambda_balance=lambda_balance)
            loss.backward()
            opt.step()
        row = dict(epoch=ep, **{f"train_{k}": v for k, v in diag.items()})
        if x_val is not None:
            row["val_elbo"] = elbo(model, x_val, m)
        history.append(row)
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            extra = f" val_elbo={row['val_elbo']:.4f}" if x_val is not None else ""
            print(f"    ep{ep:3d} recon={diag['recon']:.4f} kl={diag['kl']:.2f} "
                  f"equilibre={diag['balance']:.3f}/{diag['balance_max']:.3f}"
                  f"{extra}", flush=True)
    return model, history
