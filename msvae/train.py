"""Protocole d'entrainement des VAE."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .models import BaseVAE, VAEConfig, build_model, vae_loss


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    patience: int = 10          # early stopping sur la loss de validation
    warmup_beta: int = 10       # montee lineaire de beta (evite le posterior collapse)
    score_every: int = 0        # evalue le critere aval toutes les N epoques (0 = jamais)
    select_by: str = "val_loss"  # 'val_loss' | 'score' : critere de restauration du meilleur etat
    seed: int = 0
    num_threads: int = 4        # CPU uniquement
    device: str = "auto"        # 'auto' | 'cpu' | 'cuda'
    verbose: bool = True


def resolve_device(name: str = "auto") -> "torch.device":
    """Choisit le peripherique. Le code est identique CPU/GPU : sur GPU
    l'entrainement est ~15x plus rapide, sans changement de resultat attendu
    au bruit d'initialisation pres."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


@dataclass
class TrainResult:
    model: BaseVAE
    history: list = field(default_factory=list)
    best_val: float = np.inf
    best_epoch: int = -1
    seconds: float = 0.0


def _iterate(x: torch.Tensor, batch_size: int, rng: np.random.Generator,
             shuffle: bool = True):
    n = len(x)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield x[torch.as_tensor(order[i:i + batch_size])]


def train_vae(cfg: VAEConfig, x_train: np.ndarray, x_val: np.ndarray | None = None,
              mask: np.ndarray | None = None, tcfg: TrainConfig | None = None,
              score_fn=None, readout: np.ndarray | None = None) -> TrainResult:
    """Entraine un VAE.

    `mask` : masque (S, S) des pixels internes (VAE conv uniquement).
    L'augmentation de polarite (x ou -x aleatoirement) est appliquee a chaque
    batch ; combinee a la loss min(+/-) et a la consistance latente, elle rend
    la distribution d'entree exactement symetrique.
    """
    tcfg = tcfg or TrainConfig()
    torch.manual_seed(tcfg.seed)
    device = resolve_device(tcfg.device)
    if device.type == "cpu":
        torch.set_num_threads(tcfg.num_threads)
    rng = np.random.default_rng(tcfg.seed)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    xt = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    xv = None if x_val is None else torch.as_tensor(np.asarray(x_val, dtype=np.float32))
    mt = None
    if mask is not None and cfg.kind == "conv":
        mt = torch.as_tensor(mask.astype(np.float32))[None, None].to(device)
    ro = None
    if cfg.kind == "conv" and cfg.loss_space == "topo":
        if readout is None:
            raise ValueError("loss_space='topo' exige la matrice de lecture "
                             "image -> electrodes (projector.readout_matrix())")
        ro = torch.as_tensor(np.asarray(readout, dtype=np.float32)).to(device)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg.epochs)
    best_state, best_val, best_epoch = None, np.inf, -1
    history = []
    t0 = time.time()

    for epoch in range(tcfg.epochs):
        beta = cfg.beta * min(1.0, (epoch + 1) / max(tcfg.warmup_beta, 1))
        model.train()
        agg = {}
        nb = 0
        for xb in _iterate(xt, tcfg.batch_size, rng):
            xb = xb.to(device, non_blocking=True)
            sign = torch.from_numpy(
                rng.choice([-1.0, 1.0], size=(len(xb),)).astype(np.float32)).to(device)
            xb = xb * sign.view(-1, *([1] * (xb.dim() - 1)))
            opt.zero_grad(set_to_none=True)
            loss, diag = vae_loss(model, xb, mt, beta=beta, readout=ro)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in diag.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        sched.step()
        row = {"epoch": epoch, "beta": beta}
        row.update({f"train_{k}": v / nb for k, v in agg.items()})

        if xv is not None:
            model.eval()
            with torch.no_grad():
                vagg, vnb = {}, 0
                for xb in _iterate(xv, 2048, rng, shuffle=False):
                    _, diag = vae_loss(model, xb.to(device), mt,
                                          beta=cfg.beta, readout=ro)
                    for k, v in diag.items():
                        vagg[k] = vagg.get(k, 0.0) + v
                    vnb += 1
            row.update({f"val_{k}": v / vnb for k, v in vagg.items()})
            if (score_fn is not None and tcfg.score_every
                    and (epoch % tcfg.score_every == 0 or epoch == tcfg.epochs - 1)):
                row["down_score"] = float(score_fn(model, cfg))
                model.train()
            # Selection de l'etat a conserver. Les deux criteres n'ont NI la
            # meme echelle NI le meme signe : il ne faut jamais les melanger
            # dans la meme variable. En mode 'score', les epoques ou le critere
            # aval n'a pas ete evalue ne participent tout simplement pas.
            if tcfg.select_by == "score":
                current = row.get("down_score")
            else:
                current = row["val_loss"]
            if current is None:
                history.append(row)
                continue
            if current < best_val - 1e-6:
                best_val, best_epoch = current, epoch
                best_state = copy.deepcopy(model.state_dict())
            elif epoch - best_epoch >= tcfg.patience:
                history.append(row)
                if tcfg.verbose:
                    print(f"  early stop epoch {epoch} (best {best_epoch})")
                break
        history.append(row)
        if tcfg.verbose and (epoch % 5 == 0 or epoch == tcfg.epochs - 1):
            msg = " ".join(f"{k}={v:.4f}" for k, v in row.items()
                           if k.endswith(("recon", "kl", "pol_rel", "loss")))
            print(f"  ep{epoch:3d} {msg}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=history, best_val=best_val,
                       best_epoch=best_epoch, seconds=time.time() - t0)


def architecture_search(grid: list[VAEConfig], x_train, x_val, mask=None,
                        tcfg: TrainConfig | None = None, score_fn=None,
                        readout: np.ndarray | None = None):
    """Recherche d'architecture sur le split train/val (par sujet).

    Le classement N'UTILISE PAS la loss de validation : celle-ci vaut
    recon + beta * KL, donc deux configurations de beta differents optimisent
    des objectifs differents et leurs losses ne sont pas comparables (comparer
    ces valeurs revient a choisir systematiquement le plus petit beta).

    Critere retenu, par ordre de preference :
      * `score_fn(model, cfg) -> float` (plus petit = meilleur) si fournie :
        le pipeline y passe un critere AVAL — la GEV obtenue apres clustering
        latent, decodage et back-fitting sur les sujets de validation, ce qui
        est la grandeur qui nous interesse reellement ;
      * a defaut, l'erreur de reconstruction de validation, identique pour
        toutes les configurations.
    """
    out = []
    for i, cfg in enumerate(grid):
        width = cfg.base_width if cfg.kind == "conv" else cfg.hidden_width
        print(f"  [{i + 1}/{len(grid)}] {cfg.kind} latent={cfg.latent_dim} "
              f"beta={cfg.beta} width={width} blocs={cfg.n_blocks} "
              f"noyau={cfg.kernel_size} loss={cfg.loss_space}", flush=True)
        res = train_vae(cfg, x_train, x_val, mask, tcfg, score_fn=score_fn,
                        readout=readout)
        val_recon = (res.history[res.best_epoch].get("val_recon", np.nan)
                     if res.best_epoch >= 0 else np.nan)
        score = float(score_fn(res.model, cfg)) if score_fn is not None else val_recon
        print(f"      val_recon={val_recon:.4f} score={score:.4f} "
              f"({res.seconds:.0f}s)", flush=True)
        out.append({"cfg": cfg, "val_loss": res.best_val, "val_recon": val_recon,
                    "score": score, "result": res})
    return sorted(out, key=lambda r: r["score"])
