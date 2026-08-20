#!/usr/bin/env python
"""Re-entraine UN bras d'un run jusqu'a convergence, sans toucher au reste.

Remplace `retrain_token_in_run.py`, qui ne traitait que le bras token.

Motivation. Le bras token avait ete re-entraine avec un regime corrige (80 a
300 epoques, sans early stopping) apres diagnostic de son plateau — mais conv
et dense gardaient leurs 40 epoques (30 sur ds004504), et la verification
montre que leur meilleur `val_loss` est celui de la DERNIERE epoque : ils
descendaient encore a l'arret. Un seul bras avait donc recu un budget
d'optimisation ajuste, ce qui est le piege P1 sous une autre forme.

Le sens du biais n'est meme pas determine : si entrainer plus aide, conv et
dense sont handicapes ; si entrainer plus nuit — ce qu'affirme H2 — c'est
l'inverse. D'ou ce script, qui permet de donner a chaque bras le meme
traitement : entrainement long, early stopping desactive, et refus explicite
d'interpreter un modele qui descendait encore a l'arret.

Ne recalcule que le bras vise et remplace ses cartes dans `maps.npz`. Tout le
reste — cohorte, graine, pics, cartes des autres methodes — reste identique,
ce qui preserve la comparabilite ; un run complet reintroduirait du bruit
d'initialisation (0.03 a 0.065 de GEV entre graines).

    python scripts/retrain_arm_in_run.py --run results/synth_t035_transition \\
        --arm conv --epochs 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae import cluster  # noqa: E402
from msvae.models import (VAEConfig, match_dense_to_conv,  # noqa: E402
                          match_token_to_conv)
from msvae.pipeline import (ExperimentConfig, _electrode_positions,  # noqa: E402
                            build_bank, load_records, make_downstream_score)
from msvae.train import TrainConfig, train_vae  # noqa: E402


def build_cfg(arm, cfg, bank, records, n_ch, gamma_init):
    conv_ref = VAEConfig(kind="conv", latent_dim=cfg.latent_dim,
                         image_size=cfg.image_size, beta=1e-3,
                         n_channels_eeg=n_ch, loss_space=cfg.loss_space)
    if arm == "conv":
        return conv_ref
    if arm == "dense":
        return match_dense_to_conv(conv_ref, n_ch)
    t = match_token_to_conv(conv_ref, n_ch,
                            elec_pos=_electrode_positions(records[0], n_ch),
                            n_heads=cfg.token_heads, n_layers=cfg.token_layers)
    return VAEConfig(**{**t.to_dict(), "gamma_init": gamma_init})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--arm", choices=["conv", "dense", "token"], required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=None,
                   help="defaut : 5e-4 pour token (il plateaute a 2e-3), "
                        "2e-3 sinon")
    p.add_argument("--gamma-init", type=float, default=0.693)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--force", action="store_true",
                   help="recalcule meme si le bras est deja converge")
    p.add_argument("--score-every", type=int, default=0,
                   help="evalue la GEV aval toutes les N epoques. Donne la "
                        "courbe de H2 a budget egal : on voit la qualite aval "
                        "pendant que la reconstruction continue de baisser. "
                        "Enregistrement SEUL — la selection reste sur la loss, "
                        "selectionner sur la GEV serait selectionner sur une "
                        "metrique dont on a montre qu elle classe la verite "
                        "quatrieme (P6)")
    a = p.parse_args()

    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    known = set(ExperimentConfig().to_dict())
    cfg = ExperimentConfig(**{k: v for k, v in res["config"].items() if k in known})
    # Reprise. Ces entrainements durent des dizaines de minutes et un processus
    # de fond ne survit pas a la fin de la session qui l'a lance : sans ce
    # garde, relancer la sequence recalculerait ce qui etait deja fait.
    prev = res["models"].get(a.arm, {}).get("retrained", {})
    if (not a.force and prev.get("converged")
            and prev.get("epochs", 0) >= a.epochs):
        print(f"  {a.arm} deja converge a {prev['epochs']} epoques "
              f"(recon {prev['recon_best']:.4f}) — rien a faire. "
              f"--force pour recalculer.")
        return

    lr = a.lr if a.lr is not None else (5e-4 if a.arm == "token" else 2e-3)
    print(f"run {run} | bras {a.arm} | graine {cfg.seed} | lr={lr:g}, "
          f"{a.epochs} epoques, sans early stopping", flush=True)

    records, _ = load_records(cfg)
    bank, _ = build_bank(records, cfg.image_size)
    _, val_bank = bank.split_subjects(cfg.val_frac, seed=cfg.seed)
    full_bal = bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                             seed=cfg.seed)
    val_bal = val_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                seed=cfg.seed)
    n_ch = bank.topo.shape[1]

    mcfg = build_cfg(a.arm, cfg, bank, records, n_ch, a.gamma_init)
    is_conv = a.arm == "conv"
    x = full_bal.images if is_conv else full_bal.topo
    xv = val_bal.images if is_conv else val_bal.topo
    mask = bank.projector.mask if is_conv else None
    readout = (bank.projector.readout_matrix()
               if is_conv and cfg.loss_space == "topo" else None)

    score_fn = None
    if a.score_every:
        val_records = [r_ for r_ in records if r_.subject in set(val_bank.subjects)]
        score_fn = make_downstream_score(val_bal, val_records, cfg.k,
                                         cfg.min_segment_ms, cfg.seed)
    r = train_vae(mcfg, x, xv, mask,
                  TrainConfig(epochs=a.epochs, seed=cfg.seed, device="auto",
                              batch_size=a.batch_size, verbose=False,
                              patience=10 ** 6, lr=lr,
                              score_every=a.score_every),
                  score_fn=score_fn, readout=readout)
    rec = [h["val_recon"] for h in r.history]
    best = int(np.argmin(rec))
    print(f"  reconstruction : {rec[0]:.4f} -> {min(rec):.4f} (epoque {best})")
    if best >= len(rec) - max(2, len(rec) // 10):
        print(f"  ATTENTION : optimum a l'epoque {best}/{len(rec) - 1}, le "
              f"modele descendait encore. Relancer avec plus d'epoques avant "
              f"d'interpreter.", flush=True)
    else:
        print(f"  convergence atteinte ({len(rec) - 1 - best} epoques sans "
              f"amelioration ensuite)")

    # relecture JUSTE avant l'ecriture : entre le debut du script et ici il
    # s'est ecoule des dizaines de minutes, pendant lesquelles un autre bras a
    # pu ecrire. Repartir de la version chargee au demarrage effacerait son
    # travail.
    new_maps = {f"vae_{a.arm}_{mode}": cluster.decoded_maps_from_bank(
        r.model, bank, cfg.k, mode=mode, seed=cfg.seed, kind=a.arm)[0]
        for mode in ("two_stage", "weighted")}
    npz = dict(np.load(run / "maps.npz"))
    npz.update(new_maps)
    np.savez_compressed(run / "maps.npz", **npz)
    torch.save(dict(state=r.model.state_dict(), cfg=mcfg.to_dict()),
               run / f"model_{a.arm}.pt")

    entry = dict(cfg={k: v for k, v in mcfg.to_dict().items() if k != "elec_pos"},
                 n_params=r.model.n_params(), best_val=float(r.best_val),
                 seconds=float(r.seconds), final=r.history[-1],
                 retrained=dict(lr=lr, epochs=a.epochs, recon_best=float(min(rec)),
                                best_epoch=best, converged=bool(
                                    best < len(rec) - max(2, len(rec) // 10))))
    curve = [(h["epoch"], -h["down_score"]) for h in r.history if "down_score" in h]
    if curve:
        entry["downstream_curve"] = [[int(e), float(g)] for e, g in curve]
        best_g = max(curve, key=lambda t_: t_[1])
        print(f"  GEV aval : {curve[0][1]:.4f} (ep {curve[0][0]}) -> "
              f"{curve[-1][1]:.4f} (ep {curve[-1][0]}), "
              f"max {best_g[1]:.4f} a l'epoque {best_g[0]}")
    if a.arm == "token":
        entry["learned_locality"] = r.model.learned_locality().tolist()
    res = json.loads((run / "results.json").read_text())   # idem : on relit
    res["models"][a.arm] = entry
    (run / "results.json").write_text(json.dumps(res, indent=2))
    # sidecar par bras : survit meme si results.json est ecrase par un autre
    (run / f"retrain_{a.arm}.json").write_text(json.dumps(entry, indent=2))
    print(f"ok -> vae_{a.arm}_* remplaces dans {run}/maps.npz")


if __name__ == "__main__":
    main()
