"""Orchestration : donnees -> pics -> images -> VAE -> clusters -> parametres.

`run_experiment` execute la chaine complete et ecrit dans `out_dir` :
  results.json   : toutes les metriques
  maps.npz       : cartes de chaque methode
  figures/*.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from . import baseline, cluster, evaluate, microstates, plotting
from .features import PeakBank
from .models import VAEConfig, match_dense_to_conv
from .preprocess import PeakSet, extract_peaks
from .topo import TopoProjector
from .train import TrainConfig, architecture_search, train_vae


# --------------------------------------------------------------------- config
@dataclass
class ExperimentConfig:
    name: str = "synthetic"
    dataset: str = "synthetic"      # 'synthetic' | 'eegbci'
    n_subjects: int = 20
    duration: float = 60.0          # synthetique uniquement
    snr: float = 1.0                # synthetique uniquement
    n_states_true: int = 4          # synthetique uniquement
    k: int = 4                      # K du clustering
    latent_dim: int = 8
    image_size: int = 32
    val_frac: float = 0.25
    n_per_subject: int | None = None
    balance_percentile: float = 10.0
    epochs: int = 60
    arch_search: bool = True
    search_epochs: int = 25         # recherche d'archi : entrainement raccourci
    search_max_peaks: int = 12000   # ... et sur un sous-echantillon
    min_segment_ms: float = 30.0
    stability_repeats: int = 10
    stability_refit: bool = False   # re-entraine le VAE dans le split-half
    run_pycrostates: bool = True
    seed: int = 0
    out_dir: str = "results/synthetic"

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentState:
    """Objets intermediaires, utiles en interactif / notebook."""
    records: list = field(default_factory=list)
    bank: PeakBank | None = None
    info: object = None
    models: dict = field(default_factory=dict)
    maps: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)


# ----------------------------------------------------------------- helpers
def info_from_record(rec, montage: str | None = None):
    import mne

    montage = montage or rec.extra.get("montage", "standard_1005")
    info = mne.create_info(list(rec.ch_names), rec.sfreq, "eeg")
    info.set_montage(mne.channels.make_standard_montage(montage),
                     on_missing="ignore", verbose="error")
    return info


def load_records(cfg: ExperimentConfig):
    """Retourne (records, ground_truth_maps | None)."""
    from .data import iter_eegbci, iter_synthetic

    if cfg.dataset == "synthetic":
        recs, gt = iter_synthetic(n_subjects=cfg.n_subjects, duration=cfg.duration,
                                  snr=cfg.snr, n_states=cfg.n_states_true,
                                  seed=cfg.seed)
        for r in recs:
            r.extra["montage"] = "biosemi64"
        return recs, gt
    recs = list(iter_eegbci(n_subjects=cfg.n_subjects))
    for r in recs:
        r.extra["montage"] = "standard_1005"
    return recs, None


def build_bank(records, image_size: int, min_distance: int = 3) -> tuple[PeakBank, object]:
    """Extrait les pics de GFP de tous les sujets et construit la banque."""
    import mne

    info = info_from_record(records[0])
    projector = TopoProjector.from_info(info, size=image_size)
    peaksets = []
    for rec in records:
        raw = mne.io.RawArray(rec.data.astype(np.float64), info, verbose="error")
        # identifiant unique par sujet ; la condition (groupe) reste separee
        ps = extract_peaks(raw, subject=rec.subject, group=rec.group,
                           min_distance=min_distance)
        ps.times = ps.times + 0  # indices locaux au bloc
        peaksets.append(ps)
    bank = PeakBank.from_peaksets(peaksets, projector)
    return bank, info


def default_grid(cfg: ExperimentConfig, n_ch: int) -> list[VAEConfig]:
    """Grille reduite (contrainte CPU) : latent x beta x largeur."""
    grid = []
    for latent in (4, 8, 16):
        for beta in (1e-4, 1e-3, 1e-2):
            grid.append(VAEConfig(kind="conv", latent_dim=latent,
                                  image_size=cfg.image_size, beta=beta,
                                  base_width=16, n_blocks=3,
                                  n_channels_eeg=n_ch))
    for width in (8, 32):
        grid.append(VAEConfig(kind="conv", latent_dim=cfg.latent_dim,
                              image_size=cfg.image_size, beta=1e-3,
                              base_width=width, n_blocks=3, n_channels_eeg=n_ch))
    return grid


def fit_models(cfg: ExperimentConfig, bank: PeakBank, out: Path,
               tcfg: TrainConfig | None = None):
    """Recherche d'archi (split par sujet) puis entrainement final sur tous
    les sujets, pour le VAE conv et le VAE dense apparie en parametres."""
    n_ch = bank.topo.shape[1]
    tcfg = tcfg or TrainConfig(epochs=cfg.epochs, seed=cfg.seed)
    mask = bank.projector.mask

    train_bank, val_bank = bank.split_subjects(cfg.val_frac, seed=cfg.seed)
    train_bal = train_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                    seed=cfg.seed)
    val_bal = val_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                seed=cfg.seed)

    search = None
    if cfg.arch_search:
        grid = default_grid(cfg, n_ch)
        # sous-echantillonnage + entrainement raccourci : la recherche compare
        # des architectures, elle n'a pas besoin de la convergence finale
        rng = np.random.default_rng(cfg.seed)
        sel = np.arange(len(train_bal))
        if len(sel) > cfg.search_max_peaks:
            sel = rng.choice(sel, cfg.search_max_peaks, replace=False)
        stcfg = TrainConfig(**{**tcfg.__dict__, "epochs": cfg.search_epochs,
                               "verbose": False})
        search = architecture_search(grid, train_bal.images[sel], val_bal.images,
                                     mask, stcfg)
        best_cfg = search[0]["cfg"]
        print(f"  meilleure archi : latent={best_cfg.latent_dim} beta={best_cfg.beta} "
              f"width={best_cfg.base_width} (val_loss={search[0]['val_loss']:.4f})")
    else:
        best_cfg = VAEConfig(kind="conv", latent_dim=cfg.latent_dim,
                             image_size=cfg.image_size, beta=1e-3,
                             n_channels_eeg=n_ch)

    dense_cfg = match_dense_to_conv(best_cfg, n_ch)

    # entrainement final : tous les sujets, equilibres
    full_bal = bank.balanced(cfg.n_per_subject, cfg.balance_percentile, seed=cfg.seed)
    print(f"  entrainement final sur {len(full_bal)} pics "
          f"({len(full_bal.subjects)} sujets)")
    res_conv = train_vae(best_cfg, full_bal.images, val_bal.images, mask, tcfg)
    res_dense = train_vae(dense_cfg, full_bal.topo, val_bal.topo, None, tcfg)

    (out / "figures").mkdir(parents=True, exist_ok=True)
    plotting.plot_training(res_conv.history, out / "figures" / "training_conv.png")
    plotting.plot_training(res_dense.history, out / "figures" / "training_dense.png")

    # sauvegarde des modeles : permet de refaire les analyses aval sans
    # re-entrainer (l'entrainement est le poste de cout dominant sur CPU)
    import torch
    torch.save(dict(state=res_conv.model.state_dict(), cfg=best_cfg.to_dict()),
               out / "model_conv.pt")
    torch.save(dict(state=res_dense.model.state_dict(), cfg=dense_cfg.to_dict()),
               out / "model_dense.pt")

    return dict(conv=res_conv, dense=res_dense, conv_cfg=best_cfg,
                dense_cfg=dense_cfg, search=search,
                train_bank=train_bank, val_bank=val_bank, full_bal=full_bal)


# --------------------------------------------------------------- evaluation
def backfit_all(records, maps, min_segment_ms: float, latent_fit=None):
    """Back-projection sur tous les sujets -> parametres par sujet.

    `latent_fit` : None pour le back-fitting topographique standard (argmax de
    |correlation spatiale|), ou un tuple (kind, model, centroids, projector,
    image_scale) pour affecter chaque echantillon dans l'espace latent.
    """
    params, gevs, groups, subjects = [], [], [], []
    for rec in records:
        data = rec.data.astype(np.float64)
        if latent_fit is None:
            seg = microstates.backfit(data, maps, rec.sfreq,
                                      min_segment_ms=min_segment_ms)
        else:
            kind, model, centroids, projector, image_scale = latent_fit
            z = cluster.encode_continuous(model, data, projector, kind, image_scale)
            lab, score = cluster.latent_assignment(z, centroids)
            seg = microstates.backfit_from_labels(data, maps, lab, score,
                                                  rec.sfreq, min_segment_ms)
        p = microstates.microstate_parameters(seg, rec.boundaries)
        params.append(p)
        gevs.append(p["gev_total"])
        groups.append(rec.group)
        subjects.append(rec.subject)
    return params, np.array(gevs), np.array(groups), np.array(subjects)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_experiment(cfg: ExperimentConfig, tcfg: TrainConfig | None = None
                   ) -> ExperimentState:
    t_start = time.time()
    out = Path(cfg.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    state = ExperimentState()
    results = {"config": cfg.to_dict()}

    print(f"[1/7] donnees ({cfg.dataset})")
    records, gt_maps = load_records(cfg)
    state.records = records
    print(f"  {len(records)} enregistrements")

    print("[2/7] pics de GFP + images")
    bank, info = build_bank(records, cfg.image_size)
    state.bank, state.info = bank, info
    counts = [int((bank.subject == s).sum()) for s in bank.subjects]
    results["peaks"] = dict(total=len(bank), per_subject_median=float(np.median(counts)),
                            per_subject_min=int(np.min(counts)),
                            per_subject_max=int(np.max(counts)),
                            n_subjects=len(bank.subjects))
    print(f"  {len(bank)} pics, mediane {np.median(counts):.0f}/sujet")
    plotting.plot_images(bank.images[:16, 0], out / "figures" / "example_images.png")

    # fidelite de la representation image (l'interpolation perd-elle de l'info ?)
    rec_topo = bank.projector.to_topo(bank.images[:2000, 0] * bank.image_scale,
                                      method="pinv")
    r = np.array([np.corrcoef(a, b)[0, 1]
                  for a, b in zip(bank.topo[:2000], rec_topo)])
    results["image_roundtrip_r"] = float(np.mean(r))
    print(f"  fidelite topo->image->topo : r = {np.mean(r):.4f}")

    print("[3/7] entrainement des VAE")
    fit = fit_models(cfg, bank, out, tcfg)
    state.models = fit
    results["models"] = dict(
        conv=dict(cfg=fit["conv_cfg"].to_dict(),
                  n_params=fit["conv"].model.n_params(),
                  best_val=fit["conv"].best_val, seconds=fit["conv"].seconds,
                  final=fit["conv"].history[-1]),
        dense=dict(cfg=fit["dense_cfg"].to_dict(),
                   n_params=fit["dense"].model.n_params(),
                   best_val=fit["dense"].best_val, seconds=fit["dense"].seconds,
                   final=fit["dense"].history[-1]))
    if fit["search"] is not None:
        results["arch_search"] = [
            dict(cfg=s["cfg"].to_dict(), val_loss=s["val_loss"],
                 val_recon=s["val_recon"]) for s in fit["search"]]

    print("[4/7] clustering latent + decodage")
    maps_by_method = {}
    latent, centroids_by_method = {}, {}
    for kind, res in (("conv", fit["conv"]), ("dense", fit["dense"])):
        for mode in ("two_stage", "weighted"):
            m, z, lab, _, cen = cluster.decoded_maps_from_bank(
                res.model, bank, cfg.k, mode=mode, seed=cfg.seed, kind=kind)
            maps_by_method[f"vae_{kind}_{mode}"] = m
            latent[f"{kind}_{mode}"] = (z, lab)
            centroids_by_method[f"vae_{kind}_{mode}"] = (kind, res.model, cen)
    z, lab = latent["conv_two_stage"]
    plotting.plot_latent(z, lab, out / "figures" / "latent_conv.png",
                         "espace latent (VAE conv), clusters k-means")

    if cfg.run_pycrostates:
        print("[5/7] baseline Pycrostates")
        try:
            maps_by_method["pycrostates"] = baseline.modkmeans_group(
                bank, info, cfg.k, seed=cfg.seed, two_stage=True)
        except Exception as exc:
            print(f"  echec Pycrostates : {exc}")
    if gt_maps is not None:
        maps_by_method["ground_truth"] = gt_maps
    state.maps = maps_by_method

    print("[6/7] evaluation")
    pos3d = np.array([info["chs"][i]["loc"][:3] for i in range(len(info["ch_names"]))])
    ref_name = "pycrostates" if "pycrostates" in maps_by_method else "vae_conv_two_stage"
    reference = maps_by_method[ref_name]

    aligned = {}
    for name, m in maps_by_method.items():
        am, _, _ = evaluate.align_maps(m, reference)
        aligned[name] = am
    plotting.plot_maps_grid(aligned, info, out / "figures" / "maps_comparison.png",
                            titles=[f"K{i + 1}" for i in range(cfg.k)])

    results["sanity_canonical"] = {
        name: _jsonable(evaluate.sanity_check_canonical(m, pos3d))
        for name, m in maps_by_method.items()}
    results["cross_method_corr"] = {
        f"{a}|{b}": _jsonable(evaluate.match_maps(maps_by_method[a], maps_by_method[b])[1])
        for a in maps_by_method for b in maps_by_method if a < b}

    # back-projection et parametres
    # methodes de back-fitting : topographique pour toutes les cartes, plus
    # une variante "affectation dans le latent" pour les modeles VAE
    backfit_jobs = [(name, m, None) for name, m in maps_by_method.items()]
    for name in ("vae_conv_two_stage", "vae_dense_two_stage"):
        if name in centroids_by_method:
            kind, model, cen = centroids_by_method[name]
            backfit_jobs.append((f"{name}_latentfit", maps_by_method[name],
                                 (kind, model, cen, bank.projector,
                                  bank.image_scale)))

    results["backfit"] = {}
    params_store = {}
    for name, m, latent_fit in backfit_jobs:
        params, gevs, groups, subs = backfit_all(records, m, cfg.min_segment_ms,
                                                 latent_fit)
        params_store[name] = (params, groups, subs)
        agg = dict(
            gev_total_mean=float(np.mean(gevs)), gev_total_std=float(np.std(gevs)),
            duration_ms=_jsonable(np.nanmean([p["mean_duration_ms"] for p in params], 0)),
            occurrence=_jsonable(np.mean([p["occurrence_per_s"] for p in params], 0)),
            coverage=_jsonable(np.mean([p["coverage"] for p in params], 0)),
            gev_per_class=_jsonable(np.mean([p["gev"] for p in params], 0)),
            transition=_jsonable(np.mean([p["transition"] for p in params], 0)),
            entropy_bits=float(np.mean([p["entropy_bits"] for p in params])),
            entropy_rate_bits=float(np.mean([p["entropy_rate_bits"] for p in params])),
            lzc=float(np.mean([p["lzc"] for p in params])))
        results["backfit"][name] = agg
        print(f"  {name:24s} GEV={agg['gev_total_mean']:.3f} "
              f"duree={np.nanmean(agg['duration_ms']):.0f}ms "
              f"LZC={agg['lzc']:.3f}")

    # comparaison de groupes
    results["group_comparison"] = {}
    for name, (params, groups, subs) in params_store.items():
        if len(np.unique(groups)) == 2:
            try:
                results["group_comparison"][name] = _jsonable(
                    evaluate.group_comparison(params, groups))
            except Exception as exc:
                results["group_comparison"][name] = str(exc)

    print("[7/7] stabilite (split-half sur les sujets)")
    results["stability"] = run_stability(cfg, bank, info, fit, tcfg)

    results["seconds_total"] = time.time() - t_start
    state.results = results
    np.savez_compressed(out / "maps.npz", **maps_by_method)
    (out / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    print(f"ok — {out}/results.json ({results['seconds_total']:.0f}s)")
    return state


def run_stability(cfg: ExperimentConfig, bank, info, fit, tcfg=None) -> dict:
    """Stabilite des prototypes entre deux moities disjointes de la cohorte."""
    out = {}
    mask = bank.projector.mask

    def make_vae_fn(kind, model=None):
        def fn(subs):
            sub_bank = bank.subset(np.isin(bank.subject, subs))
            m = model
            if m is None:  # re-entrainement complet sur la moitie
                bal = sub_bank.balanced(cfg.n_per_subject, cfg.balance_percentile,
                                        seed=cfg.seed)
                x = bal.images if kind == "conv" else bal.topo
                mcfg = fit["conv_cfg"] if kind == "conv" else fit["dense_cfg"]
                m = train_vae(mcfg, x, None, mask if kind == "conv" else None,
                              tcfg).model
            return cluster.decoded_maps_from_bank(m, sub_bank, cfg.k,
                                                  mode="two_stage",
                                                  seed=cfg.seed, kind=kind)[0]
        return fn

    n_rep = cfg.stability_repeats
    for kind in ("conv", "dense"):
        model = None if cfg.stability_refit else fit[kind].model
        reps = max(2, n_rep // 4) if cfg.stability_refit else n_rep
        out[f"vae_{kind}"] = _jsonable(evaluate.split_half_stability(
            make_vae_fn(kind, model), bank.subject, n_repeats=reps, seed=cfg.seed))
        out[f"vae_{kind}"]["refit"] = cfg.stability_refit

    if cfg.run_pycrostates:
        def pyc_fn(subs):
            sub_bank = bank.subset(np.isin(bank.subject, subs))
            return baseline.modkmeans_group(sub_bank, info, cfg.k, seed=cfg.seed,
                                            two_stage=True, n_init=50)
        try:
            out["pycrostates"] = _jsonable(evaluate.split_half_stability(
                pyc_fn, bank.subject, n_repeats=max(3, n_rep // 3), seed=cfg.seed))
        except Exception as exc:
            out["pycrostates"] = str(exc)

    for k, v in out.items():
        if isinstance(v, dict):
            print(f"  {k:16s} stabilite |r| = {v['mean']:.3f} ± {v['std']:.3f}")
    return out
