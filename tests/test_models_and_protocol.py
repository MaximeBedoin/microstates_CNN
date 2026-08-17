import numpy as np
import torch

from msvae.features import PeakBank
from msvae.models import (VAEConfig, ConvVAE, DenseVAE, match_dense_to_conv,
                          polarity_invariant_recon, vae_loss)
from msvae.preprocess import PeakSet, find_gfp_peaks
from msvae.topo import TopoProjector


def test_polarity_invariant_recon_is_zero_on_sign_flip():
    x = torch.randn(8, 1, 16, 16)
    assert polarity_invariant_recon(-x, x).item() < 1e-12
    assert polarity_invariant_recon(x, x).item() < 1e-12


def test_polarity_invariant_recon_matches_mse_otherwise():
    x = torch.randn(8, 5)
    y = torch.randn(8, 5)
    got = polarity_invariant_recon(y, x, reduction="none")
    exp = torch.minimum(((y - x) ** 2).mean(1), ((y + x) ** 2).mean(1))
    assert torch.allclose(got, exp)


def test_masked_recon_ignores_outside_pixels():
    x = torch.zeros(4, 1, 8, 8)
    x[:, :, :4] = 1.0
    y = x.clone()
    y[:, :, 4:] = 99.0  # erreur uniquement hors masque
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4] = 1.0
    assert polarity_invariant_recon(y, x, mask).item() < 1e-9


def test_dense_matches_conv_parameter_budget():
    conv_cfg = VAEConfig(kind="conv", latent_dim=8, image_size=32)
    dense_cfg = match_dense_to_conv(conv_cfg, 64)
    n_conv = ConvVAE(conv_cfg).n_params()
    n_dense = DenseVAE(dense_cfg).n_params()
    assert abs(n_conv - n_dense) / n_conv < 0.10


def test_vae_loss_diagnostics_present():
    cfg = VAEConfig(kind="conv", latent_dim=4, image_size=32)
    model = ConvVAE(cfg)
    loss, diag = vae_loss(model, torch.randn(4, 1, 32, 32))
    assert set(diag) == {"loss", "recon", "kl", "pol", "pol_rel"}
    assert np.isfinite(loss.item())


def test_encoder_consistency_penalty_is_zero_for_even_encoder():
    """Verifie que le terme de consistance mesure bien ce qu'on croit."""
    cfg = VAEConfig(kind="dense", latent_dim=3, n_channels_eeg=6)
    model = DenseVAE(cfg)

    class Even(torch.nn.Module):
        def forward(self, x):
            return x.abs()

    model.enc = torch.nn.Sequential(Even(), model.enc)  # E(x) = E(-x)
    _, diag = vae_loss(model, torch.randn(8, 6))
    assert diag["pol"] < 1e-10


def _bank(n_subj=4, n_peaks=50, n_ch=20, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.standard_normal((n_ch, 3))
    pos[:, 2] = np.abs(pos[:, 2]) + 1.0
    pos = 0.09 * pos / np.linalg.norm(pos, axis=1, keepdims=True)
    tp = TopoProjector.from_positions(pos, [f"E{i}" for i in range(n_ch)], size=32)
    sets = []
    for s in range(n_subj):
        n = n_peaks * (s + 1)  # nombres de pics deliberement desequilibres
        d = rng.standard_normal((n, n_ch)).astype(np.float32)
        sets.append(PeakSet(data=d, gfp=np.ones(n, np.float32),
                            times=np.arange(n), subject=f"sub{s}",
                            group="G1" if s % 2 else "G2"))
    return PeakBank.from_peaksets(sets, tp)


def test_split_is_by_subject_without_leakage():
    bank = _bank()
    tr, va = bank.split_subjects(val_frac=0.25, seed=0)
    assert set(tr.subject) & set(va.subject) == set()
    assert len(tr) + len(va) == len(bank)


def test_balancing_caps_peaks_per_subject():
    """Le percentile bas plafonne les gros sujets sans inventer de pics pour
    les petits (qui gardent tout ce qu'ils ont)."""
    bank = _bank()  # sujets a 50, 100, 150, 200 pics
    idx, cap = bank.balanced_indices(percentile=10.0, seed=0)
    bal = bank.subset(idx)
    counts = {s: int((bal.subject == s).sum()) for s in bal.subjects}
    raw = {s: int((bank.subject == s).sum()) for s in bank.subjects}
    assert max(counts.values()) <= cap
    for s in counts:
        assert counts[s] == min(raw[s], cap)
    # desequilibre fortement reduit
    assert max(counts.values()) / min(counts.values()) < 2.0
    assert max(raw.values()) / min(raw.values()) == 4.0


def test_gfp_peaks_respect_min_distance():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((16, 5000))
    idx, gfp = find_gfp_peaks(data, min_distance=3, reject_percentile=None)
    assert np.diff(idx).min() >= 3
    assert len(gfp) == 5000
