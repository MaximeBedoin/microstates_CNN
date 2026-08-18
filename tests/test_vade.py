import numpy as np
import torch

from msvae import vade
from msvae.models import VAEConfig, build_model


def _model(latent=4, n_ch=32, k=4):
    cfg = VAEConfig(kind="dense", latent_dim=latent, n_channels_eeg=n_ch,
                    hidden_width=32, beta=1e-3)
    m = build_model(cfg)
    vade.attach_gmm_prior(m, k)
    return m


def test_posterior_est_une_distribution():
    m = _model()
    z = torch.randn(20, 4)
    g = m.prior.posterior(z)
    assert g.shape == (20, 4)
    assert torch.allclose(g.sum(1), torch.ones(20), atol=1e-5)
    assert (g >= 0).all()


def test_kl_positive_et_finie():
    m = _model()
    mu, logvar = torch.randn(16, 4), torch.zeros(16, 4)
    z = mu + torch.randn_like(mu) * 0.1
    kl = m.prior.kl(mu, logvar, z)
    assert torch.isfinite(kl)
    assert float(kl) > -1e-6


def test_kl_minimale_quand_q_coincide_avec_une_composante():
    """Poser q(z|x) exactement sur une composante doit couter moins qu'un
    point place loin de toutes les composantes."""
    m = _model()
    with torch.no_grad():
        m.prior.mu_c.copy_(torch.eye(4) * 5.0)
        m.prior.logvar_c.zero_()
    mu_on = m.prior.mu_c[0].repeat(8, 1)
    mu_off = torch.full((8, 4), 50.0)
    logvar = torch.zeros(8, 4)
    kl_on = m.prior.kl(mu_on, logvar, mu_on)
    kl_off = m.prior.kl(mu_off, logvar, mu_off)
    assert float(kl_on) < float(kl_off)


def test_vade_loss_retourne_diagnostics():
    m = _model()
    x = torch.randn(12, 32)
    loss, diag = vade.vade_loss(m, x)
    assert torch.isfinite(loss)
    assert set(diag) == {"loss", "recon", "kl", "pol", "pol_rel",
                         "balance", "balance_max"}


def test_machinerie_polarite_conservee():
    """Le changement de prior ne doit pas faire sauter l'invariance.

    Attention a ce qui est reellement garanti : la loss complete n'est PAS
    invariante par x -> -x tant que l'encodeur n'est pas consistant, et c'etait
    deja le cas du `vae_loss` d'origine — c'est precisement le role du terme de
    consistance que de le rendre invariant au fil de l'entrainement. Ce qui est
    invariant par CONSTRUCTION, et donc testable ici, c'est le terme de
    reconstruction et le terme de consistance.
    """
    from msvae.models import polarity_invariant_recon

    x, x_hat = torch.randn(12, 32), torch.randn(12, 32)
    r_pos = polarity_invariant_recon(x_hat, x)
    r_neg = polarity_invariant_recon(x_hat, -x)
    assert abs(float(r_pos) - float(r_neg)) < 1e-6

    m = _model().eval()
    _, d_pos = vade.vade_loss(m, x)
    _, d_neg = vade.vade_loss(m, -x)
    assert abs(d_pos["pol"] - d_neg["pol"]) < 1e-5


def test_init_prior_from_latent_pose_les_composantes():
    m = _model(k=3)
    x = np.random.default_rng(0).normal(size=(200, 32)).astype(np.float32)
    avant = m.prior.mu_c.detach().clone()
    vade.init_prior_from_latent(m, x, seed=0)
    assert not torch.allclose(avant, m.prior.mu_c)
    pi = torch.softmax(m.prior.pi_logits, 0)
    assert abs(float(pi.sum()) - 1.0) < 1e-5
    assert (pi > 0).all()


def test_assign_et_component_maps():
    m = _model(k=3, n_ch=32)
    x = np.random.default_rng(1).normal(size=(50, 32)).astype(np.float32)
    lab = vade.assign(m, x)
    assert lab.shape == (50,)
    assert set(np.unique(lab)) <= {0, 1, 2}
    maps = vade.component_maps(m)
    assert maps.shape == (3, 32)
    # cartes re-referencees en moyenne et de norme 1
    assert np.allclose(maps.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(np.linalg.norm(maps, axis=1), 1.0, atol=1e-5)


def test_elbo_est_fini():
    m = _model()
    x = np.random.default_rng(2).normal(size=(64, 32)).astype(np.float32)
    assert np.isfinite(vade.elbo(m, x))


def test_batch_balance_maximal_si_composantes_equilibrees():
    g_equilibre = torch.full((100, 4), 0.25)
    g_effondre = torch.zeros(100, 4)
    g_effondre[:, 0] = 1.0
    assert float(vade.batch_balance(g_equilibre)) > float(vade.batch_balance(g_effondre))
    assert abs(float(vade.batch_balance(g_equilibre)) - np.log(4)) < 1e-5
    assert float(vade.batch_balance(g_effondre)) < 1e-4


def test_lambda_balance_abaisse_la_loss():
    """Le terme d'equilibrage est SOUSTRAIT : a etat identique, activer lambda
    doit faire baisser la loss d'exactement lambda * H(q_bar)."""
    m = _model().eval()
    x = torch.randn(16, 32)
    l0, d0 = vade.vade_loss(m, x, lambda_balance=0.0)
    l1, d1 = vade.vade_loss(m, x, lambda_balance=1.0)
    assert abs((float(l0) - float(l1)) - d0["balance"]) < 1e-4
    assert d0["balance"] == d1["balance"]      # le diagnostic ne depend pas de lambda


def test_elbo_ignore_le_terme_dequilibrage():
    """Garde-fou : l'ELBO rapporte doit etre celui du modele, sans le terme
    d'equilibrage, dont le maximum vaut log K et croitrait donc avec K."""
    m = _model(k=4).eval()
    x = np.random.default_rng(3).normal(size=(64, 32)).astype(np.float32)
    xb = torch.as_tensor(x)
    attendu, _ = vade.vade_loss(m, xb, lambda_balance=0.0)
    assert abs(vade.elbo(m, x) - float(attendu)) < 1e-4
