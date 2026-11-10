"""Tests for BIGN1D, ExactInverseBIGN1D and MirrorBIGN1D."""
import pytest
import torch

from atandn.layers.bign import BIGN1D, ExactInverseBIGN1D, MirrorBIGN1D


def _small_gamma(layer: BIGN1D) -> BIGN1D:
    """Make Gamma small so that unit-scale inputs are never clamped by the inverse."""
    with torch.no_grad():
        layer.gamma_raw.fill_(-3.0)
    return layer


def test_forward_bound_and_denominator():
    torch.manual_seed(0)
    m = BIGN1D(6).eval()
    x = torch.distributions.StudentT(1.5).sample((4, 6, 500)) * 50
    y = m(x)
    g = m.output_bounds()["gamma_ii_inv_sqrt"].view(1, -1, 1)
    assert (y.abs() <= g * (1 + 1e-5)).all()
    assert (m.denominator(x) >= m.beta.view(1, -1, 1)).all()
    assert torch.isfinite(y).all()


def test_gamma_symmetric_positive():
    m = BIGN1D(8)
    g = m.gamma
    assert torch.allclose(g, g.T)
    assert (g > 0).all() and (m.beta > 0).all()
    assert (m.beta >= m.beta_min).all()


def test_matches_explicit_formula():
    m = BIGN1D(5)
    x = torch.randn(2, 5, 7)
    d = m.beta.view(1, 5, 1) + torch.einsum("ij,bjt->bit", m.gamma, x ** 2)
    assert torch.allclose(m(x), x / d.sqrt(), atol=1e-6)


def test_cross_channel_suppression():
    m = BIGN1D(4).eval()
    x = torch.randn(1, 4, 10)
    x[:, 0, :] = 1e6
    y = m(x)
    g = m.gamma
    for k in range(1, 4):
        bound = x[:, k].abs() / (g[k, 0].sqrt() * 1e6)
        assert (y[:, k].abs() <= bound * (1 + 1e-5)).all()


@pytest.mark.parametrize("method,kw", [("solve", {}), ("neumann", {"neumann_steps": 40})])
@pytest.mark.parametrize("c", [1, 6, 24])
def test_inverse_round_trip(method, kw, c):
    torch.manual_seed(c)
    fwd = BIGN1D(c)
    inv = ExactInverseBIGN1D(c, tied_to=fwd, method=method, **kw)
    x = torch.randn(3, c, 50) * (0.1 if c <= 6 else 0.03)  # keep s below the clamp level
    assert torch.allclose(inv(fwd(x)), x, atol=1e-4)
    assert inv.last_clamp_rate == 0.0


@pytest.mark.parametrize("method,kw", [("solve", {}), ("neumann", {"neumann_steps": 60})])
def test_inverse_round_trip_unit_scale(method, kw):
    torch.manual_seed(1)
    fwd = _small_gamma(BIGN1D(6))
    inv = ExactInverseBIGN1D(tied_to=fwd, method=method, **kw)
    x = torch.randn(4, 6, 80) * 0.4
    assert torch.allclose(inv(fwd(x)), x, atol=1e-4)
    assert inv.last_clamp_rate == 0.0


def test_inverse_untied_with_copied_params():
    torch.manual_seed(2)
    fwd = BIGN1D(6)
    inv = ExactInverseBIGN1D(6)
    inv.beta_raw.data.copy_(fwd.beta_raw.data)
    inv.gamma_raw.data.copy_(fwd.gamma_raw.data)
    x = torch.randn(2, 6, 30) * 0.1
    assert torch.allclose(inv(fwd(x)), x, atol=1e-4)


def test_forward_of_inverse_recovers_y():
    torch.manual_seed(3)
    fwd = _small_gamma(BIGN1D(6))
    inv = ExactInverseBIGN1D(tied_to=fwd)
    y = fwd(torch.randn(2, 6, 40) * 0.4)
    assert torch.allclose(fwd(inv(y)), y, atol=1e-4)


def test_float64_solve():
    fwd = BIGN1D(6)
    inv = ExactInverseBIGN1D(tied_to=fwd, solve_dtype=torch.float64)
    x = torch.randn(2, 6, 20) * 0.1
    assert torch.allclose(inv(fwd(x)), x, atol=1e-5)


def test_tying_shares_tensors_counted_once():
    fwd = BIGN1D(6)
    inv = ExactInverseBIGN1D(tied_to=fwd)
    assert list(inv.parameters()) and all(p is q for p, q in zip(inv.parameters(), fwd.parameters()))
    holder = torch.nn.ModuleList([fwd, inv])
    assert sum(p.numel() for p in holder.parameters()) == 6 + 36
    assert torch.equal(inv.gamma, fwd.gamma)


@pytest.mark.parametrize("method", ["solve", "neumann"])
def test_gradients_finite_including_zero_input(method):
    fwd = BIGN1D(6)
    inv = ExactInverseBIGN1D(tied_to=fwd, method=method)
    x = torch.randn(2, 6, 20)
    x[:, :, :3] = 0.0
    x.requires_grad_(True)
    out = inv(fwd(x)).sum() + inv(torch.randn(2, 6, 20) * 5).sum()
    out.backward()
    assert torch.isfinite(x.grad).all()
    for p in fwd.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()


def test_clamp_rate_reported_on_large_inputs():
    inv = ExactInverseBIGN1D(6)
    out = inv(torch.randn(2, 6, 30) * 10)
    assert 0.0 < inv.last_clamp_rate <= 1.0
    assert torch.isfinite(out).all()


def test_mirror_layer():
    m = MirrorBIGN1D(6)
    x = torch.randn(2, 6, 10, requires_grad=True)
    y = m(x)
    expected = x * (m.beta.view(1, -1, 1) + torch.einsum("ij,bjt->bit", m.gamma, x ** 2)).sqrt()
    assert torch.allclose(y, expected, atol=1e-6)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


# ---- initialisation modes and clamp diagnostics ----

@pytest.mark.parametrize("c", [6, 11, 64])
def test_default_init_round_trip_low_clamp(c):
    torch.manual_seed(c)
    fwd = BIGN1D(c)
    inv = ExactInverseBIGN1D(tied_to=fwd)
    x = torch.randn(8, c, 100)
    assert torch.allclose(inv(fwd(x)), x, atol=1e-4)
    assert inv.last_clamp_rate < 0.05
    assert inv.last_max_clamped_ratio <= 1.0 + 1e-6


@pytest.mark.parametrize("c", [6, 64, 256])
def test_diag_dominant_gamma_statistics(c):
    g = BIGN1D(c).gamma
    assert torch.allclose(torch.diagonal(g), torch.full((c,), 0.5), atol=0.02)
    assert (g.sum(1) < 1.5).all()
    assert torch.allclose(g, g.T) and (g > 0).all()


def test_uniform_mode_reproduces_legacy_statistics():
    g = BIGN1D(16, gamma_init_mode="uniform").gamma
    assert torch.allclose(g, torch.full_like(g, 0.71), atol=0.06)  # diag softplus(0.1)=0.744, off-diag softplus(0)=0.693
    assert torch.allclose(g, g.T) and (g > 0).all()
    with pytest.raises(ValueError):
        BIGN1D(4, gamma_init_mode="nope")


def test_uniform_mode_clamps_heavily_at_c64():
    torch.manual_seed(0)
    inv = ExactInverseBIGN1D(64, gamma_init_mode="uniform")
    inv(torch.randn(8, 64, 50))
    assert inv.last_clamp_rate > 0.5 and inv.last_max_clamped_ratio > 1.0


def test_untied_and_mirror_accept_init_mode():
    assert ExactInverseBIGN1D(8).gamma[0, 1] < 0.1
    assert MirrorBIGN1D(8, gamma_init_mode="uniform").gamma[0, 1] > 0.65
