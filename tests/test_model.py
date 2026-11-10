"""Tests for ATANDN and the normalization factory."""
import pytest
import torch

from atandn.layers.bign import BIGN1D, ExactInverseBIGN1D
from atandn.layers.norms import NORM_KINDS, build_norm
from atandn.models.atandn import ATANDN

CONFIGS = [(6, 300, (64, 192)), (11, 300, (64, 128)), (1, 400, (64, 256))]


@pytest.mark.parametrize("c,t,f", CONFIGS)
def test_output_shapes(c, t, f):
    m = ATANDN(c, t, f).eval()
    recon, logits = m(torch.randn(2, c, t))
    assert recon.shape == (2, c, t) and logits.shape == (2,)


@pytest.mark.parametrize("kind", NORM_KINDS)
def test_build_norm_shape(kind):
    n = build_norm(kind, 16)
    x = torch.randn(3, 16, 20)
    assert n(x).shape == x.shape


def test_layernorm_axis_and_instancenorm_axis():
    x = torch.randn(2, 16, 30) * 3 + 1
    ln = build_norm("layernorm", 16)(x)
    assert torch.allclose(ln.mean(dim=1), torch.zeros(2, 30), atol=1e-5)
    inn = build_norm("instancenorm", 16)(x)
    assert torch.allclose(inn.mean(dim=2), torch.zeros(2, 16), atol=1e-5)


def test_unknown_norm_raises():
    with pytest.raises(ValueError):
        build_norm("nope", 4)


@pytest.mark.parametrize("norm", NORM_KINDS)
@pytest.mark.parametrize("inverse,method", [("exact", "solve"), ("exact", "neumann"),
                                            ("mirror", "solve"), ("none", "solve")])
def test_all_options_forward_backward(norm, inverse, method):
    torch.manual_seed(0)
    m = ATANDN(3, 64, (16, 32), num_heads=4, norm=norm, inverse=inverse, inverse_method=method)
    x = torch.randn(4, 3, 64)
    recon, logits = m(x)
    (recon.pow(2).mean() + logits.mean()).backward()
    assert recon.shape == x.shape and torch.isfinite(recon).all()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("flags", [{"use_attention": False}, {"use_se": False},
                                   {"use_attention": False, "use_se": False}])
def test_ablation_flags(flags):
    m = ATANDN(3, 64, (16, 32), **flags)
    recon, logits = m(torch.randn(2, 3, 64))
    assert recon.shape == (2, 3, 64) and logits.shape == (2,)


def test_no_reconstruction_returns_none():
    m = ATANDN(3, 64, (16, 32), use_reconstruction=False)
    recon, logits = m(torch.randn(2, 3, 64))
    assert recon is None and logits.shape == (2,) and m.decoder is None


def test_tied_parameters_counted_once():
    tied = ATANDN(3, 64, (16, 32), norm="bign", inverse="exact")
    assert len(tied.bign_layers()) == 2
    dec_inv = [m for m in tied.decoder if isinstance(m, ExactInverseBIGN1D)]
    assert dec_inv[0].tied is tied.bign_layers()[0] and dec_inv[1].tied is None
    ids = [id(p) for p in tied.parameters()]
    assert len(ids) == len(set(ids))
    # A mirror decoder owns its first-layer parameters (C1 = 16: 16 + 16*16), the tied one does not.
    mirror = ATANDN(3, 64, (16, 32), norm="bign", inverse="mirror")
    n_tied = sum(p.numel() for p in tied.parameters())
    n_mirror = sum(p.numel() for p in mirror.parameters())
    assert n_mirror - n_tied == 16 + 16 * 16


def test_untied_when_norm_not_bign():
    m = ATANDN(3, 64, (16, 32), norm="layernorm", inverse="exact")
    assert m.bign_layers() == []
    assert all(x.tied is None for x in m.decoder if isinstance(x, ExactInverseBIGN1D))


def test_last_decoder_block_is_linear():
    m = ATANDN(3, 64, (16, 32), inverse="exact").eval()
    assert isinstance(m.decoder[-1], ExactInverseBIGN1D)
    recon, _ = m(torch.randn(8, 3, 64))
    assert (recon < 0).any()


def test_bign_layers_type():
    m = ATANDN(3, 64, (16, 32))
    assert all(isinstance(b, BIGN1D) for b in m.bign_layers())
