"""Analytic FLOPs (paper targets), parameter counting with shared tensors, latency."""
import pytest
from torch import nn

from atandn.evaluation.complexity import count_parameters, flops_per_window, measure_latency

# (C_in, window, conv_filters, without decoder, with exact-solve decoder)
CASES = {
    "M1": (6, 300, (64, 192), 9.52e7, 1.33e8),
    "M2": (11, 300, (64, 128), 5.715e7, 9.28e7),
    "OPS": (1, 400, (64, 256), 2.00e8, 2.54e8),
}


@pytest.mark.parametrize("name", list(CASES))
def test_flops_match_targets(name):
    c, t, f, no_dec, with_dec = CASES[name]
    assert flops_per_window(c, t, f, include_decoder=False)["total"] == pytest.approx(no_dec, rel=3e-3)
    assert flops_per_window(c, t, f, include_decoder=True, inverse_method="solve")["total"] == pytest.approx(with_dec, rel=3e-3)


def test_flops_components_and_neumann():
    d = flops_per_window(11, 300, (64, 128), include_decoder=False)
    assert d["total"] == pytest.approx(d["encoder_conv"] + d["bign"] + d["mha"] + d["se"] + d["classifier"])
    # hand check of the first encoder conv: L1 = (300 + 4 - 5)//2 + 1 = 150
    assert d["encoder_conv"] == 2 * 150 * 11 * 64 * 5 + 2 * 75 * 64 * 128 * 5
    n = flops_per_window(11, 300, (64, 128), inverse_method="neumann", neumann_steps=4)
    s = flops_per_window(11, 300, (64, 128), inverse_method="solve")
    assert n["inverse_bign"] < s["inverse_bign"]
    with pytest.raises(ValueError):
        flops_per_window(11, 300, (64, 128), inverse_method="bogus")


def test_count_parameters_shared_tensors_once():
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
            self.b = nn.Module()
            self.b.lin = self.a            # same module registered twice
            self.c = nn.Linear(2, 2)

    m = Tied()
    assert count_parameters(m, unique=True) == (16 + 4) + (4 + 2)
    assert count_parameters(m, unique=False) == 2 * (16 + 4) + (4 + 2)


def test_measure_latency_cpu():
    r = measure_latency(nn.Conv1d(2, 2, 3), (1, 2, 16), device="cpu", repeats=5, warmup=1)
    assert r["mean_ms"] > 0 and r["p95_ms"] >= r["median_ms"] and r["device"] == "cpu"


def test_real_model_parameter_count_if_available():
    models = pytest.importorskip("atandn.models")
    m = models.ATANDN(in_channels=6, window_size=300, conv_filters=(64, 192), num_heads=2)
    assert count_parameters(m, unique=True) <= count_parameters(m, unique=False)


@pytest.mark.parametrize("c,t,f,heads", [(6, 300, (64, 192), 2), (11, 300, (64, 128), 4)])
def test_analytic_flops_match_flop_counter(c, t, f, heads):
    """Compare the analytic terms with ``torch.utils.flop_counter.FlopCounterMode`` on an instantiated model.

    The counter covers convolutions (``aten.convolution``) and matrix products (``mm``/``bmm``/``addmm``:
    BIGN matmul, linear layers, SE, attention in/out projections). It does NOT count: the attention
    QK^T / AV products (fused SDPA kernel on CPU), ``torch.linalg.solve`` of the inverse layers, elementwise
    ops, norms, softmax and interpolation. The model is run with grad enabled, otherwise
    ``nn.MultiheadAttention`` takes a fused fast path in eval mode that the counter does not see.
    The model has a plain (linear) decoder, i.e. ``inverse="none"``.
    """
    flop_counter = pytest.importorskip("torch.utils.flop_counter")
    import torch
    from atandn.models import ATANDN

    model = ATANDN(in_channels=c, window_size=t, conv_filters=f, num_heads=heads, inverse="none").eval()
    with flop_counter.FlopCounterMode(display=False) as fc:
        model(torch.randn(1, c, t))
    counts = {str(k): v for k, v in fc.get_flop_counts()["Global"].items()}
    d = flops_per_window(c, t, f, include_decoder=True, inverse_method=None)

    # convolutions: encoder + decoder transposed convs, exact
    assert counts["aten.convolution"] == d["encoder_conv"] + d["decoder_conv"]

    # matrix products = BIGN + SE + classifier + attention projections (8 L C^2 per MHA block: in 6LC^2 + out 2LC^2)
    l1 = (t + 4 - 5) // 2 + 1
    l2 = (l1 + 4 - 5) // 2 + 1

    proj = 8 * l1 * f[0] ** 2 + 8 * l2 * f[1] ** 2 + 8 * l2 * f[1] ** 2
    matmul_measured = sum(v for k, v in counts.items() if k in ("aten.mm", "aten.bmm", "aten.addmm"))
    matmul_analytic = d["bign"] + d["se"] + d["classifier"] + proj
    assert matmul_measured == pytest.approx(matmul_analytic, rel=1e-3)

    # what the counter cannot see is exactly the attention score/value term 4 L^2 C of the analytic MHA entry
    qk_av = 4 * l1 ** 2 * f[0] + 2 * 4 * l2 ** 2 * f[1]
    assert d["mha"] == pytest.approx(proj + qk_av)


def test_parameter_count_with_tied_layers_matches_instantiated_model():
    """``count_parameters`` counts the tied BIGN tensors once; the duplicate listing counts them twice."""
    from atandn.models import ATANDN

    c, f = 6, (16, 32)
    m = ATANDN(in_channels=c, window_size=64, conv_filters=f, num_heads=2)
    unique = count_parameters(m, unique=True)
    assert unique == sum(p.numel() for p in m.parameters())
    dup = count_parameters(m, unique=False)
    # tied tensors: beta_raw (C1) and gamma_raw (C1 x C1) of the first encoder BIGN, shared with decoder layer 0
    assert dup - unique == f[0] + f[0] ** 2
    untied = ATANDN(in_channels=c, window_size=64, conv_filters=f, num_heads=2, inverse="mirror")
    assert count_parameters(untied, unique=True) == count_parameters(untied, unique=False)
