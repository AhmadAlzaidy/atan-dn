"""BIGN / attention / inverse reports (paper claims A13, A26, A35) on freshly built models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from atandn.config import MissionConfig
from atandn.evaluation.attention_report import attention_norms
from atandn.evaluation.bign_report import bign_stats
from atandn.evaluation.inverse_report import inverse_report
from atandn.layers.bign import BIGN1D
from atandn.models.atandn import ATANDN
from atandn.training import save_checkpoint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _model(pe_mode: str = "attention", seed: int = 0) -> ATANDN:
    torch.manual_seed(seed)
    return ATANDN(in_channels=3, window_size=40, conv_filters=(8, 16), num_heads=2, pe_mode=pe_mode).eval()


def _randomize(model: ATANDN) -> None:
    """Make the weights non-trivial so that the bounds are tested away from the init."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))


def test_bign_stats_keys_fresh_model_no_data():
    rows = bign_stats(_model())
    assert len(rows) == 2   # both encoder blocks
    keys = {"layer", "C", "gamma_ii_min", "gamma_ii_median", "gamma_ii_max", "inv_sqrt_max", "R", "beta_min",
            "offdiag_max", "offdiag_mean", "rowsum_max"}
    assert keys <= set(rows[0]) and "empirical_max_abs_y" not in rows[0]


def test_bign_bound_ge_observed_on_random_inputs():
    m = _model()
    _randomize(m)
    X = 50.0 * torch.randn(32, 3, 40)   # large amplitudes: the output saturates
    r = bign_stats(m, X)[0]
    assert r["bound_holds"] and r["empirical_max_abs_y"] <= r["inv_sqrt_max"] * (1 + 1e-5)
    assert r["n_windows"] == 32


def test_bign_stats_standalone_layer_value():
    layer = BIGN1D(3)
    holder = torch.nn.Sequential(layer)
    row = bign_stats(holder)[0]
    assert row["gamma_ii_min"] == pytest.approx(float(torch.diagonal(layer.gamma).detach().min()), rel=1e-6)
    assert row["R"] == pytest.approx(float((1 / torch.diagonal(layer.gamma.detach())).sum().sqrt()), rel=1e-6)


@pytest.mark.parametrize("pe_mode", ["input", "attention"])
def test_attention_lambda_bounds_observed_logits(pe_mode):
    m = _model(pe_mode, seed=3)
    _randomize(m)
    rows = attention_norms(m)
    assert len(rows) == 3
    assert rows[2]["Lambda"] is None   # bottleneck: no BIGN directly before it
    bign, block = m.encoder_blocks[0][2], m.encoder_blocks[0][3]
    c, h = block.mha.embed_dim, block.mha.num_heads
    dk = c // h
    y = bign(60.0 * torch.randn(8, 8, 20))          # BIGN output of a saturating input (C=8, T=20)
    inp = block.pos_enc(y) if block.pos_enc is not None else y
    w, b = block.mha.in_proj_weight, block.mha.in_proj_bias
    q = (inp.transpose(1, 2) @ w[:c].T + b[:c]).reshape(8, 20, h, dk)
    k = (inp.transpose(1, 2) @ w[c:2 * c].T + b[c:2 * c]).reshape(8, 20, h, dk)
    logits = torch.einsum("bihd,bjhd->bhij", q, k).abs() / dk ** 0.5
    lam = rows[0]["Lambda"] if pe_mode == "input" else rows[0]["Lambda_with_pe"]
    assert lam is not None and logits.max().item() <= lam * (1 + 1e-5)
    assert set(rows[0]) >= {"layer", "C", "H", "d_k", "spec_WQ", "spec_WK", "prod", "bQ", "bK", "R", "Lambda"}
    assert rows[0]["prod"] >= 0 and rows[0]["Lambda"] > 0


def test_inverse_report_bounds_and_ranges():
    m = _model()
    X = torch.randn(10, 3, 40)
    rep = inverse_report(m, X, max_pairs=200, batch_size=4)
    assert len(rep["layers"]) == 2 and rep["bound_holds"] is True
    assert rep["kappa_bound"] == pytest.approx(2.0 / 1e-3)
    for layer in rep["layers"]:
        assert 0.0 <= layer["clamp_rate"] <= 1.0
        assert 1.0 <= layer["kappa_p50"] <= layer["kappa_p99"] <= layer["kappa_max"] <= 2.0 / layer["eps"]
        assert layer["solve_ms_per_window"] > 0


def test_inverse_report_large_inputs_still_below_bound():
    m = _model()
    _randomize(m)
    rep = inverse_report(m, 100.0 * torch.randn(6, 3, 40), max_pairs=300)
    assert rep["bound_holds"]
    assert all(0.0 <= l["clamp_rate"] <= 1.0 for l in rep["layers"])


def test_report_scripts_without_data(tmp_path):
    import attention_report as ar
    import bign_report as br

    cfg = MissionConfig()
    cfg.model.in_channels, cfg.model.window_size, cfg.model.conv_filters, cfg.model.num_heads = 3, 40, (8, 16), 2
    ckpt = save_checkpoint(tmp_path / "m.pt", _model(), cfg, {})
    yml = tmp_path / "c.yaml"
    yml.write_text("mission: m1\n", encoding="utf-8")
    assert br.main(["--checkpoint", str(ckpt), "--config", str(yml), "--out", str(tmp_path / "b.json")]) == 0
    assert ar.main(["--checkpoint", str(ckpt), "--config", str(yml), "--out", str(tmp_path / "a.json")]) == 0
    assert json.loads((tmp_path / "b.json").read_text())[0]["C"] == 8
    assert len(json.loads((tmp_path / "a.json").read_text())) == 3


def test_model_reports_markdown():
    import model_reports as mr

    m = _model()
    X = torch.randn(4, 3, 40)
    md = mr.markdown_summary(bign_stats(m, X), attention_norms(m), inverse_report(m, X, max_pairs=50), len(X))
    assert "## BIGN" in md and "## Attention" in md and "kappa bound" in md
