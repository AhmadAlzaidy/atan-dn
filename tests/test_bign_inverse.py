"""BIGN inverse diagnostics, lazy clamp statistics, init-mode checkpoint round trip, tied-tensor checks,
OPS-SAT tie determinism and the Trainer's NaN-validation warning."""
import logging

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from atandn.config import MissionConfig, ModelConfig, TrainConfig
from atandn.data import load_opssat
from atandn.layers.bign import BIGN1D, ExactInverseBIGN1D
from atandn.models.atandn import ATANDN
from atandn.training.checkpoint import load_checkpoint, save_checkpoint
from atandn.training.trainer import Trainer


def _cfg(**kw) -> MissionConfig:
    cfg = MissionConfig()
    cfg.model = ModelConfig(in_channels=4, window_size=32, conv_filters=(8, 8), num_heads=2, **kw)
    return cfg


def _build(cfg: MissionConfig) -> ATANDN:
    return ATANDN(**{**cfg.model.__dict__, "conv_filters": tuple(cfg.model.conv_filters)})


@pytest.mark.parametrize("kw", [dict(gamma_init_mode="diag_dominant", gamma_diag=0.3, gamma_off=0.02),
                                dict(gamma_init_mode="uniform")])
def test_checkpoint_round_trip_records_init(tmp_path, kw):
    torch.manual_seed(0)
    cfg = _cfg(**kw)
    model = _build(cfg).eval()
    path = save_checkpoint(tmp_path / "m.pt", model, cfg)
    loaded, cfg2, _ = load_checkpoint(path, device="cpu")
    for k, v in kw.items():
        assert getattr(cfg2.model, k) == v
    x = torch.randn(3, 4, 32)
    with torch.no_grad():
        r1, l1 = model(x)
        r2, l2 = loaded(x)
    assert torch.allclose(l1, l2, atol=1e-6) and torch.allclose(r1, r2, atol=1e-6)
    # the config reaches the layers: a fresh model built from the stored config has the requested init
    g = _build(cfg2).bign_layers()[0].gamma.detach()
    if kw["gamma_init_mode"] == "uniform":
        assert (g > 0.5).all()
    else:
        assert float(g[0, 0]) == pytest.approx(0.3, abs=0.05) and float(g[0, 1]) < 0.1


def test_config_rejects_bad_init_mode():
    cfg = MissionConfig()
    cfg.model.gamma_init_mode = "nope"
    with pytest.raises(ValueError):
        cfg.validate()


def test_tied_keys_that_disagree_raise(tmp_path):
    torch.manual_seed(0)
    cfg = _cfg()
    model = _build(cfg).eval()
    path = save_checkpoint(tmp_path / "m.pt", model, cfg)
    blob = torch.load(path, weights_only=True)
    dec_keys = [k for k in blob["state_dict"] if k.startswith("decoder.") and ".tied." in k and k.endswith("gamma_raw")]
    assert dec_keys, "tied decoder key not found in the state_dict"
    blob["state_dict"][dec_keys[0]] = blob["state_dict"][dec_keys[0]] + 1.0
    bad = tmp_path / "bad.pt"
    torch.save(blob, bad)
    with pytest.raises(ValueError, match="tied tensors"):
        load_checkpoint(bad, device="cpu")
    load_checkpoint(path, device="cpu")  # the untouched checkpoint still loads


def test_clamp_statistics_are_lazy_properties():
    assert isinstance(ExactInverseBIGN1D.last_clamp_rate, property)
    assert isinstance(ExactInverseBIGN1D.last_max_clamped_ratio, property)
    fwd = BIGN1D(3)
    inv = ExactInverseBIGN1D(tied_to=fwd)
    assert inv.last_clamp_rate == 0.0 and inv.last_clamp_rate_tensor is None
    inv(fwd(torch.randn(2, 3, 10)))
    t = inv.last_clamp_rate_tensor                      # stored on the device, not yet converted to a Python float
    assert isinstance(t, torch.Tensor) and t.dim() == 0 and not t.requires_grad
    assert isinstance(inv.last_clamp_rate, float) and inv.last_clamp_rate == float(t)
    assert isinstance(inv.last_max_clamped_ratio, float)


def test_round_trip_error_diagnostic():
    torch.manual_seed(0)
    c = 8
    bign = BIGN1D(c)
    ok = ExactInverseBIGN1D.round_trip_error(torch.randn(4, c, 100), bign)
    assert ok["rel_error"] < 1e-4 and ok["clamp_rate"] == 0.0 and ok["max_clamped_ratio"] < 1.0
    big = ExactInverseBIGN1D.round_trip_error(torch.randn(4, c, 100) * 10, bign)   # saturated inputs
    assert big["clamp_rate"] > 0.0 and big["max_clamped_ratio"] > 1.0 and big["rel_error"] > 0.1
    # Neumann with K = 4 is an approximation even at unit scale
    ne = ExactInverseBIGN1D.round_trip_error(torch.randn(4, c, 100), bign, method="neumann", neumann_steps=4)
    assert ne["rel_error"] > 1e-3
    assert ne["rel_error"] > 100 * ok["rel_error"]


def _write_segments(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_opssat_tie_order_deterministic_and_logged(tmp_path, caplog):
    rows = []
    for ch in ("C", "A", "B"):
        for i in range(4):
            rows.append({"channel": ch, "timestamp": f"2022-06-01T00:00:0{i}.000Z", "value": float(i * 10 + ord(ch) - 65),
                         "anomaly": int(ch == "B" and i == 2), "segment": 1, "train": 1})
    _write_segments(tmp_path / "a.csv", rows)
    _write_segments(tmp_path / "b.csv", rows[::-1])
    _write_segments(tmp_path / "c.csv", list(np.random.default_rng(0).permutation(np.array(rows, dtype=object))))
    with caplog.at_level(logging.INFO, logger="atandn.data.opssat"):
        a = load_opssat(tmp_path / "a.csv")
    assert any("8 duplicate timestamps" in r.getMessage() for r in caplog.records)
    for name in ("b.csv", "c.csv"):
        m = load_opssat(tmp_path / name)
        assert np.array_equal(m.values, a.values) and np.array_equal(m.labels, a.labels)
    # within a timestamp: channel A, B, C
    assert a.values[:3, 0].tolist() == [0.0, 1.0, 2.0]


def test_trainer_warns_when_validation_score_is_nan(caplog, monkeypatch):
    torch.manual_seed(0)
    model = ATANDN(in_channels=4, window_size=32, conv_filters=(8, 8), num_heads=2)
    x, y = torch.randn(8, 4, 32), (torch.rand(8) > 0.5).float()
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    tr = Trainer(model, TrainConfig(max_epochs=2, warmup_epochs=1, patience=5), device="cpu")
    monkeypatch.setattr(tr, "_validate", lambda loader: (float("nan"), True))
    with caplog.at_level(logging.WARNING, logger="atandn.training.trainer"):
        res = tr.fit(loader, loader)
    assert res["best_epoch"] == -1
    assert any("LAST EMA weights" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------------------------------------
# documented orders of magnitude (docs/BIGN_INVERSE.md), measured with scripts/inverse_diagnostics.py
# ---------------------------------------------------------------------------------------------------------
def _measure(*a, **kw):
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from inverse_diagnostics import measure
    return measure(*a, **kw)


@pytest.mark.parametrize("c", [1, 6, 64])
def test_documented_unit_scale_error_is_float_precision(c):
    r = _measure(c, 1.0, "diag_dominant", "solve", 4, seeds=(0, 1))
    assert r["rel_error"] < 1e-5 and r["clamp_rate"] == 0.0


def test_documented_saturation_and_uniform_init_numbers():
    r10 = _measure(6, 10.0, "diag_dominant", "solve", 4, seeds=(0, 1))
    assert 0.1 < r10["rel_error"] < 0.7 and 0.1 < r10["clamp_rate"] < 0.4          # doc: 0.38 / 0.22
    assert _measure(1, 10.0, "diag_dominant", "solve", 4, seeds=(0, 1))["rel_error"] < 0.05   # doc: 2e-4
    assert _measure(6, 1000.0, "diag_dominant", "solve", 4, seeds=(0, 1))["rel_error"] > 0.9   # doc: 0.99
    u = _measure(6, 1.0, "uniform", "solve", 4, seeds=(0, 1))
    assert 0.3 < u["rel_error"] < 0.9 and u["clamp_rate"] > 0.1                    # doc: 0.60
    assert _measure(64, 1.0, "uniform", "solve", 4, seeds=(0, 1))["rel_error"] > 0.6   # doc: 0.87


def test_documented_neumann_numbers():
    k4 = _measure(6, 1.0, "diag_dominant", "neumann", 4, seeds=(0, 1))
    k50 = _measure(6, 1.0, "diag_dominant", "neumann", 50, seeds=(0, 1))
    assert 0.05 < k4["rel_error"] < 0.3 and k50["rel_error"] < 0.01                # doc: 0.11 / 3e-4
    assert _measure(6, 10.0, "diag_dominant", "neumann", 50, seeds=(0, 1))["rel_error"] > 0.2   # doc: 0.52
