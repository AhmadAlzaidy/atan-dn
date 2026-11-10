"""Smoke tests of Trainer, EMA, checkpoint, training analysis and the Optuna search on tiny synthetic data."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("atandn.losses")
optuna = pytest.importorskip("optuna")

from atandn.analysis import derive_mask_parameters
from atandn.config import mission_defaults
from atandn.data import WindowDataset, make_test_windows
from atandn.inference import score_windows
from atandn.training import EMA, Trainer, build_loaders, build_objective, run_tpe_search, suggest_params
from atandn.utils import set_seed


class TinyNet(nn.Module):
    """Stand-in for ATANDN: forward returns (recon, logits (B,))."""

    def __init__(self, c: int = 2, t: int = 32, **_):
        super().__init__()
        self.enc = nn.Conv1d(c, 4, 3, padding=1)
        self.dec = nn.Conv1d(4, c, 3, padding=1)
        self.bn = nn.BatchNorm1d(4)
        self.clf = nn.Linear(4 * t, 1)

    def forward(self, x):
        h = torch.relu(self.bn(self.enc(x)))
        return self.dec(h), self.clf(h.flatten(1)).squeeze(-1)


def _stream(n=1500, c=2, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, c))
    y = np.zeros(n, dtype=int)
    for s in range(100, n - 60, 200):
        v[s:s + 40] += 4.0
        y[s:s + 40] = 1
    return v, y


def _tiny_cfg(w=32):
    cfg = mission_defaults("m2")
    cfg.model.in_channels, cfg.model.window_size = 2, w
    cfg.model.conv_filters, cfg.model.num_heads = (8, 8), 2
    cfg.train.max_epochs, cfg.train.patience, cfg.train.batch_size = 4, 3, 16
    cfg.train.warmup_epochs, cfg.train.overlap = 1, 0.5
    cfg.eval.sensitive_threshold = 0.5
    return cfg


def test_ema_apply_restore():
    m = nn.Linear(2, 2)
    ema = EMA(m, 0.5)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update()
    with ema.average_weights():
        assert torch.allclose(m.weight, w0 + 0.5)
    assert torch.allclose(m.weight, w0 + 1.0)


def test_trainer_fit_tiny_model_selects_on_validation_tail():
    set_seed(0)
    v, y = _stream()
    cfg = _tiny_cfg()
    fit = WindowDataset(v[:1000], y[:1000], 32, 8)
    val = WindowDataset(v[1000:], y[1000:], 32, 8)
    fl, vl = build_loaders(fit, val, cfg.train)
    seen = []
    tr = Trainer(TinyNet(), cfg.train, device="cpu")
    out = tr.fit(fl, vl, epoch_callback=lambda e, s, h: seen.append(e))
    assert out["epochs_run"] == len(seen) <= cfg.train.max_epochs
    assert out["has_val_positives"] and 0.0 <= out["best_score"] <= 1.0
    assert all(np.isfinite(h["train_loss"]) for h in out["history"])


def test_trainer_with_weighted_sampling_and_no_val():
    set_seed(0)
    v, y = _stream()
    cfg = _tiny_cfg()
    cfg.train.use_weighted_sampling = True
    fl, vl = build_loaders(WindowDataset(v, y, 32, 8, augment=True), None, cfg.train)
    assert vl is None
    out = Trainer(TinyNet(), cfg.train, device="cpu").fit(fl)
    assert out["epochs_run"] >= 1


def test_derive_mask_parameters_uses_train_only():
    set_seed(0)
    v, y = _stream()
    cfg = _tiny_cfg()
    ts = pd.date_range("2020-01-01", periods=len(v), freq="30s")
    ev = [(ts[300], ts[320])]
    params = derive_mask_parameters(TinyNet().eval(), v, y, ts, ev, cfg, device="cpu")
    for k in ("mask_buffer_after_minutes", "mask_buffer_before_minutes", "noise_cluster_median", "min_consecutive_windows", "normalizer"):
        assert k in params
    assert params["fitted_on"] == "training split only" and params["n_known_events"] == 1


def test_suggest_params_and_tpe_smoke():
    set_seed(0)
    v, y = _stream()
    cfg = _tiny_cfg()
    fixed = {"window_size": 32, "conv_f1": 8, "conv_f2": 8, "num_heads": 2, "batch_size": 16, "overlap": 0.5}
    factory = lambda mc: TinyNet(mc.in_channels, mc.window_size)
    obj = build_objective(cfg, v, y, model_factory=factory, device="cpu", fixed=fixed, max_epochs=2)
    study = run_tpe_search(obj, n_trials=3, seed=0, n_startup_trials=2)
    assert len(study.trials) == 3
    assert any(t.state.name == "COMPLETE" for t in study.trials)
    p = suggest_params(optuna.trial.FixedTrial({"window_size": 300, "overlap": 0.8, "pos_weight": 5.0, "lr": 1e-3,
        "recon_weight": 0.2, "dropout": 0.2, "batch_size": 32, "focal_gamma": 2.0, "weight_decay": 1e-4,
        "recon_score_weight": 0.1, "num_heads": 2, "conv_f1": 64, "conv_f2": 128,
        "use_weighted_sampling": True, "use_augmentation": False}))
    assert p["window_size"] == 300 and p["use_weighted_sampling"] is True


def test_real_model_train_score_and_checkpoint_if_available(tmp_path):
    models = pytest.importorskip("atandn.models")
    from atandn.training import load_checkpoint, save_checkpoint
    set_seed(0)
    v, y = _stream(n=900)
    cfg = _tiny_cfg(w=32)
    cfg.train.max_epochs = 2
    model = models.ATANDN(**{**cfg.model.__dict__, "conv_filters": tuple(cfg.model.conv_filters)})
    fl, vl = build_loaders(WindowDataset(v[:600], y[:600], 32, 8), WindowDataset(v[620:], y[620:], 32, 8), cfg.train)
    Trainer(model, cfg.train, device="cpu").fit(fl, vl)
    win = make_test_windows(v, y, np.arange(len(v)), 32, 10)
    sc = score_windows(model, win["X"][:20], batch_size=8, device="cpu")
    assert sc["prob"].shape == (20,) and np.isfinite(sc["recon_err"]).all()
    p = save_checkpoint(tmp_path / "m.pt", model, cfg, {"threshold": np.float32(0.3)})
    m2, cfg2, extra = load_checkpoint(p, "cpu")
    assert extra["threshold"] == pytest.approx(0.3) and cfg2.model.window_size == 32
    sc2 = score_windows(m2, win["X"][:20], batch_size=8, device="cpu")
    np.testing.assert_allclose(sc["prob"], sc2["prob"], atol=1e-5)
