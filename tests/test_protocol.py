"""Evaluation-protocol tests: event-aware validation split, grid threshold, streaming scoring, frozen pipeline."""
from __future__ import annotations

import logging
import tracemalloc

import numpy as np
import pytest
import torch
from torch import nn

from atandn.config import MissionConfig, EvalConfig, mission_defaults
from atandn.data import (MissionData, count_events, event_aware_validation_split, iter_window_batches,
                         make_test_windows, window_starts)
from atandn.evaluation.pipeline import (ProtocolData, effective_alpha, evaluate_mission, fit_frozen_protocol,
                                        prepare_protocol_data)
from atandn.inference import average_precision, grid_f1_threshold, score_stream, score_windows, select_threshold
from atandn.training import checkpoint_config_conflicts


class Toy(nn.Module):
    """Tiny stand-in with the ATANDN interface: forward -> (recon | None, logits)."""

    def __init__(self, c: int = 2, t: int = 32, recon: bool = True):
        super().__init__()
        self.enc = nn.Conv1d(c, 4, 3, padding=1)
        self.decoder = nn.Conv1d(4, c, 3, padding=1) if recon else None
        self.clf = nn.Linear(4, 1)

    def forward(self, x):
        h = torch.relu(self.enc(x))
        logits = self.clf(h.mean(dim=2)).squeeze(-1)
        return (None if self.decoder is None else self.decoder(h)), logits


def _full_events(y: np.ndarray, s: int, e: int) -> int:
    """Brute force: runs of the whole label array that lie entirely inside ``[s, e)``."""
    d = np.diff(np.r_[0, (np.asarray(y) > 0).astype(int), 0])
    return sum(1 for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if a >= s and b <= e)


def _labels(n: int, events: list[tuple[int, int]]) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    for a, b in events:
        y[a:b] = 1
    return y


# ---------------------------------------------------------------------------------------------------------
# validation split
# ---------------------------------------------------------------------------------------------------------
def test_count_events():
    assert count_events(np.array([0, 1, 1, 0, 1, 0, 0, 1])) == 3
    assert count_events(np.array([1, 1, 0])) == 1 and count_events(np.zeros(5)) == 0


def test_validation_tail_kept_when_it_has_enough_events():
    y = _labels(1000, [(810, 820), (850, 860), (900, 910), (950, 960)])
    vs = event_aware_validation_split(y, 0.2, gap=31, min_events=3)
    assert (vs.val.start, vs.val.stop) == (800, 1000) and vs.fit == slice(0, 800 - 31)
    assert vs.report["tail_moved"] is False and vs.report["n_val_events"] == 4 and vs.report["val_has_positives"]


def test_validation_falls_back_to_earlier_block_when_tail_has_no_positives(caplog):
    """Mission-2-like: the last 20% of the training split has no positive; the events are earlier."""
    y = _labels(1000, [(300, 310), (350, 360), (400, 410), (700, 710)])
    with caplog.at_level(logging.WARNING, logger="atandn.data.splits"):
        vs = event_aware_validation_split(y, 0.2, gap=31, min_events=3)
    assert vs.report["tail_moved"] is True and vs.report["val_has_positives"] is True
    assert count_events(y[vs.val]) >= 3
    # latest qualifying block: moving it one sample later would lose a FULL event
    s = vs.val.start
    assert _full_events(y, s, s + 200) >= 3 and (s + 201 > 1000 or _full_events(y, s + 1, s + 201) < 3)
    assert vs.report["val_events"] == 3 and vs.report["min_events_used"] == 3
    assert vs.fit.stop == vs.val.start - 31                      # only data BEFORE the block (minus the gap)
    assert any("latest block" in r.message for r in caplog.records)


def test_validation_no_qualifying_block_keeps_tail_and_flags_no_positives(caplog):
    y = _labels(1000, [])
    with caplog.at_level(logging.WARNING, logger="atandn.data.splits"):
        vs = event_aware_validation_split(y, 0.2, gap=31, min_events=3)
    assert (vs.val.start, vs.val.stop) == (800, 1000) and vs.report["tail_moved"] is False
    assert vs.report["val_has_positives"] is False and vs.report["min_events_satisfied"] is False
    assert any("NO positive" in r.message for r in caplog.records)


def test_validation_split_too_short():
    with pytest.raises(ValueError):
        event_aware_validation_split(np.zeros(50), 0.2, gap=45)


# ---------------------------------------------------------------------------------------------------------
# threshold
# ---------------------------------------------------------------------------------------------------------
def test_grid_threshold_is_on_grid_and_f1_optimal():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    s = np.clip(0.3 + 0.4 * y + 0.15 * rng.standard_normal(400), 0, 1)
    thr = grid_f1_threshold(s, y)
    grid = np.arange(0.05, 0.95 + 1e-9, 0.005)
    assert np.isclose(grid, thr).any()

    def f1(t):
        p = s > t
        tp = (p & (y == 1)).sum()
        return 2 * tp / (2 * tp + (p & (y == 0)).sum() + ((~p) & (y == 1)).sum())
    assert f1(thr) == pytest.approx(max(f1(t) for t in grid))
    assert select_threshold(s, y, "grid") == thr
    assert 0.0 <= select_threshold(s, y, "exact") <= 1.0
    with pytest.raises(ValueError):
        grid_f1_threshold(s, np.zeros(400))
    with pytest.raises(ValueError):
        select_threshold(s, y, "nope")


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0])) == pytest.approx(1.0)
    assert average_precision(np.array([0.9, 0.1, 0.8]), np.array([1, 1, 0])) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert np.isnan(average_precision(np.array([0.1, 0.2]), np.zeros(2)))


# ---------------------------------------------------------------------------------------------------------
# streaming scoring
# ---------------------------------------------------------------------------------------------------------
def test_stream_scores_equal_materialized_scores():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    v = rng.standard_normal((700, 2))
    model = Toy()
    win = make_test_windows(v, np.zeros(700), np.arange(700), 32, 10)
    ref = score_windows(model, win["X"], batch_size=512, device="cpu")
    for bs in (1, 7, 512):
        got = score_stream(model, v, 32, 10, batch_size=bs, device="cpu")
        np.testing.assert_allclose(got["prob"], ref["prob"], atol=1e-6)
        np.testing.assert_allclose(got["recon_err"], ref["recon_err"], atol=1e-6)
        np.testing.assert_array_equal(got["end_index"], win["end_index"])


def test_streaming_memory_is_bounded_on_a_long_stream():
    n, w, stride, bs = 300_000, 64, 10, 128
    v = np.random.default_rng(0).standard_normal((n, 2)).astype(np.float32)
    starts = window_starts(n, w, stride)
    full_bytes = len(starts) * 2 * w * 4                           # what materializing every window would cost
    sizes = [len(b) for b in iter_window_batches(v[:20_000], window_starts(20_000, w, stride), w, bs)]
    assert max(sizes) == bs                                        # never more than one chunk of windows
    model = Toy(t=w)
    tracemalloc.start()
    out = score_stream(model, v, w, stride, batch_size=bs, device="cpu")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(out["prob"]) == len(starts)
    assert full_bytes > 10e6 and peak < full_bytes / 8, (peak, full_bytes)


# ---------------------------------------------------------------------------------------------------------
# frozen pipeline
# ---------------------------------------------------------------------------------------------------------
def _stream(n: int, events: list[tuple[int, int]], seed: int = 0) -> MissionData:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    y = _labels(n, events)
    for a, b in events:
        x[a:b] += 3.0
    return MissionData(x, y, np.arange(n).astype("datetime64[m]"), ["a", "b"], {})


def _cfg() -> MissionConfig:
    cfg = mission_defaults("m1")
    cfg.model.in_channels, cfg.model.window_size = 2, 32
    cfg.train.val_fraction = 0.25
    cfg.eval.use_known_event_mask = False
    return cfg


def test_threshold_normalizer_and_mask_independent_of_test_split():
    torch.manual_seed(0)
    train = _stream(1200, [(100, 130), (400, 430), (900, 930), (1050, 1080)])
    cfg, model = _cfg(), Toy()
    t1 = _stream(800, [(100, 120), (500, 530)], 1)
    t2 = MissionData(t1.values, np.zeros_like(t1.labels), t1.timestamps, t1.channel_names, {})
    reps = [evaluate_mission(model, prepare_protocol_data(train, t, cfg), cfg, device="cpu") for t in (t1, t2)]
    for k in ("threshold", "normalizer", "alpha_rec", "postprocessing", "validation", "threshold_source"):
        assert reps[0][k] == reps[1][k], k
    assert reps[1]["n_test_events"] == 0 and reps[0]["n_test_events"] == 2
    assert reps[0]["threshold_source"] == "validation"
    # fit_frozen_protocol never needs the test split
    fz = fit_frozen_protocol(model, ProtocolData(train, None, prepare_protocol_data(train, None, cfg).split, []), cfg,
                             device="cpu")
    assert fz["threshold"] == reps[0]["threshold"] and fz["window_size"] == 32


def test_no_positive_in_validation_raises_unless_fallback_is_allowed():
    torch.manual_seed(0)
    train = _stream(1200, [(100, 130)])                           # the only event is too early for any admissible block
    test = _stream(600, [(100, 120)], 2)
    cfg, model = _cfg(), Toy()
    with pytest.raises(ValueError, match="no positive window"):
        evaluate_mission(model, prepare_protocol_data(train, test, cfg), cfg, device="cpu")
    cfg.eval.allow_train_threshold_fallback = True
    rep = evaluate_mission(model, prepare_protocol_data(train, test, cfg), cfg, device="cpu")
    assert rep["threshold_source"] == "train_fallback" and rep["val_has_positives"] is False


def test_few_test_events_warning_and_counts(caplog):
    train = _stream(1200, [(100, 130), (400, 430), (900, 930)])
    test = _stream(600, [(100, 120), (300, 330)], 3)
    cfg = _cfg()
    with caplog.at_level(logging.WARNING, logger="atandn.evaluation.pipeline"):
        rep = evaluate_mission(Toy(), prepare_protocol_data(train, test, cfg), cfg, device="cpu")
    assert any("anomalous event" in r.message for r in caplog.records)
    assert (rep["n_test_events"], rep["n_test_pos_samples"]) == (2, 50)
    assert rep["n_test_pos_windows"] > 0 and rep["n_test_windows"] == len(window_starts(600, 32, 10))


def test_alpha_forced_to_zero_without_reconstruction_head():
    cfg = _cfg()
    cfg.eval.alpha_rec = 0.4
    assert effective_alpha(Toy(recon=True), cfg) == 0.4
    assert effective_alpha(Toy(recon=False), cfg) == 0.0
    cfg.model.use_reconstruction = False
    assert effective_alpha(Toy(recon=True), cfg) == 0.0


def test_mask_buffer_defaults_and_validation():
    e = EvalConfig()
    assert e.mask_buffer_before_minutes == 0.0 and e.mask_buffer_after_minutes is None
    cfg = MissionConfig()
    cfg.eval.mask_buffer_before_minutes = -1.0
    with pytest.raises(ValueError):
        cfg.validate()


def test_known_event_mask_uses_after_buffer_only_by_default():
    train = _stream(1200, [(100, 130), (400, 430), (900, 930), (1050, 1080)])
    train.meta["known_event_intervals"] = [(np.datetime64(500, "m"), np.datetime64(520, "m"))]
    test = _stream(600, [(100, 120)], 4)
    cfg = _cfg()
    cfg.eval.use_known_event_mask = True
    cfg.eval.mask_source = "rare_events"
    cfg.eval.mask_buffer_after_minutes = 30.0
    rep = evaluate_mission(Toy(), prepare_protocol_data(train, test, cfg), cfg, device="cpu")
    assert rep["postprocessing"]["mask_buffer_before_minutes"] == 0.0
    assert rep["postprocessing"]["mask_buffer_after_minutes"] == 30.0 and rep["postprocessing"]["known_event_mask"]


def test_checkpoint_config_conflicts():
    a, b = mission_defaults("m1"), mission_defaults("m1")
    assert checkpoint_config_conflicts(a, b, {"train_ratio": 0.5, "test_stride": 10}) == []
    a.data.test_stride = 5
    a.eval.min_consecutive = 3
    msgs = checkpoint_config_conflicts(a, b, {"train_ratio": 0.5, "test_stride": 10})
    assert any("test_stride" in m for m in msgs) and any("min_consecutive" in m for m in msgs)
