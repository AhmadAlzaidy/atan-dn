"""Round-2 review fixes: progressive full-event validation split, in-block threshold windows, spike filter in the
threshold search, NaN scores, data identity."""
from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from atandn.config import EvalConfig, MissionConfig
from atandn.data import event_aware_validation_split, window_starts
from atandn.evaluation.pipeline import evaluate_mission, fit_frozen_protocol, prepare_protocol_data
from atandn.inference import grid_f1_threshold, select_threshold, select_threshold_filtered
from test_protocol import Toy, _cfg, _full_events, _labels, _stream


# ---------------------------------------------------------------------------------------------------------
# progressive / full-event validation split (M1, M2)
# ---------------------------------------------------------------------------------------------------------
def test_events_per_block_counts_only_full_events_bruteforce():
    from atandn.data.splits import _events_per_block
    rng = np.random.default_rng(0)
    for _ in range(150):
        n = int(rng.integers(30, 120))
        y = np.zeros(n, dtype=int)
        for _k in range(int(rng.integers(0, 6))):
            a = int(rng.integers(0, n))
            y[a:a + int(rng.integers(1, 12))] = 1
        n_val = int(rng.integers(5, n // 2))
        got = _events_per_block(y, n_val)
        want = [_full_events(y, s, s + n_val) for s in range(n - n_val + 1)]
        assert got.tolist() == want


def test_clipped_event_is_not_counted_as_full():
    """An event cut by the block boundary must not satisfy min_events (review M2: 1 sample inside the block)."""
    y = _labels(1000, [(190, 205)])                     # tail empty; a block starting at 200 would only clip it
    vs = event_aware_validation_split(y, 0.2, gap=31, min_events=1, min_fit=32)
    assert vs.report["val_events"] == 1 and vs.report["tail_moved"] is True
    s = vs.val.start
    assert s <= 190 and s + 200 >= 205                  # the block contains the whole event


def test_progressive_relaxation_m1_single_event_block(caplog):
    """M1: events only at the start; a block with 1 full event exists, so it is used instead of the empty tail."""
    y = _labels(1000, [(0, 10), (50, 60), (100, 110)])
    with caplog.at_level(logging.WARNING, logger="atandn.data.splits"):
        vs = event_aware_validation_split(y, 0.2, gap=31, min_events=3, min_fit=32)
    assert vs.report["val_has_positives"] is True and vs.report["val_events"] == 1
    assert vs.report["min_events_used"] == 1 and vs.report["min_events_satisfied"] is False
    assert vs.val.start == 100 and vs.fit.stop == 100 - 31
    msgs = [r.message for r in caplog.records]
    assert sum("relaxing" in m for m in msgs) == 2          # 3 -> 2 and 2 -> 1


def test_progressive_relaxation_four_events_in_head_none_in_tail():
    y = _labels(2400, [(300, 310), (900, 910), (1500, 1510), (1800, 1810)])
    vs = event_aware_validation_split(y, 0.2, gap=31, min_events=3, min_fit=32, min_fit_fraction=0.3)
    assert vs.report["tail_moved"] and vs.report["val_events"] == 2 and vs.report["min_events_used"] == 2
    assert (vs.val.start, vs.val.stop) == (1500, 1980)


def test_min_fit_fraction_prevents_degenerate_fit_part_m2_like():
    """Real-M2-like: positives only in the first half of the training split, none in the tail."""
    y = _labels(3000, [(100, 110), (400, 410), (700, 710), (1000, 1010), (1300, 1310)])
    free = event_aware_validation_split(y, 0.25, gap=31, min_events=3, min_fit=32, min_fit_fraction=0.0)
    assert free.val.start == 700 and free.report["val_events"] == 3         # without the guard: fit part 22% only
    vs = event_aware_validation_split(y, 0.25, gap=31, min_events=3, min_fit=32, min_fit_fraction=0.3)
    assert vs.fit.stop >= 0.3 * 3000 - 1                                   # fit part keeps >= 30% of the split
    assert vs.report["tail_moved"] and vs.report["val_events"] == 2 and vs.report["min_events_used"] == 2
    assert vs.val.start == 1000 and vs.report["val_has_positives"]


def test_no_full_event_anywhere_keeps_tail_without_positives():
    y = _labels(1000, [(300, 600)])                         # one event longer than the block (200)
    vs = event_aware_validation_split(y, 0.2, gap=31, min_events=3, min_fit=32)
    assert (vs.val.start, vs.val.stop) == (800, 1000) and vs.report["tail_moved"] is False
    assert vs.report["val_has_positives"] is False and vs.report["min_events_used"] == 0


def test_config_val_min_fit_fraction_default_and_validation():
    assert MissionConfig().train.val_min_fit_fraction == 0.3
    c = MissionConfig()
    c.train.val_min_fit_fraction = 0.9
    with pytest.raises(ValueError):
        c.validate()


# ---------------------------------------------------------------------------------------------------------
# H1: threshold windows lie inside the validation block
# ---------------------------------------------------------------------------------------------------------
def test_threshold_uses_only_windows_inside_a_moved_validation_block(monkeypatch):
    from atandn.evaluation import pipeline
    torch.manual_seed(0)
    train = _stream(1200, [(100, 130), (400, 430), (500, 520), (600, 630)])   # tail [900, 1200) has no event
    cfg, model = _cfg(), Toy()
    data = prepare_protocol_data(train, None, cfg)
    sp = data.split
    assert sp.report["tail_moved"] and (sp.val.start, sp.val.stop) == (400, 700) and sp.fit.stop == 369
    seen = {}
    real = pipeline.select_threshold_filtered

    def spy(scores, labels, method="grid", min_len=1, fallback_quantile=None):
        seen["n"], seen["pos"] = len(scores), int(np.sum(labels))
        return real(scores, labels, method, min_len, fallback_quantile)

    monkeypatch.setattr(pipeline, "select_threshold_filtered", spy)
    fz = fit_frozen_protocol(model, data, cfg, device="cpu")
    w, stride = 32, cfg.data.test_stride
    starts = window_starts(len(train), w, stride)
    inside = [s0 for s0 in starts if s0 >= 400 and s0 + w <= 700]
    fit_w = [s0 for s0 in starts if s0 + w <= sp.fit.stop]
    assert len(inside) == 27 == (300 - w) // stride + 1
    assert fz["validation"]["n_val_windows"] == seen["n"] == 27             # none of the windows after the block
    assert fz["validation"]["n_fit_windows"] == len(fit_w)
    assert max(s0 + w for s0 in fit_w) <= min(inside)                       # no fit window overlaps a validation window
    assert seen["pos"] == fz["validation"]["n_val_pos_windows"] > 0


def test_windows_straddling_the_fit_validation_boundary_belong_to_neither():
    torch.manual_seed(0)
    train = _stream(1200, [(100, 130), (400, 430), (500, 520), (600, 630)])
    cfg = _cfg()
    data = prepare_protocol_data(train, None, cfg)
    fz = fit_frozen_protocol(Toy(), data, cfg, device="cpu")
    starts = window_starts(len(train), 32, cfg.data.test_stride)
    fit_ids = {s0 for s0 in starts if s0 + 32 <= data.split.fit.stop}
    val_ids = {s0 for s0 in starts if s0 >= data.split.val.start and s0 + 32 <= data.split.val.stop}
    assert not fit_ids & val_ids
    assert fz["validation"]["n_fit_windows"] + fz["validation"]["n_val_windows"] < len(starts)


# ---------------------------------------------------------------------------------------------------------
# M4: spike filter inside the threshold selection; min_consecutive derived / explicit
# ---------------------------------------------------------------------------------------------------------
def test_threshold_selection_applies_the_spike_filter():
    s = np.full(60, 0.11)
    y = np.zeros(60, dtype=int)
    s[10:15], y[10:15] = [0.9, 0.9, 0.9, 0.4, 0.4], 1          # one event, partly weak
    for i in (25, 32, 40, 50):                                    # isolated false spikes
        s[i] = 0.6
    t1, _ = select_threshold_filtered(s, y, "grid", 1)
    t2, f2 = select_threshold_filtered(s, y, "grid", 2)
    assert t1 == pytest.approx(0.6) and 0.11 <= t2 < 0.4 and f2 == pytest.approx(1.0)
    assert t1 == grid_f1_threshold(s, y)
    _, fe = select_threshold_filtered(s, y, "exact", 2)
    assert fe == pytest.approx(1.0)


def test_min_consecutive_derived_or_explicit_is_logged_and_frozen(caplog):
    torch.manual_seed(0)
    train = _stream(1200, [(100, 130), (400, 430), (900, 930), (1050, 1080)])
    cfg, model = _cfg(), Toy()
    data = prepare_protocol_data(train, None, cfg)
    cfg.eval.min_consecutive = "derived"
    with caplog.at_level(logging.INFO, logger="atandn.evaluation.pipeline"):
        fz = fit_frozen_protocol(model, data, cfg, device="cpu")
    assert fz["min_consecutive_source"] == "derived_from_training"
    assert fz["min_consecutive"] == fz["mask_params"]["min_consecutive_windows"]
    assert any("min_consecutive" in r.message for r in caplog.records)
    cfg.eval.min_consecutive = 3
    fz3 = fit_frozen_protocol(model, data, cfg, device="cpu")
    assert (fz3["min_consecutive"], fz3["min_consecutive_source"]) == (3, "config")
    rep = evaluate_mission(model, prepare_protocol_data(train, _stream(600, [(100, 120)], 3), cfg), cfg, frozen=fz3,
                           device="cpu")
    assert rep["postprocessing"]["min_consecutive"] == 3
    assert EvalConfig().min_consecutive == "derived"
    for bad in (0, "x", 1.5, True):
        c = MissionConfig()
        c.eval.min_consecutive = bad
        with pytest.raises(ValueError):
            c.validate()


# ---------------------------------------------------------------------------------------------------------
# low items: NaN scores, F1 == 0, data identity
# ---------------------------------------------------------------------------------------------------------
def test_grid_threshold_rejects_nan_and_warns_on_zero_f1(caplog):
    s, y = np.array([0.2, np.nan, 0.9, 0.1]), np.array([0, 1, 1, 0])
    with pytest.raises(ValueError, match="NaN"):
        grid_f1_threshold(s, y)
    with pytest.raises(ValueError, match="NaN"):
        select_threshold(s, y, "exact")
    with caplog.at_level(logging.WARNING, logger="atandn.inference.threshold"):
        thr = grid_f1_threshold(np.array([0.01, 0.02, 0.01]), np.array([0, 1, 0]))   # positives below the grid
    assert thr == pytest.approx(0.05) and any("F1 is 0" in r.message for r in caplog.records)


def test_data_identity_mismatch_warns(caplog):
    torch.manual_seed(0)
    train = _stream(1200, [(100, 130), (400, 430), (900, 930), (1050, 1080)])
    test = _stream(600, [(100, 120)], 3)
    cfg, model = _cfg(), Toy()
    fz = fit_frozen_protocol(model, prepare_protocol_data(train, None, cfg), cfg, device="cpu")
    assert fz["data_identity"]["n"] == 1200
    with caplog.at_level(logging.WARNING, logger="atandn.evaluation.pipeline"):
        evaluate_mission(model, prepare_protocol_data(train, test, cfg), cfg, frozen=fz, device="cpu")
    assert not any("differs from the data" in r.message for r in caplog.records)
    other = _stream(1200, [(100, 130), (400, 430), (900, 930), (1050, 1080)], seed=5)
    with caplog.at_level(logging.WARNING, logger="atandn.evaluation.pipeline"):
        evaluate_mission(model, prepare_protocol_data(other, test, cfg), cfg, frozen=fz, device="cpu")
    assert any("differs from the data" in r.message for r in caplog.records)
