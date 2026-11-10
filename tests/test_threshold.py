"""Threshold, normaliser and post-processing tests (everything fitted on training data only)."""
import numpy as np
import pandas as pd
import pytest

from atandn.evaluation.metrics import window_metrics
from atandn.inference import (ScoreNormalizer, apply_known_event_mask, dual_evidence, f1_optimal_threshold,
                              min_consecutive_filter)


def _brute_force(scores, labels):
    cands = np.unique(scores)
    cands = np.r_[cands[0] - 1, (cands[1:] + cands[:-1]) / 2, cands[-1] + 1]
    return max(window_metrics(labels, scores > t)["f1"] for t in cands)


def test_threshold_is_f1_optimal_on_training_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.2).astype(int)
    s = rng.normal(0, 1, 300) + 1.5 * y
    thr = f1_optimal_threshold(s, y)
    assert window_metrics(y, s > thr)["f1"] == pytest.approx(_brute_force(s, y))


def test_threshold_depends_only_on_training_data():
    rng = np.random.default_rng(2)
    y_tr = (rng.random(200) < 0.3).astype(int)
    s_tr = rng.normal(0, 1, 200) + 2 * y_tr
    thr = f1_optimal_threshold(s_tr, y_tr)
    # the threshold is a function of (s_tr, y_tr) only: it must not change when the test scores change, nor when
    # the training arrays are copied; it must change when the TRAINING scores change.
    s_tr_copy, y_tr_copy = s_tr.copy(), y_tr.copy()
    for shift in (0.0, 5.0, -5.0):
        s_te = rng.normal(shift, 1, 100)
        pred_before = (s_te > thr)
        thr2 = f1_optimal_threshold(s_tr_copy, y_tr_copy)
        assert thr2 == thr
        assert np.array_equal(s_tr, s_tr_copy) and np.array_equal(y_tr, y_tr_copy)  # inputs untouched
        assert np.array_equal(pred_before, s_te > thr2)
        # the positive rate follows the test scores (shift +5 -> all positive, -5 -> none) while thr stays fixed
        if shift > 0:
            assert (s_te > thr2).all()
        elif shift < 0:
            assert not (s_te > thr2).any()
    assert f1_optimal_threshold(s_tr + 3.0, y_tr) != thr


def test_threshold_without_positives():
    s = np.linspace(0, 1, 100)
    with pytest.raises(ValueError):
        f1_optimal_threshold(s, np.zeros(100))
    assert f1_optimal_threshold(s, np.zeros(100), fallback_quantile=0.99) == pytest.approx(np.quantile(s, 0.99))


def test_normalizer_fitted_on_train_clips_test():
    train = np.linspace(0, 1, 101)
    n = ScoreNormalizer("minmax").fit(train)
    out = n.transform(np.array([-1.0, 0.5, 10.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]
    assert (n.lo, n.hi) == (0.0, 1.0)
    n2 = ScoreNormalizer.from_state_dict(n.state_dict())
    assert n2.transform(np.array([0.25]))[0] == pytest.approx(0.25)
    with pytest.raises(RuntimeError):
        ScoreNormalizer().transform(train)


def test_dual_evidence_formula():
    n = ScoreNormalizer("minmax").fit(np.array([0.0, 2.0]))
    s = dual_evidence(np.array([0.8]), np.array([1.0]), 0.25, n)
    assert s[0] == pytest.approx(0.75 * 0.8 + 0.25 * 0.5)


def test_min_consecutive_filter():
    p = np.array([0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    assert min_consecutive_filter(p, 1).tolist() == p.tolist()
    assert min_consecutive_filter(p, 2).tolist() == [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0]
    assert min_consecutive_filter(p, 3).tolist() == [0] * 6 + [1, 1, 1, 0, 0]


def test_known_event_mask_with_buffer_and_tz():
    ts = pd.date_range("2020-01-01", periods=10, freq="10min", tz="UTC")
    scores = np.ones(10)
    iv = [(pd.Timestamp("2020-01-01 00:30"), pd.Timestamp("2020-01-01 00:40"))]
    out = apply_known_event_mask(scores, ts, iv)
    assert out.tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
    out = apply_known_event_mask(scores, ts, iv, buffer_after_minutes=10)
    assert out.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
    assert scores.sum() == 10                      # input not modified
