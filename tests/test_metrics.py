"""Metric tests with known confusion matrices."""
import numpy as np
import pytest

from atandn.evaluation.metrics import point_adjusted_metrics, pointwise_metrics, window_metrics, windows_to_samples


def test_window_metrics_known_confusion():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    m = window_metrics(y, p)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 5, 1)
    assert m["precision"] == pytest.approx(0.75) and m["recall"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6) and m["fnr"] == pytest.approx(0.25)
    assert m["f1"] == pytest.approx(0.75)


def test_f05_formula():
    y = np.array([1] * 6 + [0] * 4)
    p = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])   # tp=3 fp=2 fn=3
    m = window_metrics(y, p)
    P, R = 3 / 5, 3 / 6
    assert m["f0_5"] == pytest.approx(1.25 * P * R / (0.25 * P + R))
    assert m["f0_5"] != pytest.approx(m["f1"])


def test_degenerate_cases_do_not_divide_by_zero():
    m = window_metrics(np.zeros(5), np.zeros(5))
    assert m["f1"] == 0.0 and m["precision"] == 0.0 and m["tn"] == 5


def test_windows_to_samples_rule_a():
    n, w, s = 60, 20, 10
    end = np.array([19, 29, 39, 49, 59])
    pred = np.array([0, 1, 0, 0, 1])
    sp, valid = windows_to_samples(pred, end, n, stride=s, window_size=w)
    assert sp[20:30].tolist() == [1] * 10 and sp[30:40].sum() == 0 and sp[50:60].tolist() == [1] * 10
    assert not valid[:19].any()          # first window_size-1 samples excluded
    assert valid[19:].all()
    # the tail not covered by a window is invalid
    sp2, valid2 = windows_to_samples(pred[:3], end[:3], n, stride=s, window_size=w)
    assert not valid2[40:].any() and valid2[19:40].all()


def test_pointwise_metrics_uses_valid_mask():
    yt = np.array([0, 0, 1, 1, 0])
    yp = np.array([1, 0, 1, 0, 0])
    full = pointwise_metrics(yt, yp)
    part = pointwise_metrics(yt, yp, np.array([False, True, True, True, True]))
    assert full["fp"] == 1 and part["fp"] == 0 and part["tp"] == 1


def test_point_adjusted_metrics():
    yt = np.array([0, 1, 1, 1, 1, 0])
    yp = np.array([0, 0, 1, 0, 0, 0])
    adj = point_adjusted_metrics(yt, yp)
    assert adj["recall"] == 1.0 and window_metrics(yt, yp)["recall"] == 0.25
