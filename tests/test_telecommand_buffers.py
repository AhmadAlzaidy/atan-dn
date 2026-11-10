"""Stage 2 (paper text): settling buffers from TRAINING telecommands, on synthetic telemetry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from atandn.analysis.telecommand_buffers import binom_sf, derive_settling_buffers, telecommand_intervals
from atandn.inference import apply_known_event_mask

N, TRAIN_END, GAP = 60000, 40000, 2000
T0 = pd.Timestamp("2020-01-01")


def _make(shape: str = "pulse"):
    """1-minute telemetry, 2 channels. Type A every GAP minutes causes a response of known length; type B never."""
    ts = pd.date_range(T0, periods=N, freq="min")
    x = np.zeros((N, 2))
    a_idx = np.arange(1000, N - 400, GAP)                       # type A command sample indices
    b_idx = a_idx[a_idx + 1000 < N - 400] + 1000                 # type B: far from any response
    durs = []
    for i, t in enumerate(a_idx):
        d = 20 + (i * 7) % 41
        if shape == "pulse":
            x[t + 1:t + 1 + d, 0] = 10.0                         # exactly d samples above the band
        else:
            m = np.arange(1, 250)
            x[t + 1:t + 250, 0] = 10.0 * np.exp(-m / 20.0)
        durs.append(d)
    types = np.array(["A"] * len(a_idx) + ["B"] * len(b_idx))
    tc = ts[np.r_[a_idx, b_idx]]
    return ts, x, tc, types, np.array(durs), a_idx


def _derive(x, ts, tc, types, labels=None, **kw):
    return derive_settling_buffers(x, ts, tc, slice(0, TRAIN_END), labels=labels, tc_types=types, **kw)


def test_step_response_gives_expected_settling_quantile():
    ts, x, tc, types, durs, a_idx = _make()
    res = _derive(x, ts, tc, types, np.zeros(N))
    train_durs = durs[a_idx < TRAIN_END - 400]
    exp = float(np.quantile(train_durs, 0.95))
    assert res["selected_types"] == ["A"]                        # B never causes an excursion
    assert res["n_excursion"] == len(train_durs)
    assert abs(res["buffer_minutes_raw"] - exp) <= 1.0           # within one sample
    assert res["buffer_minutes"] == np.ceil(res["buffer_minutes_raw"])
    assert res["max_observed_settling_minutes"] == float(train_durs.max())
    assert abs(res["buffer_fraction_of_max"] - res["buffer_minutes"] / train_durs.max()) < 1e-12
    assert res["n_telecommands_train"] > res["n_excursion"]      # B commands counted but not used
    assert res["fitted_on"] == "training split only"


def test_test_split_telecommands_and_values_never_influence_buffers():
    ts, x, tc, types, _, _ = _make()
    base = _derive(x, ts, tc, types, np.zeros(N))
    # (a) extra commands in the test split (and values that would give a huge settling there)
    x2 = x.copy()
    x2[TRAIN_END + 100:TRAIN_END + 400, 0] = 10.0
    tc2 = tc.append(pd.DatetimeIndex([ts[TRAIN_END + 99], ts[TRAIN_END + 5000]]))
    types2 = np.r_[types, ["A", "A"]]
    assert _derive(x2, ts, tc2, types2, np.zeros(N)) == base
    # (b) permuted test values and labels
    rng = np.random.default_rng(0)
    x3, y3 = x.copy(), np.zeros(N)
    x3[TRAIN_END:] = x[TRAIN_END:][rng.permutation(N - TRAIN_END)]
    y3[TRAIN_END:] = 1.0
    assert _derive(x3, ts, tc, types, y3) == base
    # (c) the same on the training part alone
    assert derive_settling_buffers(x[:TRAIN_END], ts[:TRAIN_END], tc, slice(None), labels=np.zeros(TRAIN_END),
                                   tc_types=types) == base


def test_buffer_is_monotone_in_k_and_quantile():
    ts, x, tc, types, _, _ = _make("decay")
    b = {k: _derive(x, ts, tc, types, np.zeros(N), k=k)["buffer_minutes"] for k in (2.0, 3.0, 6.0)}
    assert b[2.0] >= b[3.0] >= b[6.0] and b[2.0] > b[6.0] > 0
    q = [_derive(x, ts, tc, types, np.zeros(N), quantile=q_)["buffer_minutes"] for q_ in (0.5, 0.95)]
    assert q[0] <= q[1]


def test_labelled_anomaly_in_window_is_excluded():
    ts, x, tc, types, _, a_idx = _make()
    y = np.zeros(N)
    y[a_idx[3] + 150:a_idx[3] + 160] = 1                         # anomaly inside the 300-minute response window
    res = _derive(x, ts, tc, types, y)
    assert res["skipped"]["anomaly_in_window"] >= 1
    assert res["n_excursion"] == _derive(x, ts, tc, types, np.zeros(N))["n_excursion"] - 1


def test_no_selection_without_excursions_gives_zero_buffer():
    ts, x, tc, types, _, _ = _make()
    res = _derive(np.zeros_like(x), ts, tc, types, np.zeros(N))
    assert res["buffer_minutes"] == 0.0 and res["selected_types"] == [] and res["n_excursion"] == 0


def test_intervals_after_only_and_mask():
    ts, x, tc, types, _, a_idx = _make()
    res = _derive(x, ts, tc, types, np.zeros(N))
    iv = telecommand_intervals(tc, types, res["selected_types"], lo=ts[TRAIN_END - 1])
    assert len(iv) == int((a_idx >= TRAIN_END).sum())            # only A, only the test part
    s = apply_known_event_mask(np.ones(N), ts, iv, 0.0, res["buffer_minutes"])
    t = ts.get_loc(iv[0][0])
    assert s[t - 1] == 1.0 and s[t] == 0.0 and s[t + int(res["buffer_minutes"])] == 0.0
    assert s[t + int(res["buffer_minutes"]) + 1] == 1.0          # after-only, ends at t + buffer


def test_binom_tail_matches_direct_sum():
    from math import comb
    n, p = 12, 0.3
    for k in (0, 1, 5, 12, 13):
        exact = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
        assert abs(binom_sf(k, n, p) - exact) < 1e-12
