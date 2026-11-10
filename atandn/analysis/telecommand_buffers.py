"""Stage 2 as the paper describes it: maneuver buffers from TRAINING-split telecommand execution times.

Paper (``s5_setup.tex``, Stage 2): "maneuver exclusion buffers are calculated strictly from training-split telecommand
execution timestamps by measuring the 95th percentile settling duration required for telemetry to return within
+/- 3 sigma of nominal baseline".

Procedure implemented here (training data only; the test split is never passed in):

1. **Events.** The training telecommand executions ``(time, type)`` inside the training slice (duplicates removed).
   Every execution is one event; at most ``max_events_per_type`` executions per type are analysed (evenly spaced, so
   the cost stays bounded for the 1.6 and 1.9 million executions of M1/M2).
2. **Baseline.** For an event at time ``t`` the nominal baseline of every channel is its mean ``mu_c`` and standard
   deviation ``sigma_c`` over the pre-command window ``[t - baseline_minutes, t]``, computed from samples that are not
   labelled anomalous (training labels). ``sigma_c`` is floored at ``sigma_floor_frac`` times the nominal standard
   deviation of the channel over the whole training slice (zero-order-hold telemetry is often constant in a window).
   Events with fewer than ``min_baseline_samples`` valid baseline samples, a post-command window that is incomplete,
   or a post-command window that contains a labelled anomaly are skipped (counted).
3. **Excursion and settling.** The aggregate deviation is ``a(t') = max_c |x_c(t') - mu_c| / sigma_c`` for ``t'`` in
   ``(t, t + max_minutes]``. The command causes an *excursion* if ``a`` exceeds ``k`` for at least
   ``min_run_samples`` consecutive samples (a sustained exceedance, not one noisy sample) starting within
   ``onset_minutes`` of ``t``. The excursion lasts while exceedances follow each other with gaps shorter than ``hold_minutes``; the
   *settling duration* is the time from ``t`` until telemetry is back within ``+/- k sigma`` and stays there for
   ``hold_minutes`` (an excursion that is still running at ``t + max_minutes`` is censored at ``max_minutes`` and
   counted).
4. **Selection of the telecommands that matter (training-derived).** The training data contain ~1.6 million (M1) and
   ~1.9 million (M2) executions of 681 and 123 command types; masking after all of them would hide most of the
   timeline. A TYPE is selected if (i) at least ``min_events_per_type`` of its executions were analysed, (ii) its
   excursion rate is at least ``min_excursion_rate`` and (iii) the rate is higher than the chance rate ``p0`` of the
   same excursion test at ``n_reference`` random training times (one-sided exact binomial test, Bonferroni-corrected
   level ``selection_alpha / n_types``). Only the selected types are masked, at training and at test time.
5. **Buffer.** ``buffer = ceil(quantile of the settling durations)`` of the excursion events of the selected types
   (the paper: quantile 0.95). The mask intervals are ``[t_tc, t_tc + buffer]`` (after the command only).

All statistics are returned (counts, selection, percentiles, maximum observed settling and the buffer as a fraction of
it) so that they can be logged and put into the paper's derived-parameter table. Deviations from the paper text: the
baseline window length, the sigma floor, the hold time and the selection rule are not specified there; they are
parameters of this function and reported.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from atandn.data.splits import to_naive_utc

log = logging.getLogger(__name__)

_NS_MIN = 60e9


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact upper tail ``P(X >= k)`` of ``Binomial(n, p)`` (log-space sum, no SciPy)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k, n + 1)]
    m = max(terms)
    return float(min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms)))


class _Stream:
    """Standardised training stream with prefix sums for the pre-command baseline statistics."""

    def __init__(self, values: np.ndarray, ts_ns: np.ndarray, labels: np.ndarray | None, sigma_floor_frac: float):
        x = np.asarray(values, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        n, c = x.shape
        anom = np.zeros(n, dtype=bool) if labels is None else (np.asarray(labels) > 0)
        fin = np.isfinite(x)
        valid = fin & ~anom[:, None]
        if valid.sum() == 0:
            raise ValueError("no valid (finite, non-anomalous) training sample")
        xz = np.where(valid, x, np.nan)
        mean, std = np.nanmean(xz, axis=0), np.nanstd(xz, axis=0)
        std = np.where(std > 0, std, 1.0)
        self.xs = np.where(fin, (x - mean) / std, 0.0)            # standardised, non-finite -> nominal 0
        self.floor = float(sigma_floor_frac)                      # floor in standardised units
        self.ts, self.n, self.c = ts_ns, n, c
        self.anom_cum = np.r_[0, np.cumsum(anom)].astype(np.int64)
        v = valid.astype(np.float64)
        self.cnt = np.r_[np.zeros((1, c)), np.cumsum(v, axis=0)]
        self.s1 = np.r_[np.zeros((1, c)), np.cumsum(self.xs * v, axis=0)]
        self.s2 = np.r_[np.zeros((1, c)), np.cumsum(self.xs ** 2 * v, axis=0)]
        d = np.diff(ts_ns)
        self.dt_ns = float(np.median(d[d > 0])) if (d > 0).any() else 1.0

    def baseline(self, t_ns: np.ndarray, baseline_ns: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(mu, sigma, n_valid_min)`` of the pre-command window ``[t - B, t]`` for every event time."""
        i1 = np.searchsorted(self.ts, t_ns, side="right")
        i0 = np.searchsorted(self.ts, t_ns - baseline_ns, side="left")
        n = self.cnt[i1] - self.cnt[i0]
        safe = np.maximum(n, 1.0)
        mu = (self.s1[i1] - self.s1[i0]) / safe
        var = np.maximum((self.s2[i1] - self.s2[i0]) / safe - mu ** 2, 0.0)
        return mu, np.maximum(np.sqrt(var), self.floor), n.min(axis=1)


def _responses(st: _Stream, t_ns: np.ndarray, k: float, max_minutes: float, baseline_minutes: float,
               hold_minutes: float, onset_minutes: float, min_baseline_samples: int, min_run_samples: int = 3,
               chunk: int = 128) -> dict:
    """Excursion/settling analysis of the events at ``t_ns``; arrays aligned with ``t_ns``.

    Returns ``status`` (0 ok, 1 baseline too short, 2 incomplete window, 3 anomaly in window), ``excursion`` (bool)
    ``settle`` (minutes, NaN without excursion) and ``censored`` (bool).
    """
    m = len(t_ns)
    status = np.zeros(m, dtype=np.int8)
    exc = np.zeros(m, dtype=bool)
    settle = np.full(m, np.nan)
    cens = np.zeros(m, dtype=bool)
    dt = st.dt_ns
    length = int(math.ceil(max_minutes * _NS_MIN / dt)) + 1
    hold_s = max(int(math.ceil(hold_minutes * _NS_MIN / dt)), 1)
    onset_s = int(math.floor(onset_minutes * _NS_MIN / dt))
    idx0 = np.searchsorted(st.ts, t_ns, side="right")                   # first sample strictly after the command
    mu, sg, nmin = st.baseline(t_ns, baseline_minutes * _NS_MIN)
    status[nmin < min_baseline_samples] = 1
    status[(status == 0) & (idx0 + length > st.n)] = 2
    ok = np.where(status == 0)[0]
    if ok.size:
        a = st.anom_cum[np.minimum(idx0[ok] + length, st.n)] - st.anom_cum[idx0[ok]]
        status[ok[a > 0]] = 3
    ok = np.where(status == 0)[0]
    ar = np.arange(length)
    for s in range(0, ok.size, chunk):
        sel = ok[s:s + chunk]
        win = st.xs[idx0[sel][:, None] + ar[None, :]]                    # (B, L, C)
        agg = (np.abs(win - mu[sel][:, None, :]) / sg[sel][:, None, :]).max(axis=2)
        over = agg > k
        for row, e in zip(over, sel):
            d = np.diff(np.r_[0, row.astype(np.int8), 0])
            starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            q = np.flatnonzero(((ends - starts) >= min_run_samples) & (starts <= onset_s))
            if q.size == 0:                                              # no sustained exceedance soon after t
                continue
            pos = np.flatnonzero(row)
            pos = pos[pos >= starts[q[0]]]
            brk = np.flatnonzero(np.diff(pos) > hold_s)
            end = int(pos[brk[0]]) if brk.size else int(pos[-1])
            exc[e] = True
            if not brk.size and end >= length - 1 - hold_s:               # still running at the horizon
                cens[e] = True
                settle[e] = max_minutes
            else:
                settle[e] = min(max_minutes, (end + 1) * dt / _NS_MIN)
    return {"status": status, "excursion": exc, "settle": settle, "censored": cens}


def _empty_result(reason: str) -> dict:
    return {"buffer_minutes": 0.0, "buffer_minutes_raw": 0.0, "n_excursion": 0, "selected_types": [],
            "n_censored": 0, "settling": {}, "max_observed_settling_minutes": 0.0, "buffer_fraction_of_max": None,
            "reason": reason, "fitted_on": "training split only"}


def derive_settling_buffers(
    values: np.ndarray,
    timestamps,
    tc_times,
    train_slice: slice,
    k: float = 3.0,
    quantile: float = 0.95,
    max_minutes: float = 300.0,
    baseline_minutes: float = 120.0,
    labels: np.ndarray | None = None,
    tc_types=None,
    hold_minutes: float = 10.0,
    onset_minutes: float = 30.0,
    min_baseline_samples: int = 10,
    min_run_samples: int = 3,
    sigma_floor_frac: float = 0.05,
    max_events_per_type: int = 300,
    min_events_per_type: int = 5,
    selection_alpha: float = 0.01,
    min_excursion_rate: float = 0.5,
    n_reference: int = 3000,
    seed: int = 0,
    ceil: bool = True,
) -> dict:
    """95th-percentile settling duration after training telecommands (see the module docstring for the procedure).

    Only ``values[train_slice]``, ``timestamps[train_slice]`` and ``labels[train_slice]`` are read; telecommands
    outside the time range of that slice are ignored, so telecommands of the test split cannot influence the result.

    Args:
        values: ``(N, C)`` telemetry (whole stream or the training part).
        timestamps: ``(N,)`` timestamps (any form accepted by ``to_naive_utc``), increasing.
        tc_times: telecommand execution times (UTC), any order.
        train_slice: the training part of ``values``/``timestamps``/``labels``.
        k: width of the nominal band in sigmas. ``quantile``: percentile of the settling durations (0.95).
        max_minutes: observation horizon after a command (300 as in the FP analysis); longer excursions are censored.
        baseline_minutes: length of the pre-command window for the baseline mean/std.
        labels: ``(N,)`` 0/1 anomaly labels; labelled samples are excluded from baselines and events whose
            post-command window contains one are skipped. ``None``: nothing is excluded (a WARNING is logged).
        tc_types: telecommand type of every entry of ``tc_times`` (same length). Needed for the selection of the
            command types that cause excursions; ``None``: all executions are one group (no selection possible).
        hold_minutes / onset_minutes / min_baseline_samples / min_run_samples / sigma_floor_frac: see the module docstring.
        max_events_per_type / min_events_per_type / selection_alpha / min_excursion_rate / n_reference / seed:
            selection of the types (module docstring, step 4); the subsampling is deterministic.
        ceil: round the buffer up to whole minutes.

    Returns:
        JSON-able dict with ``buffer_minutes`` (``buffer_minutes_raw`` unrounded), ``selected_types``, ``n_telecommands_train``
        (executions in the training slice), ``n_types_train``, ``n_events_analysed``, ``n_excursion`` (events of the
        selected types used for the percentile), ``n_excursion_all_types``, ``n_censored``, skip counts, ``settling``
        (p50/p90/p95/p99/max in minutes), ``max_observed_settling_minutes``, ``buffer_fraction_of_max``,
        ``reference_excursion_rate`` and the parameters used.
    """
    sl = slice(None) if train_slice is None else train_slice
    x = np.asarray(values)[sl]
    ts = to_naive_utc(np.asarray(timestamps)[sl])
    if len(ts) == 0:
        raise ValueError("empty training slice")
    ts_ns = ts.values.astype("datetime64[ns]").astype(np.int64)
    if np.any(np.diff(ts_ns) < 0):
        raise ValueError("timestamps of the training slice must be non-decreasing")
    y = None if labels is None else np.asarray(labels)[sl]
    if y is None:
        log.warning("derive_settling_buffers: no labels given, labelled anomalies are NOT excluded from the baseline")
    params = {"k": k, "quantile": quantile, "max_minutes": max_minutes, "baseline_minutes": baseline_minutes,
              "hold_minutes": hold_minutes, "onset_minutes": onset_minutes, "sigma_floor_frac": sigma_floor_frac,
              "min_baseline_samples": min_baseline_samples, "min_run_samples": min_run_samples, "max_events_per_type": max_events_per_type,
              "min_events_per_type": min_events_per_type, "selection_alpha": selection_alpha,
              "min_excursion_rate": min_excursion_rate, "n_reference": n_reference, "seed": seed}

    tc = pd.DataFrame({"time": to_naive_utc(tc_times)})
    tc["type"] = "all" if tc_types is None else np.asarray(tc_types).astype(str)
    tc = tc[(tc["time"] >= ts[0]) & (tc["time"] <= ts[-1])].drop_duplicates().sort_values(["type", "time"], kind="stable")
    n_tc = int(len(tc))
    n_types = int(tc["type"].nunique())
    common = {"n_telecommands_train": n_tc, "n_types_train": n_types, "params": params,
              "train_time_range": [str(ts[0]), str(ts[-1])], "typed": tc_types is not None}
    if n_tc == 0:
        log.warning("no telecommand inside the training slice: buffer 0")
        return {**_empty_result("no telecommands in the training slice"), **common}

    st = _Stream(x, ts_ns, y, sigma_floor_frac)
    rng = np.random.default_rng(seed)

    # events analysed: at most max_events_per_type evenly spaced executions per type
    keep = []
    for _, g in tc.groupby("type", sort=True):
        ix = g.index.to_numpy()
        if len(ix) > max_events_per_type:
            ix = ix[np.linspace(0, len(ix) - 1, max_events_per_type).round().astype(int)]
        keep.append(ix)
    ev = tc.loc[np.concatenate(keep)]
    ev_ns = ev["time"].values.astype("datetime64[ns]").astype(np.int64)
    kw = dict(k=k, max_minutes=max_minutes, baseline_minutes=baseline_minutes, hold_minutes=hold_minutes,
              onset_minutes=onset_minutes, min_baseline_samples=min_baseline_samples, min_run_samples=min_run_samples)
    res = _responses(st, ev_ns, **kw)
    ok = res["status"] == 0
    skips = {"baseline_too_short": int((res["status"] == 1).sum()), "incomplete_window": int((res["status"] == 2).sum()),
             "anomaly_in_window": int((res["status"] == 3).sum())}

    # chance excursion rate: same test at random training times (unconditional on commands)
    ref_pos = rng.choice(st.n, size=min(int(n_reference), st.n), replace=False)
    ref = _responses(st, ts_ns[np.sort(ref_pos)], **kw)
    ref_ok = ref["status"] == 0
    p0 = float(ref["excursion"][ref_ok].mean()) if ref_ok.any() else 0.0
    p0 = max(p0, 1.0 / max(int(ref_ok.sum()), 1))

    df = pd.DataFrame({"type": ev["type"].to_numpy(), "ok": ok, "exc": res["excursion"] & ok})
    per = df[df["ok"]].groupby("type").agg(n=("ok", "size"), n_exc=("exc", "sum"))
    per["rate"] = per["n_exc"] / per["n"]
    level = selection_alpha / max(len(per), 1)
    per["p_value"] = [binom_sf(int(e), int(n), p0) for e, n in zip(per["n_exc"], per["n"])]
    if tc_types is None:
        per["selected"] = per["n"] >= 1
    else:
        per["selected"] = (per["n"] >= min_events_per_type) & (per["rate"] >= min_excursion_rate) & (per["p_value"] < level)
    selected = sorted(per.index[per["selected"]].tolist())

    in_sel = df["type"].isin(selected).to_numpy() & df["exc"].to_numpy()
    dur = res["settle"][in_sel]
    out: dict[str, Any] = {
        **common, "n_events_analysed": int(ok.sum()), "n_events_sampled": int(len(ev)), "skipped": skips,
        "n_excursion_all_types": int(df["exc"].sum()), "reference_excursion_rate": p0,
        "n_types_selected": len(selected), "selected_types": selected, "selection_level": level,
        "selection_rule": ("type selected if >= min_events_per_type analysed events, excursion rate >= "
                           "min_excursion_rate and one-sided binomial p < selection_alpha / n_types_tested vs the "
                           "reference rate at random training times" if tc_types is not None else
                           "no types given: all executions form one group, no selection"),
        "n_types_tested": int(len(per)),
    }
    if dur.size == 0:
        log.warning("no excursion after the selected telecommands in the training slice: buffer 0")
        return {**_empty_result("no excursion events"), **out}
    raw = float(np.quantile(dur, quantile))
    buf = float(math.ceil(raw)) if ceil else raw
    mx = float(dur.max())
    n_cens = int(res["censored"][in_sel].sum())
    if raw >= max_minutes:
        log.warning("the %.0f%% settling quantile reached the horizon max_minutes=%.0f: %d of %d excursions never settled "
                    "within the horizon (persistent level shift?), so the buffer equals the horizon and is not a "
                    "settling time", 100 * quantile, max_minutes, n_cens, dur.size)
    out.update({
        "buffer_minutes": buf, "buffer_minutes_raw": raw, "n_excursion": int(dur.size),
        "n_censored": n_cens,
        "settling": {"p50": float(np.quantile(dur, 0.5)), "p90": float(np.quantile(dur, 0.9)),
                     "p95": float(np.quantile(dur, 0.95)), "p99": float(np.quantile(dur, 0.99)), "max": mx},
        "max_observed_settling_minutes": mx, "buffer_fraction_of_max": float(buf / mx) if mx > 0 else float("nan"),
        "fitted_on": "training split only",
    })
    log.info("settling buffer %.0f min (P%.0f = %.1f min, max %.1f min) from %d excursion events of %d selected "
             "types (of %d analysed types, %d training executions)", buf, 100 * quantile, raw, mx, dur.size,
             len(selected), len(per), n_tc)
    return out


def telecommand_intervals(tc_times, tc_types, selected_types, lo=None, hi=None) -> list[tuple]:
    """Point intervals ``(t, t)`` of the executions of the selected types with ``lo < t <= hi`` (sorted, unique times).

    Combined with ``apply_known_event_mask(..., buffer_after_minutes=buffer)`` this gives the paper's mask
    ``[t_tc, t_tc + buffer]``. ``selected_types=None`` takes every execution. Only the execution times are used
    (the operational schedule), no telemetry and no label.
    """
    t = to_naive_utc(tc_times)
    keep = np.ones(len(t), dtype=bool)
    if selected_types is not None:
        keep &= np.isin(np.asarray(tc_types).astype(str), np.asarray(list(selected_types), dtype=str))
    if lo is not None:
        keep &= np.asarray(t > pd.Timestamp(lo))
    if hi is not None:
        keep &= np.asarray(t <= pd.Timestamp(hi))
    u = pd.DatetimeIndex(t[keep]).unique().sort_values()
    return list(zip(u, u))
