"""Split file and score file formats shared by ``scripts/export_split.py``, the baseline runners and the adapter.

Only ``numpy`` and ``pandas`` are imported (no torch), so that the baseline environment (old torch) and the main
environment can both use it.

Split JSON (``atandn-split/1``), written by ``scripts/export_split.py``. All indices are ABSOLUTE sample indices of the
full prepared stream, half-open ``[start, stop)``:

* ``fit``: the part a baseline may train on (data before the validation block, minus the gap),
* ``val``: the validation block carved from the TRAINING split (threshold, normalizer-free model selection),
* ``test``: the test split (``train_stop`` .. ``n_samples``), never used for any fitting,
* ``timestamps``: identity fingerprint (``sha1`` of the naive-UTC int64 nanosecond timestamps, ``n``, first, last),
* ``window_size`` / ``test_stride``: the ATAN-DN window parameters that defined the validation block and its gap.

Score file (``.npz``), written by the baseline runners with ``--export-scores``: for each part ``fit``, ``val``,
``test`` the arrays ``<part>_scores`` (one raw anomaly score per window; higher = more anomalous) and ``<part>_end``
(ABSOLUTE sample index of the LAST sample of each window; the window covers ``end - window_size + 1 .. end``), plus the
scalars ``window_size`` and ``stride`` of the baseline, strings ``model`` and ``mission`` and the identity hash of
the stream (``ts_sha1``). The score of a window uses only samples up to its end (causal).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SPLIT_SCHEMA = "atandn-split/1"
PARTS = ("fit", "val", "test")


def naive_utc_ns(timestamps: Any) -> np.ndarray:
    """Timestamps as timezone-naive UTC ``int64`` nanoseconds (the representation that is hashed)."""
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps, format="mixed"))
    except (TypeError, ValueError):       # pandas < 2 has no format="mixed"
        idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return np.asarray(idx.values.astype("datetime64[ns]")).astype(np.int64)


def timestamps_identity(timestamps: Any) -> dict:
    """Identity fingerprint of a timestamp sequence: ``n``, ``first``, ``last`` and ``sha1`` of the int64 values."""
    ns = naive_utc_ns(timestamps)
    return {"n": int(len(ns)), "first": str(np.datetime64(int(ns[0]), "ns")) if len(ns) else None,
            "last": str(np.datetime64(int(ns[-1]), "ns")) if len(ns) else None,
            "sha1": hashlib.sha1(np.ascontiguousarray(ns).tobytes()).hexdigest()}


def write_split(path: str | Path, split: dict) -> Path:
    """Write a split dictionary as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return p


def validate_split(split: dict) -> dict:
    """Check structure and ordering of a split dictionary; returns it.

    Raises:
        ValueError: missing keys, wrong schema, or ranges that overlap / are out of order.
    """
    if split.get("schema") != SPLIT_SCHEMA:
        raise ValueError(f"split schema must be '{SPLIT_SCHEMA}', got {split.get('schema')!r}")
    for k in ("n_samples", "fit", "val", "test", "timestamps"):
        if k not in split:
            raise ValueError(f"split is missing the key '{k}'")
    r = {p: (int(split[p]["start"]), int(split[p]["stop"])) for p in PARTS}
    for p, (a, b) in r.items():
        if not 0 <= a < b <= int(split["n_samples"]):
            raise ValueError(f"split part '{p}' = [{a}, {b}) is empty or outside [0, {split['n_samples']})")
    if not (r["fit"][1] <= r["val"][0] and r["val"][1] <= r["test"][0]):
        raise ValueError("split parts must be ordered and disjoint: fit < val < test")
    return split


def read_split(path: str | Path) -> dict:
    """Read and validate a split JSON."""
    return validate_split(json.loads(Path(path).read_text(encoding="utf-8")))


def part_range(split: dict, part: str) -> tuple[int, int]:
    """``(start, stop)`` of ``part`` (``fit``, ``val`` or ``test``)."""
    return int(split[part]["start"]), int(split[part]["stop"])


def check_stream_matches_split(split: dict, timestamps: Any) -> None:
    """Raise ``ValueError`` unless the stream loaded now is the one the split was exported for."""
    now = timestamps_identity(timestamps)
    want = split["timestamps"]
    if now["n"] != int(split["n_samples"]) or now["sha1"] != want["sha1"]:
        raise ValueError(
            f"the data loaded now (n={now['n']}, {now['first']} .. {now['last']}) is not the data the split was "
            f"exported for (n={split['n_samples']}, {want.get('first')} .. {want.get('last')}); the sample order or "
            "the file differs, so the indices of the split are meaningless")


def load_stream(path: str | Path, kind: str) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load a prepared mission file exactly as ``atandn.data`` does: ``(values (N, C) float32, labels, timestamps)``.

    ``kind="esa"``: parquet with ``chchannel_*`` feature columns, ``label`` and the timestamp index.
    ``kind="opssat"``: ``segments.csv`` stable-sorted on ``timestamp`` then ``channel`` (as ``atandn.data.load_opssat``),
    one feature ``value``. The labels are returned for completeness; the honest runners never read them.
    """
    path = Path(path)
    if kind == "opssat":
        df = pd.read_csv(path)
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        except (TypeError, ValueError):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["timestamp"] + (["channel"] if "channel" in df.columns else []),
                            kind="stable").reset_index(drop=True)
        anomaly = df["anomaly"]
        labels = ((anomaly.astype(str).str.lower() == "anomaly") if anomaly.dtype == object
                  else anomaly.astype(np.int64) > 0).to_numpy().astype(np.int64)
        return (df["value"].to_numpy(dtype=np.float32).reshape(-1, 1), labels,
                pd.DatetimeIndex(df["timestamp"]))
    if kind != "esa":
        raise ValueError("kind must be 'esa' or 'opssat'")
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c.startswith("chchannel_")] or [c for c in df.columns if c != "label"]
    return (df[feats].to_numpy(dtype=np.float32), df["label"].to_numpy().astype(np.int64),
            pd.DatetimeIndex(df.index))


def window_starts_in(start: int, stop: int, window_size: int, stride: int) -> np.ndarray:
    """Absolute start indices of the windows that lie ENTIRELY inside ``[start, stop)`` at the given stride."""
    if stop - start < window_size:
        return np.zeros(0, dtype=np.int64)
    return start + np.arange(0, stop - start - window_size + 1, stride, dtype=np.int64)


def save_scores(path: str | Path, parts: dict[str, tuple[np.ndarray, np.ndarray]], *, window_size: int, stride: int,
                model: str, mission: str, ts_sha1: str, extra: dict | None = None) -> Path:
    """Write the score file (see the module docstring). ``parts`` maps ``fit``/``val``/``test`` to ``(scores, end)``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for name, (s, e) in parts.items():
        if name not in PARTS:
            raise ValueError(f"unknown part '{name}'")
        s, e = np.asarray(s, dtype=np.float64), np.asarray(e, dtype=np.int64)
        if s.shape != e.shape:
            raise ValueError(f"part '{name}': scores and end indices differ in shape")
        arrays[f"{name}_scores"], arrays[f"{name}_end"] = s, e
    np.savez(p, window_size=np.int64(window_size), stride=np.int64(stride), model=np.asarray(model),
             mission=np.asarray(mission), ts_sha1=np.asarray(ts_sha1), **arrays, **(extra or {}))
    log.info("wrote scores %s (%s)", p, ", ".join(f"{k}: {len(v[0])}" for k, v in parts.items()))
    return p


def load_scores(path: str | Path) -> dict:
    """Read a score file: ``{"window_size", "stride", "model", "mission", "ts_sha1", "fit": (scores, end), ...}``."""
    z = np.load(Path(path), allow_pickle=False)
    out: dict[str, Any] = {"window_size": int(z["window_size"]), "stride": int(z["stride"]),
                           "model": str(z["model"]), "mission": str(z["mission"]), "ts_sha1": str(z["ts_sha1"])}
    for name in PARTS:
        if f"{name}_scores" in z.files:
            out[name] = (np.asarray(z[f"{name}_scores"], dtype=np.float64), np.asarray(z[f"{name}_end"], dtype=np.int64))
    return out
