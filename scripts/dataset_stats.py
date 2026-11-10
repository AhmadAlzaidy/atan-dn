"""Dataset statistics of one mission: channels, date range, sampling step, split sizes, anomalies, events, windows.

Parts: ``full`` (whole stream), ``train`` / ``test`` (chronological split at ``data.train_ratio``) and, inside the
training split, ``fit`` / ``val`` (the event-aware validation block of ``prepare_protocol_data``). For each part:
samples, anomalous samples and percent, number of anomalous EVENTS (``count_events``: contiguous runs of positive
samples), and the number of sliding windows (window ``model.window_size`` at ``data.test_stride``) with their
positive windows. Only descriptive counts; nothing is fitted. Writes ``<out-prefix>.json`` and ``<out-prefix>.md``.

Exit codes: 0 ok, 1 runtime failure, 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from _common import add_common_args, load_mission, load_run_config, prepare_cfg_for_data, resolve_paths, run_cli

log = logging.getLogger("atandn.scripts.dataset_stats")


def _part_stats(labels, window: int, stride: int) -> dict[str, Any]:
    import numpy as np

    from atandn.data import count_events, window_labels, window_starts

    y = np.asarray(labels)
    n = int(len(y))
    starts = window_starts(n, window, stride)
    wl = window_labels(y, starts, window) if len(starts) else np.zeros(0, dtype=np.int64)
    pos = int((y > 0).sum())
    return {"samples": n, "anomalous_samples": pos, "anomaly_percent": float(100.0 * pos / n) if n else 0.0,
            "events": int(count_events(y)), "windows": int(len(starts)), "positive_windows": int(wl.sum()),
            "positive_window_percent": float(100.0 * wl.sum() / len(wl)) if len(wl) else 0.0}


def compute_stats(cfg, mission) -> dict[str, Any]:
    """Statistics of ``mission`` (a :class:`MissionData`) under the window/split settings of ``cfg``."""
    import numpy as np
    import pandas as pd

    from atandn.data.splits import to_naive_utc
    from atandn.evaluation.pipeline import prepare_protocol_data

    w, stride = int(cfg.model.window_size), int(cfg.data.test_stride)
    ts = to_naive_utc(mission.timestamps)
    steps = np.diff(ts.values.astype("datetime64[ns]").astype(np.int64)) / 1e9 if len(ts) > 1 else np.zeros(0)
    train, test = mission.split(cfg.data.train_ratio)
    data = prepare_protocol_data(train, test, cfg)
    sp = data.split
    parts = {
        "full": _part_stats(mission.labels, w, stride),
        "train": _part_stats(train.labels, w, stride),
        "fit": _part_stats(train.labels[sp.fit], w, stride),
        "val": _part_stats(train.labels[sp.val], w, stride),
        "test": _part_stats(test.labels, w, stride),
    }
    return {
        "mission": cfg.name, "channels": int(mission.values.shape[1]), "channel_names": list(mission.channel_names),
        "first_timestamp": str(ts[0]) if len(ts) else None, "last_timestamp": str(ts[-1]) if len(ts) else None,
        "duration_days": float((ts[-1] - ts[0]) / pd.Timedelta(days=1)) if len(ts) > 1 else 0.0,
        "sampling_step_seconds": {"median": float(np.median(steps)) if len(steps) else None,
                                  "min": float(steps.min()) if len(steps) else None,
                                  "max": float(steps.max()) if len(steps) else None},
        "window_size": w, "test_stride": stride, "train_ratio": float(cfg.data.train_ratio),
        "validation_split": {"fit": [int(sp.fit.start), int(sp.fit.stop)], "val": [int(sp.val.start), int(sp.val.stop)],
                             "report": sp.report},
        "known_event_intervals": int(len(mission.meta.get("known_event_intervals", []))),
        "parts": parts,
    }


def to_markdown(st: dict[str, Any]) -> str:
    """Markdown rendering of :func:`compute_stats`."""
    step = st["sampling_step_seconds"]
    lines = [f"# Dataset statistics: {st['mission']}", "",
             f"* channels: {st['channels']}", f"* range: {st['first_timestamp']} .. {st['last_timestamp']} "
             f"({st['duration_days']:.1f} days)",
             f"* sampling step (s): median {step['median']}, min {step['min']}, max {step['max']}",
             f"* window {st['window_size']} at stride {st['test_stride']}, train ratio {st['train_ratio']}",
             f"* known planned-event intervals: {st['known_event_intervals']}", "",
             "| Part | Samples | Anomalous samples | Anomaly % | Events | Windows | Positive windows |",
             "|---|---|---|---|---|---|---|"]
    for name, p in st["parts"].items():
        lines.append(f"| {name} | {p['samples']} | {p['anomalous_samples']} | {p['anomaly_percent']:.2f} | {p['events']} | "
                     f"{p['windows']} | {p['positive_windows']} |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``dataset_stats.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--out-prefix", type=Path, default=None, help="output prefix (default: <out-dir>/dataset_stats)")
    p.add_argument("--window-size", type=int, default=None, help="window length for the window counts (default: config)")
    return p


def run(args: argparse.Namespace) -> None:
    """Load the mission, compute the statistics, write JSON and Markdown."""
    cfg, _, paths_cfg = load_run_config(args.config)
    if args.window_size is not None:
        cfg.model.window_size = int(args.window_size)
    rp = resolve_paths(args, cfg, paths_cfg)
    mission = load_mission(cfg, rp.data_root)
    prepare_cfg_for_data(cfg, mission.values.shape[1])
    st = compute_stats(cfg, mission)
    pre = args.out_prefix or (rp.out_dir / "dataset_stats")
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.with_suffix(".json").write_text(json.dumps(st, indent=2), encoding="utf-8")
    pre.with_suffix(".md").write_text(to_markdown(st), encoding="utf-8")
    log.info("wrote %s.{json,md}", pre)
    for name, p in st["parts"].items():
        log.info("%-5s samples %9d | anomalous %.2f %% | events %4d | windows %7d", name, p["samples"],
                 p["anomaly_percent"], p["events"], p["windows"])


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
