"""Shared helpers of the model-report scripts (bign_report, attention_report, inverse_report, model_reports)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from _common import ScriptError, load_mission, load_run_config

log = logging.getLogger("atandn.scripts.reports")


def add_report_args(p: argparse.ArgumentParser, need_data: bool = False) -> None:
    """Add ``--checkpoint``, ``--config``, ``--data-root``, ``--device`` and ``--log-level``."""
    p.add_argument("--checkpoint", required=True, type=Path, help="model.pt written by scripts/train.py")
    p.add_argument("--config", required=True, type=Path, help="mission YAML (gives data.path)")
    p.add_argument("--data-root", type=Path, default=None,
                   help="folder data.path is relative to; " + ("required here" if need_data else
                   "if omitted, only the weight-based numbers are computed (no test windows)"))
    p.add_argument("--device", default="cpu", help="'cpu', 'cuda' or 'auto' (default: cpu)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def load_model_and_test_windows(args: argparse.Namespace, need_data: bool = False) -> tuple[Any, Any, Any]:
    """Load the checkpoint and, if ``--data-root`` is given, the TEST windows of its mission.

    The chronological split uses the checkpoint train_ratio, the windows its window_size and test_stride (z-scored
    per window). Only the test split is returned, for descriptive statistics.

    Returns:
        ``(model, cfg, X)`` with ``X`` an ``(N, C, T)`` float32 array or None.
    """
    from atandn.data.windows import make_test_windows
    from atandn.training import load_checkpoint

    if not Path(args.checkpoint).is_file():
        raise ScriptError(f"checkpoint not found: {args.checkpoint}")
    model, cfg, _extra = load_checkpoint(args.checkpoint, args.device)
    if args.data_root is None:
        if need_data:
            raise ScriptError("--data-root is required for this report", 2)
        return model, cfg, None
    run_cfg, _, _paths = load_run_config(args.config)
    mission = load_mission(run_cfg, Path(args.data_root))
    if mission.values.shape[1] != cfg.model.in_channels:
        raise ScriptError(f"data has {mission.values.shape[1]} channels, checkpoint expects {cfg.model.in_channels}", 2)
    _train, test = mission.split(cfg.data.train_ratio)
    w = make_test_windows(test.values, test.labels, test.timestamps, cfg.model.window_size, cfg.data.test_stride)
    log.info("test windows: %d (window %d, stride %d)", len(w["X"]), cfg.model.window_size, cfg.data.test_stride)
    return model, cfg, w["X"]
