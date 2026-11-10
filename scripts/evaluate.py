"""Evaluate a trained checkpoint on the TEST split (scored once), with the shared protocol.

All protocol values come from the checkpoint written by train.py (frozen on the TRAINING split): window size,
test stride, train ratio, reconstruction-error normalizer (1st/99th percentile of the fitting-part errors),
training-derived mask parameters and the F1-optimal threshold of the VALIDATION block (a block of the training
split). The test split is scored once with those frozen values, then the known-event mask (buffer after each
known event) and the spike filter are applied; test labels are used only afterwards for metrics (window-level,
point-wise rule A without point-adjustment and, with --esa-adb, the official ESA-ADB event-wise score if its package
is installed). The code is atandn.evaluation.pipeline.evaluate_mission, the same function the experiment runners use.

If the YAML disagrees with the checkpoint (train_ratio, test_stride, eval settings) the run is refused unless
--override is given; with --override the YAML values are used and the protocol is re-fitted from the training split.
If train.py could not freeze the protocol (e.g. no positive window in the validation block) the protocol is re-fitted
here with the YAML's eval section; set eval.allow_train_threshold_fallback: true in the YAML or pass
--allow-train-threshold-fallback to fit the threshold on the fitting part instead (flagged threshold_source).

Writes a JSON report (default <out-dir>/report.json).
Exit codes: 0 ok, 1 runtime failure, 2 invalid config/arguments.
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Sequence

from _common import ScriptError, add_common_args, load_mission, load_run_config, resolve_paths, run_cli

log = logging.getLogger("atandn.scripts.evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``evaluate.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, with_device=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="model.pt from train.py (default: <out-dir>/model.pt)")
    p.add_argument("--override", action="store_true",
                   help="allow the YAML to disagree with the checkpoint: the YAML data/eval settings are used and the "
                        "protocol is re-fitted from the training split (default: refuse)")
    p.add_argument("--allow-train-threshold-fallback", action="store_true",
                   help="when the protocol has to be re-fitted and the validation block has no positive window, fit the "
                        "threshold on the fitting part (flagged threshold_source='train_fallback'); same as the YAML "
                        "key eval.allow_train_threshold_fallback: true")
    p.add_argument("--report", type=Path, default=None, help="output JSON (default: <out-dir>/report.json)")
    p.add_argument("--report-unmasked", action="store_true",
                   help="also report the metrics WITHOUT Stage 3 (no known-event mask, no spike filter) next to the ones "
                        "with Stage 3 (report key stage3_comparison); the same frozen threshold is used, it is NOT "
                        "re-tuned for the unmasked chain")
    p.add_argument("--esa-adb", action="store_true",
                   help="also compute the official ESA-ADB event-wise score (needs the official package; "
                        "otherwise a note is logged and the score is reported as not computed)")
    return p


def run(args: argparse.Namespace) -> None:
    """Score the test split once with the frozen protocol and write the report."""
    from atandn.evaluation.pipeline import evaluate_mission, prepare_protocol_data
    from atandn.training import checkpoint_config_conflicts, load_checkpoint
    from atandn.utils.io import save_json

    run_cfg, _, paths_cfg = load_run_config(args.config)
    rp = resolve_paths(args, run_cfg, paths_cfg)
    ckpt = args.checkpoint or rp.checkpoint
    if not Path(ckpt).is_file():
        raise ScriptError(f"checkpoint not found: {ckpt} (run train.py first)")
    model, cfg, extra = load_checkpoint(ckpt, args.device)
    frozen = extra.get("frozen")

    conflicts = checkpoint_config_conflicts(run_cfg, cfg, frozen)
    if cfg.model.window_size != run_cfg.model.window_size:
        log.info("model.window_size: config=%d, checkpoint=%d (the checkpoint's value, from tuning, is used)",
                 run_cfg.model.window_size, cfg.model.window_size)
    if conflicts:
        msg = "the config disagrees with the checkpoint: " + "; ".join(conflicts)
        if not args.override:
            raise ScriptError(msg + ". Use the checkpoint's settings (fix the YAML) or pass --override.", 2)
        log.warning("%s -- --override given: using the YAML values and re-fitting the protocol from the training "
                    "split", msg)
        cfg = copy.deepcopy(cfg)
        cfg.data, cfg.eval = copy.deepcopy(run_cfg.data), copy.deepcopy(run_cfg.eval)
        frozen = None
    if frozen is None:
        log.warning("no frozen protocol in the checkpoint%s: fitting it from the training split now with the YAML's "
                    "eval section", f" ({extra['frozen_error']})" if extra.get("frozen_error") else "")
        cfg = copy.deepcopy(cfg)
        cfg.eval = copy.deepcopy(run_cfg.eval)
        if args.allow_train_threshold_fallback:
            cfg.eval.allow_train_threshold_fallback = True
    elif args.allow_train_threshold_fallback:
        log.info("--allow-train-threshold-fallback ignored: the checkpoint holds a frozen protocol (no refit)")

    mission = load_mission(run_cfg, rp.data_root)
    if mission.values.shape[1] != cfg.model.in_channels:
        raise ScriptError(f"data has {mission.values.shape[1]} channels, checkpoint expects {cfg.model.in_channels}", 2)
    train, test = mission.split(cfg.data.train_ratio)
    data = prepare_protocol_data(train, test, cfg)
    report = evaluate_mission(model, data, cfg, frozen=frozen, device=args.device, esa_adb=args.esa_adb,
                              report_unmasked=args.report_unmasked,
                              report_extra={"checkpoint": Path(ckpt).name,
                                            "used_legacy_defaults": bool(extra.get("used_legacy_defaults", False))})
    out = save_json(report, args.report or rp.report)
    log.info("window F1 %.4f | point-wise F1 %.4f | wrote %s", report["window_metrics"]["f1"],
             report["pointwise_metrics_rule_A"]["f1"], out)
    if "stage3_comparison" in report:
        c = report["stage3_comparison"]
        log.info("window F1 with Stage 3 %.4f, without Stage 3 %.4f (same frozen threshold)",
                 c["with_stage3"]["window_metrics"]["f1"], c["without_stage3"]["window_metrics"]["f1"])


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
