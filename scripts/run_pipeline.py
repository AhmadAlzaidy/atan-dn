"""Run prepare -> tune -> train -> evaluate for one mission; resumable.

A step is skipped when its output already exists (data file, best_params.json, model.pt, report.json) unless
--force is given. Use --from-step / --to-step to run a part of the pipeline.
Exit codes: 0 ok, otherwise the exit code of the failing step.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import add_common_args, load_run_config, resolve_paths, run_cli

log = logging.getLogger("atandn.scripts.run_pipeline")
STEPS = ("prepare", "tune", "train", "evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``run_pipeline.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, with_device=True)
    p.add_argument("--raw-dir", type=Path, default=None, help="raw ESA-ADB folder for the prepare step")
    p.add_argument("--n-trials", type=int, default=None, help="TPE trials (default: search.n_trials)")
    p.add_argument("--max-epochs", type=int, default=None, help="epoch cap for tuning and training")
    p.add_argument("--from-step", choices=STEPS, default="prepare", help="first step to run (default: prepare)")
    p.add_argument("--to-step", choices=STEPS, default="evaluate", help="last step to run (default: evaluate)")
    p.add_argument("--force", action="store_true", help="re-run steps whose outputs already exist")
    p.add_argument("--esa-adb", action="store_true", help="forwarded to evaluate.py")
    return p


def run(args: argparse.Namespace) -> None:
    """Execute the selected steps in order."""
    import evaluate
    import prepare_data
    import train
    import tune

    cfg, _, paths_cfg = load_run_config(args.config)
    rp = resolve_paths(args, cfg, paths_cfg)
    base = ["--config", str(args.config), "--data-root", str(rp.data_root), "--out-dir", str(rp.out_dir),
            "--log-level", args.log_level]
    dev = ["--device", args.device]
    ep = ["--max-epochs", str(args.max_epochs)] if args.max_epochs is not None else []
    plan = {
        "prepare": (rp.data_root / cfg.data.path, prepare_data.main,
                    base + (["--raw-dir", str(args.raw_dir)] if args.raw_dir else [])),
        "tune": (rp.best_params, tune.main,
                 base + dev + ep + (["--n-trials", str(args.n_trials)] if args.n_trials is not None else [])),
        "train": (rp.checkpoint, train.main, base + dev + ep),
        "evaluate": (rp.report, evaluate.main, base + dev + (["--esa-adb"] if args.esa_adb else [])),
    }
    first, last = STEPS.index(args.from_step), STEPS.index(args.to_step)
    for name in STEPS[first:last + 1]:
        output, fn, argv = plan[name]
        if output.is_file() and not args.force and not (name == "prepare" and cfg.data.kind == "opssat"):
            log.info("[%s] skipped: %s exists (use --force to redo)", name, output)
            continue
        log.info("[%s] running", name)
        code = fn(argv + (["--force"] if name == "prepare" and args.force else []))
        if code != 0:
            raise SystemExit(code)
    log.info("pipeline finished; artefacts in %s", rp.out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    try:
        return run_cli(build_parser, run, argv)
    except SystemExit as exc:   # a failing step
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)


if __name__ == "__main__":
    raise SystemExit(main())
