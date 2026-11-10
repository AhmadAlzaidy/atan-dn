"""Inverse-BIGN report of a trained checkpoint on TEST windows: clamp rate, condition number (p50/p99/max) after
clamping, solve-only time per window and the a-priori bound 2/eps. Needs --data-root."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from _common import run_cli
from _reports import add_report_args, load_model_and_test_windows

log = logging.getLogger("atandn.scripts.inverse_report")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``inverse_report.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_report_args(p, need_data=True)
    p.add_argument("--out", type=Path, default=Path("inverse_report.json"), help="output JSON")
    p.add_argument("--max-pairs", type=int, default=20000, help="(window, time) pairs sampled for kappa (default 20000)")
    return p


def run(args: argparse.Namespace) -> None:
    """Compute and write the report."""
    from atandn.evaluation.inverse_report import inverse_report
    from atandn.utils.io import save_json

    model, _cfg, X = load_model_and_test_windows(args, need_data=True)
    out = save_json(inverse_report(model, X, max_pairs=args.max_pairs), args.out)
    log.info("wrote %s", out)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
