"""Run the BIGN, attention and inverse reports on the TEST windows of a checkpoint mission.

Writes bign_report.json, attention_report.json, inverse_report.json and model_reports.md into --out-dir. The numbers
are descriptive statistics of the trained model; no test label is used and nothing is fitted on the test split.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from _common import run_cli
from _reports import add_report_args, load_model_and_test_windows

log = logging.getLogger("atandn.scripts.model_reports")


def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def _table(rows: list, cols: list) -> list:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return out


def markdown_summary(bign: list, att: list, inv: dict, n_windows: int) -> str:
    """Markdown summary of the three reports."""
    lines = ["# Model reports (descriptive, test windows only)", "", f"Test windows: {n_windows}.", "",
             "## BIGN (|y_i| <= gamma_ii^-1/2)", ""]
    lines += _table(bign, ["layer", "C", "gamma_ii_min", "gamma_ii_median", "inv_sqrt_max", "R", "beta_min", "offdiag_max",
                           "rowsum_max", "empirical_max_abs_y", "bound_holds"])
    lines += ["", "## Attention (Lambda of Prop. attn)", ""]
    lines += _table(att, ["layer", "C", "H", "spec_WQ", "spec_WK", "prod", "bQ", "bK", "R", "Lambda", "Lambda_with_pe"])
    lines += ["", f"## Exact inverse (kappa bound 2/eps = {_fmt(inv.get('kappa_bound'))})", ""]
    lines += _table(inv.get("layers", []), ["layer", "C", "clamp_rate", "kappa_p50", "kappa_p99", "kappa_max",
                                            "solve_ms_per_window", "kappa_bound"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``model_reports.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_report_args(p, need_data=True)
    p.add_argument("--out-dir", required=True, type=Path, help="folder for the three JSON files and the Markdown summary")
    p.add_argument("--max-pairs", type=int, default=20000)
    return p


def run(args: argparse.Namespace) -> None:
    """Compute the three reports and write them."""
    from atandn.evaluation.attention_report import attention_norms
    from atandn.evaluation.bign_report import bign_stats
    from atandn.evaluation.inverse_report import inverse_report
    from atandn.utils.io import save_json

    model, _cfg, X = load_model_and_test_windows(args, need_data=True)
    bign, att, inv = bign_stats(model, X), attention_norms(model), inverse_report(model, X, max_pairs=args.max_pairs)
    d = Path(args.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    save_json(bign, d / "bign_report.json")
    save_json(att, d / "attention_report.json")
    save_json(inv, d / "inverse_report.json")
    (d / "model_reports.md").write_text(markdown_summary(bign, att, inv, len(X)), encoding="utf-8")
    log.info("wrote reports to %s", d)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
