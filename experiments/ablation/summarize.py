"""Format ablation results as a LaTeX (booktabs) table and a Markdown table.

The script only formats what is stored in a results file written by ``run_ablation.py`` (``*_results.json``, the
``summary`` block is used; or ``*_results.csv``, from which mean and std over seeds are recomputed). It does not
train anything and never fills a missing value: a metric that does not exist for a variant is shown as ``--``.

Example::

    python experiments/ablation/summarize.py results/ablation_norm_m1/ablation_norm_results.json \
        --metrics window_f1 point_f1 kurtosis_norm fisher_latent --out-prefix results/ablation_norm_m1/table
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atandn.utils.logging import setup_logging  # noqa: E402
from experiments.protocol import aggregate, read_rows_csv  # noqa: E402

log = logging.getLogger("experiments.summarize")

DEFAULT_METRICS = ("window_precision", "window_recall", "window_fpr", "window_f1", "window_f0_5",
                   "pa_precision", "pa_recall", "pa_fpr", "pa_f1", "pa_f0_5",
                   "point_f1", "kurtosis_norm", "fisher_latent", "n_params")
_LABELS = {"precision": "P", "recall": "R", "fpr": "FPR", "fnr": "FNR", "f1": "F1", "f0_5": "F0.5"}


def column_label(col: str) -> str:
    """Readable header for a result column (``window_f1`` -> ``Window F1``)."""
    for prefix, text in (("window_", "Window "), ("point_", "Point "), ("event_", "Event "), ("pa_", "PA ")):
        if col.startswith(prefix):
            rest = col[len(prefix):]
            return text + _LABELS.get(rest, rest.replace("_", " "))
    return {"kurtosis_norm": "Kurt$_{ex}$", "fisher_latent": "Fisher $J$", "n_params": "Params"}.get(col, col.replace("_", " "))


def load_summary(path: str | Path, ddof: int = 1) -> tuple[list[str], dict[str, dict[str, dict]], dict[str, str]]:
    """Return ``(variant order, summary, descriptions)`` from a results JSON or CSV."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        order = list(meta.get("variants") or data["summary"].keys())
        return order, data["summary"], dict(meta.get("descriptions", {}))
    rows = read_rows_csv(p)
    order: list[str] = []
    for r in rows:
        if r.get("variant") not in order:
            order.append(str(r["variant"]))
    return order, aggregate(rows, "variant", ddof), {}


def _cell(stats: dict | None, decimals: int, md: bool) -> str:
    if not stats:
        return "--"
    if stats["n"] <= 1 and stats["std"] == 0.0:
        return f"{stats['mean']:.{decimals}f}"
    pm = "±" if md else r"$\pm$"
    return f"{stats['mean']:.{decimals}f} {pm} {stats['std']:.{decimals}f}"


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def to_markdown(order: Sequence[str], summary: dict, metrics: Sequence[str], decimals: int = 3) -> str:
    """Markdown table, one row per variant, ``mean ± std`` cells."""
    head = ["Variant"] + [column_label(m).replace("$", "").replace("_{ex}", "_ex") for m in metrics] + ["n seeds"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for v in order:
        st = summary.get(v, {})
        n = max([s["n"] for s in st.values()] or [0])
        lines.append("| " + " | ".join([v] + [_cell(st.get(m), 0 if m == "n_params" else decimals, True) for m in metrics] + [str(n)]) + " |")
    return "\n".join(lines) + "\n"


def to_latex(order: Sequence[str], summary: dict, metrics: Sequence[str], decimals: int = 3, caption: str = "",
             label: str = "tab:ablation") -> str:
    """LaTeX ``table`` with booktabs rules (requires ``\\usepackage{booktabs}``)."""
    cols = "l" + "c" * len(metrics)
    lines = [r"\begin{table}[t]", r"\centering", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(["Variant"] + [column_label(m) for m in metrics]) + r" \\", r"\midrule"]
    for v in order:
        st = summary.get(v, {})
        cells = [_cell(st.get(m), 0 if m == "n_params" else decimals, False) for m in metrics]
        lines.append(" & ".join([_tex_escape(v)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    lines += [rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def to_table6_latex(summary: dict, decimals: int = 3, caption: str = "Ablation study on M1 test telemetry across 3 random seeds (100, 777, 2026).",
                    label: str = "tab:ablation_table6") -> str:
    """Format Table 6 matching the manuscript layout: Component vs Configuration."""
    components = [
        ("Normalization", [("BIGN (Full Model)", "full"), ("Standard Normalization", "standard_norm")]),
        ("Attention Mechanism", [("Multi-Head Attention", "full"), ("Convolutional Feed-Forward", "conv_feedforward")]),
        ("Dual-Objective", [("Dual-Head", "full"), ("Classifier-Only", "classifier_only")]),
        ("Post-Processing", [("Stage 3 Masking", "full"), ("Raw Alarms (No Filtering)", "raw_alarms")]),
    ]
    metrics = ["pa_precision", "pa_recall", "pa_fpr", "pa_f1", "pa_f0_5"]
    cols = "llccccc"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        r"Component & Configuration & Precision & Recall & FPR & F1 & F0.5 \\",
        r"\midrule",
    ]
    for comp_idx, (comp_name, variants) in enumerate(components):
        for v_idx, (cfg_name, v_key) in enumerate(variants):
            c_cell = comp_name if v_idx == 0 else ""
            st = summary.get(v_key, {})
            cells = []
            for m in metrics:
                s = st.get(m)
                if s is None and m.startswith("pa_"):
                    s = st.get("window_" + m[3:])
                cells.append(_cell(s, decimals, False))
            lines.append(f"{c_cell} & {cfg_name} & " + " & ".join(cells) + r" \\")
        if comp_idx < len(components) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    lines += [rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="*_results.json or *_results.csv from run_ablation.py")
    p.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS), help="result columns to show")
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--caption", default="")
    p.add_argument("--label", default="tab:ablation")
    p.add_argument("--ddof", type=int, default=1, help="ddof when aggregating a CSV")
    p.add_argument("--out-prefix", help="write <prefix>.tex and <prefix>.md (otherwise print both to stdout)")
    p.add_argument("--force", action="store_true", help="overwrite existing output files")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    setup_logging()
    order, summary, _ = load_summary(args.results, args.ddof)
    tex = to_latex(order, summary, args.metrics, args.decimals, args.caption, args.label)
    md = to_markdown(order, summary, args.metrics, args.decimals)
    if args.out_prefix:
        targets = [Path(args.out_prefix + ".tex"), Path(args.out_prefix + ".md")]
        clash = [t for t in targets if t.exists()]
        if clash and not args.force:
            log.error("refusing to overwrite %s (use --force)", ", ".join(map(str, clash)))
            return 1
        targets[0].parent.mkdir(parents=True, exist_ok=True)
        targets[0].write_text(tex, encoding="utf-8")
        targets[1].write_text(md, encoding="utf-8")
        log.info("wrote %s and %s", targets[0], targets[1])
    else:
        sys.stdout.write(md + "\n" + tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
