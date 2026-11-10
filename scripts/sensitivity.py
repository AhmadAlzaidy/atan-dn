"""Hyperparameter sensitivity table (paper Table 3 style) from the stored TPE trials.

Mode 1 (default, cheap): ``--trials trials.csv`` (written by ``scripts/tune.py``; one file per mission). For every
parameter in ``--params`` the COMPLETED trials with the lowest, the best-scoring and the highest value of that parameter
are reported, with the trial value = validation average precision of the event-aware validation block of the TRAINING
split (the objective of the search; never the test F1). Caveat: in a TPE study all parameters vary at once, so a row
shows the score of a trial that happens to have an extreme value of one parameter, not a controlled change; when the
lowest/highest value is shared by several trials the best-scoring of them is shown. The "best" row is the best trial of
the study, whatever its parameter value.

Mode 2 (``--retrain``, expensive): one-at-a-time retraining with the other hyperparameters frozen at ``--best-params``.
Each parameter is set to the low and the high edge of its TPE search range (read from
``atandn.training.search.suggest_params``) and the model is trained with ``atandn.training.fit.fit_on_split`` (the call
inside ``experiments.protocol.train_and_evaluate``) on the fitting part of the training split; the score is the
validation-block metric of the selected epoch (``best_score``, average precision when the block has positives). The test
split is not scored at all. Cost: ``2 x len(params) x len(seeds)`` full trainings per mission (about ``6 x`` the cost of one
final training for the three default parameters and one seed) plus one training at the best values for reference.

Outputs (``--out-prefix``): ``<prefix>.json``, ``<prefix>.md`` and ``<prefix>.tex``. Exit codes: 0 ok, 1 runtime failure,
2 invalid input.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

from _common import ScriptError, run_cli

log = logging.getLogger("atandn.scripts.sensitivity")

DEFAULT_PARAMS = ("pos_weight", "window_size", "recon_weight")
_LABELS = {"pos_weight": "Pos. Weight", "window_size": "Window Size", "recon_weight": "Recon. Weight"}


# ---------------------------------------------------------------------------------------------------------------
# trial based sensitivity
# ---------------------------------------------------------------------------------------------------------------
def load_trials(path: str | Path):
    """Read ``trials.csv`` (``Study.trials_dataframe``) into a DataFrame with ``number``, ``value``, ``state``, ``params_*``."""
    import pandas as pd

    p = Path(path)
    if not p.is_file():
        raise ScriptError(f"trials file not found: {p}", 2)
    df = pd.read_csv(p)
    for col in ("number", "value", "state"):
        if col not in df.columns:
            raise ScriptError(f"{p} has no column '{col}' (expected the output of scripts/tune.py)", 2)
    return df


def sensitivity_rows(df, param: str) -> list[dict[str, Any]]:
    """Low / best / high rows of one parameter among the completed trials.

    Returns:
        ``[{"role": "low"|"best"|"high", "param_value", "value", "trial", "n_completed"}, ...]`` (3 rows).

    Raises:
        ScriptError: the parameter has no column or no completed trial exists.
    """
    col = f"params_{param}"
    if col not in df.columns:
        raise ScriptError(f"trials have no parameter '{param}' (columns: "
                          f"{sorted(c[7:] for c in df.columns if c.startswith('params_'))})", 2)
    ok = df[(df["state"].astype(str) == "COMPLETE") & df["value"].notna() & df[col].notna()]
    if ok.empty:
        raise ScriptError(f"no completed trial with a value for '{param}'", 2)

    def row(role: str, r) -> dict[str, Any]:
        v = r[col]
        return {"role": role, "param_value": float(v), "value": float(r["value"]), "trial": int(r["number"]),
                "n_completed": int(len(ok))}

    lo = ok[ok[col] == ok[col].min()].sort_values("value", ascending=False, kind="stable").iloc[0]
    hi = ok[ok[col] == ok[col].max()].sort_values("value", ascending=False, kind="stable").iloc[0]
    best = ok.sort_values("value", ascending=False, kind="stable").iloc[0]
    return [row("low", lo), row("best", best), row("high", hi)]


def search_ranges() -> dict[str, tuple[float, float]]:
    """Low/high edge of every numeric parameter of the TPE search space (read from ``suggest_params``)."""
    from atandn.training.search import suggest_params

    class _Recorder:
        def __init__(self) -> None:
            self.ranges: dict[str, tuple[float, float]] = {}

        def suggest_int(self, name, low, high, step=1, log=False):
            self.ranges[name] = (low, high)
            return low

        def suggest_float(self, name, low, high, step=None, log=False):
            self.ranges[name] = (low, high)
            return low

        def suggest_categorical(self, name, choices):
            nums = [c for c in choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
            if nums:
                self.ranges[name] = (min(nums), max(nums))
            return choices[0]

    rec = _Recorder()
    suggest_params(rec)
    return rec.ranges


# ---------------------------------------------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------------------------------------------
def _fmt_param(name: str, v: float) -> str:
    return str(int(round(v))) if name in {"window_size", "batch_size", "num_heads", "conv_f1", "conv_f2"} else f"{v:.3g}"


def _cell(row: dict | None, name: str, md: bool, bold: bool) -> str:
    if row is None:
        return "--"
    val = row["value"]
    txt = f"{val:.4f}" if bold else f"{val:.2f}"
    if "std" in row and row.get("n", 1) > 1:
        txt += (" ± " if md else r" $\pm$ ") + f"{row['std']:.2f}"
    txt += f" ({_fmt_param(name, row['param_value'])})"
    if bold:
        txt = f"**{txt}**" if md else rf"\textbf{{{txt}}}"
    return txt


def to_markdown(table: dict[str, dict[str, list[dict]]], metric: str) -> str:
    """Markdown table: one block of three rows (L, best, H) per parameter, one column per mission."""
    missions = list(table)
    params = list(next(iter(table.values())))
    lines = [f"Sensitivity ({metric}); rows: lowest (L), best (bold), highest (H) value of the parameter, in brackets.", "",
             "| Parameter | Row | " + " | ".join(missions) + " |", "|---|---|" + "---|" * len(missions)]
    for p in params:
        for i, role in enumerate(("L", "best", "H")):
            cells = [_cell(table[m][p][i], p, True, role == "best") for m in missions]
            lines.append(f"| {_LABELS.get(p, p) if i == 0 else ''} | {role} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(table: dict[str, dict[str, list[dict]]], metric: str, caption: str = "", label: str = "tab:sensitivity") -> str:
    """LaTeX ``tabular`` in the layout of the paper's Table 3 (three rows per parameter, bold best row)."""
    missions = list(table)
    params = list(next(iter(table.values())))
    esc = lambda s: s.replace("_", r"\_").replace("%", r"\%")
    cap = caption or (f"Sensitivity of the main hyperparameters ({esc(metric)}). Rows report the completed trial with "
                      "the lowest (L), best and highest (H) value of the parameter (value in brackets).")
    out = [r"\begin{table}[!h]", r"\centering", rf"\caption{{{cap}}}", rf"\label{{{label}}}", r"\footnotesize",
           r"\renewcommand{\arraystretch}{1.2}", r"\begin{tabular}{l l " + "c" * len(missions) + "}", r"\toprule",
           " & ".join(["\\textbf{Parameter}", "\\textbf{Row}"] + [rf"\textbf{{{esc(m)}}}" for m in missions]) + r" \\",
           r"\midrule"]
    for k, p in enumerate(params):
        for i, role in enumerate(("L", "best", "H")):
            cells = [_cell(table[m][p][i], p, False, role == "best") for m in missions]
            name = rf"\multirow{{3}}{{*}}{{{esc(_LABELS.get(p, p))}}}" if i == 0 else ""
            out.append(" & ".join([name, role] + cells) + r" \\")
        if k < len(params) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# one-at-a-time retraining
# ---------------------------------------------------------------------------------------------------------------
def retrain_edges(cfg, train, params: Sequence[str], best_params: dict, seeds: Sequence[int],
                  device: str = "auto") -> dict[str, list[dict[str, Any]]]:
    """One-at-a-time retraining at the low/high search-range edges with the other parameters frozen.

    Returns ``{param: [low_row, best_row, high_row]}``; each row has ``param_value``, ``value`` (mean validation-block
    metric of the selected epoch over ``seeds``), ``std``, ``n`` and ``val_metric``. The test split is not touched.
    """
    import copy

    import numpy as np

    from atandn.evaluation.pipeline import prepare_protocol_data
    from atandn.training import apply_params
    from atandn.training.fit import fit_on_split
    from atandn.utils.seed import set_seed

    ranges = search_ranges()
    out: dict[str, list[dict[str, Any]]] = {}

    def train_once(p: dict, seed: int) -> tuple[float, str]:
        c = apply_params(cfg, p)
        c.train.seed = int(seed)
        set_seed(int(seed))
        data = prepare_protocol_data(train, None, c)
        _, fit_out = fit_on_split(c, train, data.split, device)
        return float(fit_out["best_score"]), str(fit_out["val_metric"])

    for name in params:
        if name not in ranges:
            raise ScriptError(f"'{name}' has no numeric search range; cannot retrain at its edges", 2)
        rows = []
        for role, v in (("low", ranges[name][0]), ("best", best_params[name]), ("high", ranges[name][1])):
            p = {**best_params, name: v}
            res = [train_once(p, s) for s in seeds]
            vals = np.array([r[0] for r in res])
            log.info("%s=%s (%s): %s %.4f", name, v, role, res[0][1], vals.mean())
            rows.append({"role": role, "param_value": float(v), "value": float(vals.mean()),
                         "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, "n": len(vals),
                         "val_metric": res[0][1]})
        out[name] = rows
    return out


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Argument parser of ``sensitivity.py``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", nargs="+", type=Path, default=[], help="trials.csv of tune.py (one per mission)")
    p.add_argument("--names", nargs="+", default=None, help="column names of the missions (default: folder names)")
    p.add_argument("--params", nargs="+", default=list(DEFAULT_PARAMS), help="parameters (default: %(default)s)")
    p.add_argument("--out-prefix", type=Path, default=Path("sensitivity"), help="output prefix (.json/.md/.tex)")
    p.add_argument("--retrain", action="store_true", help="one-at-a-time retraining at the range edges (expensive)")
    p.add_argument("--config", type=Path, default=None, help="mission YAML (retrain mode)")
    p.add_argument("--data-root", type=Path, default=None, help="data root (retrain mode)")
    p.add_argument("--best-params", type=Path, default=None, help="best_params.json to freeze (retrain mode)")
    p.add_argument("--seeds", nargs="+", type=int, default=[100], help="seeds of the retraining (default: 100)")
    p.add_argument("--device", default="auto")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def run(args: argparse.Namespace) -> None:
    """Build the table(s) and write the outputs."""
    table: dict[str, dict[str, list[dict]]] = {}
    metric = "validation average precision of the TPE trials"
    if args.retrain:
        from _common import load_run_config, resolve_paths
        from experiments.protocol import load_best_params

        from atandn.data import load_mission_from_config

        if not (args.config and args.best_params):
            raise ScriptError("--retrain needs --config and --best-params", 2)
        cfg, _, paths_cfg = load_run_config(args.config)
        cfg = load_best_params(args.best_params, cfg)
        best = json.loads(Path(args.best_params).read_text(encoding="utf-8"))
        best = best.get("params", best)
        rp = resolve_paths(argparse.Namespace(data_root=args.data_root, out_dir=None, raw_dir=None), cfg, paths_cfg)
        mission = load_mission_from_config(cfg, rp.data_root)
        cfg.model.in_channels = int(mission.values.shape[1])
        train, _unused_test = mission.split(cfg.data.train_ratio)
        del _unused_test
        table[cfg.name] = retrain_edges(cfg, train, args.params, best, args.seeds, args.device)
        metric = "validation-block score of one-at-a-time retraining (other parameters frozen)"
    else:
        if not args.trials:
            raise ScriptError("give --trials (or --retrain)", 2)
        names = args.names or [Path(t).resolve().parent.name for t in args.trials]
        if len(names) != len(args.trials):
            raise ScriptError("--names must have one entry per --trials file", 2)
        for name, path in zip(names, args.trials):
            df = load_trials(path)
            table[name] = {p: sensitivity_rows(df, p) for p in args.params}
    pre = Path(args.out_prefix)
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.with_suffix(".json").write_text(json.dumps({"metric": metric, "table": table}, indent=2), encoding="utf-8")
    pre.with_suffix(".md").write_text(to_markdown(table, metric), encoding="utf-8")
    pre.with_suffix(".tex").write_text(to_latex(table, metric), encoding="utf-8")
    log.info("wrote %s.{json,md,tex}", pre)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    return run_cli(build_parser, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
