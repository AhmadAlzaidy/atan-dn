"""Multi-seed runner: frozen hyperparameters, honest evaluation, mean/std over seeds.

For every seed the model is trained with the hyperparameters of the best-params JSON (no new search); the
validation block is carved from the TRAINING split, the threshold is the F1-optimal grid value on that block, the
normalizer and mask parameters come from the training split, and the test split is scored once (shared pipeline
``atandn.evaluation.pipeline.evaluate_mission``, the same code as ``scripts/evaluate.py``; point-adjustment supported).
Without ``--best-params`` the run is refused unless ``--allow-legacy-defaults`` is given (rows then carry
``used_legacy_defaults: true``). Reported: precision, recall, FPR, FNR, F1 and F0.5
at window level, point level (rule A), and point-adjusted event level (PA matching ESA-ADB Supp Table 9),
plus the optional official event-wise metrics.

Outputs in ``--out-dir``: ``seed_<seed>.json`` (one per finished seed; the run is RESUMABLE, finished seeds are
skipped unless ``--force``), ``multi_seed_runs.csv``, ``multi_seed_summary.json`` (``meta``, ``rows``, ``summary``)
and ``multi_seed_summary.csv`` (mean/std table). The last three are rebuilt from the seed files on every call.

Example::

    python experiments/multi_seed/run_multi_seed.py --config configs/m1.yaml --best-params best_m1.json \
        --data-root <data_dir> --out-dir results/multi_seed_m1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from atandn.config import MissionConfig, load_config  # noqa: E402
from atandn.data import MissionData  # noqa: E402
from atandn.utils.io import save_json  # noqa: E402
from atandn.training.search import LegacyDefaultsError, check_legacy_defaults  # noqa: E402
from atandn.utils.logging import setup_logging  # noqa: E402
from experiments.protocol import (METRIC_KEYS, PAPER_SEEDS, aggregate, load_best_params,  # noqa: E402
                                  load_mission_data, strip_private, train_and_evaluate, write_rows_csv)

log = logging.getLogger("experiments.multi_seed")


def resolve_seeds(seeds: Sequence[int] | None, n_seeds: int | None) -> list[int]:
    """Seed list: explicit ``seeds``; else the first ``n_seeds`` of the paper seeds (100, 777, 2026), extended with
    ``2027, 2028, ...`` when ``n_seeds`` exceeds three; default is the three paper seeds."""
    if seeds:
        out = [int(s) for s in seeds]
    else:
        n = n_seeds if n_seeds is not None else len(PAPER_SEEDS)
        if n < 1:
            raise ValueError("n_seeds must be >= 1")
        out = list(PAPER_SEEDS[:n]) + [PAPER_SEEDS[-1] + k for k in range(1, n - len(PAPER_SEEDS) + 1)]
    if len(set(out)) != len(out):
        raise ValueError("seeds must be distinct")
    return out


def _seed_path(out_dir: Path, seed: int) -> Path:
    return out_dir / f"seed_{seed}.json"


def run_multi_seed(cfg: MissionConfig, train: MissionData, test: MissionData, seeds: Sequence[int],
                   out_dir: str | Path, device: str = "auto", event_wise: bool = False,
                   event_kwargs: dict[str, Any] | None = None, force: bool = False, ddof: int = 1,
                   meta: dict[str, Any] | None = None, used_legacy_defaults: bool = False) -> list[dict[str, Any]]:
    """Run (or resume) all seeds and write the per-seed files plus the aggregate CSV/JSON.

    Args:
        cfg: mission config with the frozen hyperparameters.
        seeds: seeds to run; seeds with an existing ``seed_<seed>.json`` are reused unless ``force``.
        force: retrain every seed and overwrite its file.
        ddof: delta degrees of freedom of the reported std.
        used_legacy_defaults: written into every row and the meta (True only with ``--allow-legacy-defaults``).

    Returns:
        one result row per seed (in ``seeds`` order).
    """
    out = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        path = _seed_path(out, seed)
        if path.exists() and not force:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
            log.info("seed %d: reusing %s", seed, path)
            continue
        log.info("seed %d: training", seed)
        row = strip_private(train_and_evaluate(cfg, train, test, seed, device, event_wise, event_kwargs,
                                              report_extra={"used_legacy_defaults": used_legacy_defaults}))
        save_json(row, path)
        rows.append(row)
    summary = aggregate(rows, None, ddof).get("all", {})
    table = [{"scope": scope, "metric": k, **summary[f"{scope}_{k}"]} for scope in ("window", "point", "pa")
             for k in METRIC_KEYS if f"{scope}_{k}" in summary]
    info = {"mission": cfg.name, "seeds": [int(s) for s in seeds], "std_ddof": ddof,
            "protocol": "validation block carved from the TRAINING split; threshold = F1-optimal grid on that block; normalizer from the fitting part; mask/spike filter from the training split; test scored once; point-adjustment supported (matching ESA-ADB Supp Table 9); shared pipeline atandn.evaluation.pipeline",
            "used_legacy_defaults": used_legacy_defaults,
            "config": cfg.to_dict(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {})}
    write_rows_csv(rows, out / "multi_seed_runs.csv")
    write_rows_csv(table, out / "multi_seed_summary.csv")
    save_json({"meta": info, "rows": rows, "summary": summary}, out / "multi_seed_summary.json")
    log.info("wrote aggregate files to %s", out)
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="mission YAML (configs/m1.yaml, m2.yaml, ops_sat.yaml)")
    p.add_argument("--best-params", help="best-params JSON with the frozen hyperparameters")
    p.add_argument("--allow-legacy-defaults", action="store_true",
                   help="run without --best-params using predefined mission defaults")
    p.add_argument("--data-root", required=True, help="folder with the prepared mission data")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", nargs="+", type=int, help=f"explicit seeds (default {list(PAPER_SEEDS)})")
    p.add_argument("--n-seeds", type=int, help="number of seeds: first N of the paper seeds, extended by 2027, 2028, ...")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs (smoke runs; recorded in meta)")
    p.add_argument("--event-wise", action="store_true", help="also compute official ESA-ADB event-wise metrics")
    p.add_argument("--esa-module", default="esa_adb")
    p.add_argument("--esa-attribute", default="evaluate")
    p.add_argument("--ddof", type=int, default=1, help="ddof of the reported std (default 1, sample std)")
    p.add_argument("--force", action="store_true", help="retrain seeds that already have a result file")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    try:
        legacy = check_legacy_defaults(bool(args.best_params), args.allow_legacy_defaults, Path(__file__).name)
    except LegacyDefaultsError as exc:
        log.error("%s", exc)
        return 2
    if args.best_params:
        cfg = load_best_params(args.best_params, cfg)
    if args.max_epochs is not None:
        cfg.train.max_epochs = args.max_epochs
        cfg.validate()
    seeds = resolve_seeds(args.seeds, args.n_seeds)
    train, test = load_mission_data(cfg, args.data_root)
    meta = {"best_params": args.best_params, "used_legacy_defaults": legacy, "max_epochs_override": args.max_epochs, "event_wise": args.event_wise}
    run_multi_seed(cfg, train, test, seeds, args.out_dir, args.device, args.event_wise,
                   {"module": args.esa_module, "attribute": args.esa_attribute}, args.force, args.ddof, meta, legacy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
