"""Ablation runner: train every variant with the SAME frozen hyperparameters and evaluate honestly.

Per (variant, seed) the shared pipeline ``atandn.evaluation.pipeline.evaluate_mission`` is used (the same code as
``scripts/evaluate.py``, including the known-event mask and the spike filter): the validation block is carved from the
TRAINING split, the threshold is the F1-optimal grid value on it, the normalizer and the mask parameters are
derived from the training split FOR EACH trained model (every variant and seed has its own), the test split is scored
once, no point-adjustment. Without ``--best-params`` the run is refused unless ``--allow-legacy-defaults`` is given
(rows then carry ``used_legacy_defaults: true``). Output (in ``--out-dir``):

* ``<name>_results.csv``  one row per variant x seed
* ``<name>_results.json`` ``{"meta", "rows", "summary"}`` where ``summary[variant][column] = {mean, std, n}``
* ``runs/<variant>__seed<seed>.json`` one file per finished run (used by ``--resume``)

Existing results are never overwritten unless ``--force`` is given; ``--resume`` reuses finished runs.

Example::

    python experiments/ablation/run_ablation.py --config configs/m1.yaml --best-params best_m1.json \
        --data-root <data_dir> --study configs/ablation_norm.yaml --out-dir results/ablation_norm_m1
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
if str(_ROOT) not in sys.path:  # allow ``python experiments/ablation/run_ablation.py`` from anywhere
    sys.path.insert(0, str(_ROOT))

from atandn.config import MissionConfig, load_config  # noqa: E402
from atandn.data import MissionData  # noqa: E402
from atandn.utils.io import save_json  # noqa: E402
from atandn.training.search import LegacyDefaultsError, check_legacy_defaults  # noqa: E402
from atandn.utils.logging import setup_logging  # noqa: E402
from experiments.ablation.variants import VARIANTS, apply_variant, get_variant  # noqa: E402
from experiments.protocol import (PAPER_SEEDS, aggregate, load_best_params, load_mission_data,  # noqa: E402
                                  strip_private, train_and_evaluate, write_rows_csv)

log = logging.getLogger("experiments.ablation")


def load_study_file(path: str | Path) -> tuple[list[str], list[int]]:
    """Read ``variants`` and ``seeds`` from a study YAML (``configs/ablation_*.yaml``)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    variants = [str(v) for v in raw.get("variants", [])]
    seeds = [int(s) for s in raw.get("seeds", PAPER_SEEDS)]
    if not variants:
        raise ValueError(f"{path}: 'variants' list is empty")
    for v in variants:
        get_variant(v)
    return variants, seeds


def _run_path(out_dir: Path, variant: str, seed: int) -> Path:
    return out_dir / "runs" / f"{variant}__seed{seed}.json"


def run_ablation(cfg: MissionConfig, train: MissionData, test: MissionData, variants: Sequence[str],
                 seeds: Sequence[int], out_dir: str | Path, name: str = "ablation", device: str = "auto",
                 event_wise: bool = False, event_kwargs: dict[str, Any] | None = None, force: bool = False,
                 resume: bool = False, ddof: int = 1, meta: dict[str, Any] | None = None,
                 used_legacy_defaults: bool = False) -> list[dict[str, Any]]:
    """Run all ``variants`` x ``seeds`` and write CSV + JSON results.

    Args:
        cfg: mission config with the frozen hyperparameters (every variant derives from it).
        train / test: chronological split of the mission stream.
        variants / seeds: registry names and seeds.
        out_dir / name: output folder and file prefix.
        force: overwrite existing result files and run files.
        resume: skip (variant, seed) pairs that already have a finished run file.
        ddof: delta degrees of freedom of the reported std.
        used_legacy_defaults: written into every row and the meta (True only with ``--allow-legacy-defaults``).

    Returns:
        the list of result rows (one per variant x seed; failed runs have ``status`` starting with ``error``).

    Raises:
        FileExistsError: if results already exist and neither ``force`` nor (for run files) ``resume`` is set.
    """
    out = Path(out_dir)
    csv_path, json_path = out / f"{name}_results.csv", out / f"{name}_results.json"
    if (csv_path.exists() or json_path.exists()) and not force:
        raise FileExistsError(f"{csv_path} / {json_path} exist; pass --force to overwrite")
    existing_runs = [_run_path(out, v, s) for v in variants for s in seeds if _run_path(out, v, s).exists()]
    if existing_runs and not (force or resume):
        raise FileExistsError(f"{len(existing_runs)} finished run file(s) exist in {out / 'runs'}; pass --resume to reuse or --force to overwrite")
    rows: list[dict[str, Any]] = []
    for vname in variants:
        var = get_variant(vname)
        vcfg = apply_variant(cfg, var)
        for seed in seeds:
            rp = _run_path(out, vname, seed)
            if resume and not force and rp.exists():
                rows.append(json.loads(rp.read_text(encoding="utf-8")))
                log.info("resume: reusing %s seed %d", vname, seed)
                continue
            log.info("variant %s | seed %d", vname, seed)
            try:
                res = strip_private(train_and_evaluate(vcfg, train, test, seed, device, event_wise, event_kwargs,
                                                  report_extra={"used_legacy_defaults": used_legacy_defaults}))
            except Exception as exc:  # keep going; the failure is recorded and the exit code reflects it
                log.exception("variant %s seed %d failed", vname, seed)
                rows.append({"variant": vname, "group": var.group, "seed": int(seed), "status": f"error: {exc}",
                             "used_legacy_defaults": used_legacy_defaults})
                continue
            res = {"variant": vname, "group": var.group, **res}
            save_json(res, rp)
            rows.append(res)
    summary = aggregate(rows, "variant", ddof)
    info = {"name": name, "mission": cfg.name, "variants": list(variants), "seeds": [int(s) for s in seeds],
            "std_ddof": ddof, "protocol": "validation block carved from the TRAINING split; threshold = F1-optimal grid on that block; normalizer from the fitting part; mask/spike filter from the training split; test scored once; no point-adjustment; shared pipeline atandn.evaluation.pipeline",
            "used_legacy_defaults": used_legacy_defaults, "config": cfg.to_dict(), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "descriptions": {v: get_variant(v).description for v in variants}, **(meta or {})}
    write_rows_csv(rows, csv_path)
    save_json({"meta": info, "rows": rows, "summary": summary}, json_path)
    log.info("wrote %s and %s", csv_path, json_path)
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="mission YAML (configs/m1.yaml, m2.yaml, ops_sat.yaml)")
    p.add_argument("--best-params", help="best-params JSON; hyperparameters are frozen for ALL variants")
    p.add_argument("--allow-legacy-defaults", action="store_true",
                   help="run without --best-params using predefined mission defaults")
    p.add_argument("--data-root", required=True, help="folder with the prepared mission data")
    p.add_argument("--study", help="study YAML with 'variants' and 'seeds' (configs/ablation_norm.yaml, ...)")
    p.add_argument("--variants", nargs="+", help=f"variant names (override --study); available: {', '.join(VARIANTS)}")
    p.add_argument("--seeds", nargs="+", type=int, help=f"seeds (override --study); default {list(PAPER_SEEDS)}")
    p.add_argument("--out-dir", required=True, help="output folder")
    p.add_argument("--name", default=None, help="file prefix (default: study file stem or 'ablation')")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs (smoke runs; recorded in meta)")
    p.add_argument("--event-wise", action="store_true", help="also compute official ESA-ADB event-wise metrics")
    p.add_argument("--esa-module", default="esa_adb", help="module of the official ESA-ADB scorer")
    p.add_argument("--esa-attribute", default="evaluate", help="callable inside --esa-module")
    p.add_argument("--ddof", type=int, default=1, help="ddof of the reported std (default 1, sample std)")
    p.add_argument("--force", action="store_true", help="overwrite existing results")
    p.add_argument("--resume", action="store_true", help="reuse finished runs in <out-dir>/runs")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 1 if any run failed."""
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
    variants: list[str] = list(args.variants or [])
    seeds: list[int] = list(args.seeds or [])
    if args.study:
        s_variants, s_seeds = load_study_file(args.study)
        variants = variants or s_variants
        seeds = seeds or s_seeds
    if not variants:
        build_parser().error("give --study or --variants")
    seeds = seeds or list(PAPER_SEEDS)
    name = args.name or (Path(args.study).stem if args.study else "ablation")
    train, test = load_mission_data(cfg, args.data_root)
    meta = {"best_params": args.best_params, "used_legacy_defaults": legacy, "max_epochs_override": args.max_epochs, "event_wise": args.event_wise}
    rows = run_ablation(cfg, train, test, variants, seeds, args.out_dir, name, args.device, args.event_wise,
                        {"module": args.esa_module, "attribute": args.esa_attribute}, args.force, args.resume,
                        args.ddof, meta, legacy)
    return 1 if any(str(r.get("status", "")).startswith("error") for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
