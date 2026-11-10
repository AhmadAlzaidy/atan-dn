"""Shared helpers of the command-line scripts (argument groups, config/extras loading, data loading).

Only the standard library is imported at module level so that ``--help`` stays fast; heavy imports
(``torch``, ``atandn``) happen inside the functions.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXIT_OK = 0
EXIT_ERROR = 1      # runtime failure (missing file, training error, ...)
EXIT_CONFIG = 2     # invalid configuration or arguments

log = logging.getLogger("atandn.scripts")


class ScriptError(RuntimeError):
    """Raised for an expected failure; carries the process exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR):
        super().__init__(message)
        self.code = code


@dataclass
class SearchSettings:
    """Hyperparameter-search settings (``search:`` section of the YAML)."""

    n_trials: int = 75
    seed: int = 42
    n_startup_trials: int = 10
    max_epochs: int | None = None
    fixed: dict = field(default_factory=dict)


@dataclass
class RunPaths:
    """Locations of the pipeline artefacts of one run."""

    data_root: Path
    out_dir: Path
    raw_dir: Path | None

    @property
    def best_params(self) -> Path:
        """JSON written by ``tune.py``."""
        return self.out_dir / "best_params.json"

    @property
    def checkpoint(self) -> Path:
        """Checkpoint written by ``train.py``."""
        return self.out_dir / "model.pt"

    @property
    def mask_params(self) -> Path:
        """Stage-2 mask parameters written by ``train.py``."""
        return self.out_dir / "mask_params.json"

    @property
    def report(self) -> Path:
        """Evaluation report written by ``evaluate.py``."""
        return self.out_dir / "report.json"


def add_common_args(parser: argparse.ArgumentParser, with_device: bool = False) -> None:
    """Add ``--config``, ``--data-root``, ``--out-dir``, ``--log-level`` (and optionally ``--device``)."""
    parser.add_argument("--config", required=True, type=Path, help="mission YAML (see configs/README.md)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="folder that data.path is relative to (default: paths.data_root of the config, else '.')")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="folder for run artefacts (default: paths.out_dir of the config, else runs/<mission>)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="logging level (default: INFO)")
    if with_device:
        parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' or 'cuda:N' (default: auto)")


def setup_logging(level: str) -> None:
    """Configure logging for a script (stderr)."""
    from atandn.utils.logging import setup_logging as _setup

    _setup(level)


def load_run_config(path: Path) -> tuple[Any, SearchSettings, dict]:
    """Load a mission YAML: ``(MissionConfig, SearchSettings, paths-dict)``.

    Raises:
        ScriptError: (exit code 2) if the file is missing or invalid.
    """
    import yaml

    from atandn.config import load_config

    path = Path(path)
    if not path.is_file():
        raise ScriptError(f"config file not found: {path}", EXIT_CONFIG)
    try:
        cfg = load_config(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        s = raw.get("search") or {}
        unknown = set(s) - {f for f in SearchSettings.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown keys in 'search': {sorted(unknown)}")
        search = SearchSettings(**s)
        paths = raw.get("paths") or {}
        bad = set(paths) - {"data_root", "raw_dir", "out_dir"}
        if bad:
            raise ValueError(f"unknown keys in 'paths': {sorted(bad)}")
    except (ValueError, TypeError) as exc:
        raise ScriptError(f"invalid config {path}: {exc}", EXIT_CONFIG) from exc
    return cfg, search, paths


def resolve_paths(args: argparse.Namespace, cfg: Any, paths: dict) -> RunPaths:
    """Combine command-line flags, the ``paths:`` section and defaults into :class:`RunPaths`."""
    data_root = args.data_root or Path(paths.get("data_root") or ".")
    out_dir = args.out_dir or Path(paths.get("out_dir") or Path("runs") / cfg.name)
    raw = getattr(args, "raw_dir", None) or paths.get("raw_dir")
    return RunPaths(Path(data_root), Path(out_dir), Path(raw) if raw else None)


def load_mission(cfg: Any, data_root: Path):
    """Load the mission stream described by ``cfg.data`` (parquet for ESA, segments.csv for OPS-SAT).

    ESA also loads the telecommand executions when ``eval.mask_source`` is ``telecommands`` (see
    :func:`atandn.data.loader.load_mission_from_config`); OPS-SAT follows ``data.opssat_mode``.
    """
    from atandn.data import load_mission_from_config
    from atandn.data.esa import MISSION_FILES

    path = Path(data_root) / cfg.data.path
    if not path.is_file():
        raise ScriptError(f"data file not found: {path} (run scripts/prepare_data.py first, or fix --data-root)")
    if cfg.data.kind != "opssat":
        if cfg.name not in MISSION_FILES:
            raise ScriptError(f"ESA mission '{cfg.name}' is not one of {sorted(MISSION_FILES)}", EXIT_CONFIG)
        if path.name != MISSION_FILES[cfg.name]:
            raise ScriptError(f"data.path file name must be '{MISSION_FILES[cfg.name]}' for mission {cfg.name}", EXIT_CONFIG)
    try:
        return load_mission_from_config(cfg, data_root)
    except FileNotFoundError as exc:
        raise ScriptError(str(exc), EXIT_CONFIG) from exc


def prepare_cfg_for_data(cfg: Any, n_channels: int) -> None:
    """Set ``model.in_channels`` from the data (warn if the config disagreed)."""
    if cfg.model.in_channels != n_channels:
        log.warning("model.in_channels=%d in the config but the data has %d channels; using %d",
                    cfg.model.in_channels, n_channels, n_channels)
        cfg.model.in_channels = n_channels


def params_from_cfg(cfg: Any) -> dict:
    """The 15 searched hyperparameters as currently set in ``cfg`` (used to fill what a search left fixed)."""
    m, t = cfg.model, cfg.train
    return {
        "window_size": m.window_size, "overlap": t.overlap, "pos_weight": t.pos_weight, "lr": t.lr,
        "recon_weight": t.recon_weight, "dropout": m.dropout, "batch_size": t.batch_size,
        "focal_gamma": t.focal_gamma, "weight_decay": t.weight_decay, "recon_score_weight": t.recon_score_weight,
        "num_heads": m.num_heads, "conv_f1": m.conv_filters[0], "conv_f2": m.conv_filters[1],
        "use_weighted_sampling": t.use_weighted_sampling, "use_augmentation": t.use_augmentation,
    }


def run_cli(build_parser: Callable[[], argparse.ArgumentParser], run: Callable[[argparse.Namespace], None],
            argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run, and map exceptions to exit codes (0 ok, 1 runtime error, 2 config/usage error)."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        run(args)
    except ScriptError as exc:
        log.error("%s", exc)
        return exc.code
    except (FileNotFoundError, ImportError) as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    except ValueError as exc:
        log.error("invalid input: %s", exc)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception:  # noqa: BLE001 - last-resort handler of a CLI
        log.exception("unexpected failure")
        return EXIT_ERROR
    return EXIT_OK
