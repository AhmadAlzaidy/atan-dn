"""One place that turns a ``DataConfig`` into a :class:`MissionData` (used by the scripts and the experiments)."""
from __future__ import annotations

import logging
from pathlib import Path

from atandn.config import MissionConfig
from atandn.data.esa import load_esa_mission
from atandn.data.opssat import load_opssat, load_opssat_multichannel
from atandn.data.splits import MissionData

log = logging.getLogger(__name__)


def wants_telecommands(cfg: MissionConfig) -> bool:
    """True if the known-event mask is on and takes its intervals from telecommands (ESA missions only)."""
    return bool(cfg.eval.use_known_event_mask and cfg.eval.mask_source == "telecommands" and cfg.data.kind == "esa")


def _find_telecommand_root(root: Path, rel: str) -> Path:
    """Folder with ``telecommands.csv``: ``rel`` below ``root`` or below one of its three nearest parents."""
    tried = []
    for base in [root, *list(root.parents)[:3]]:
        cand = base / rel
        tried.append(str(cand))
        if (cand / "telecommands.csv").is_file():
            return cand
    raise FileNotFoundError("telecommands.csv not found; tried: " + ", ".join(tried))


def load_mission_from_config(cfg: MissionConfig, data_root: str | Path, parquet_in_root: bool = False) -> MissionData:
    """Load the mission stream described by ``cfg.data`` below ``data_root``.

    ``parquet_in_root=True`` (the experiment runners): for ESA, ``data_root`` is the folder that directly holds the
    prepared parquet (``labels.csv`` next to it) instead of the folder ``data.path`` is relative to.

    ESA: the prepared parquet (``data.path``) and, when ``eval.mask_source == "telecommands"`` with the mask on, the
    telecommand executions from the raw mission folder ``data_root / data.telecommand_dir`` (``meta["telecommands"]``).
    OPS-SAT: ``data.opssat_mode`` selects the interleaved legacy stream (C = 1) or the nine-channel pivot (C = 9).

    Raises:
        FileNotFoundError: data file missing, or telecommands needed but ``data.telecommand_dir`` is not set/valid.
    """
    root = Path(data_root)
    path = root / cfg.data.path
    if cfg.data.kind == "opssat":
        if cfg.data.opssat_mode == "multichannel":
            return load_opssat_multichannel(path)
        return load_opssat(path)
    tc_root = None
    if wants_telecommands(cfg):
        if not cfg.data.telecommand_dir:
            raise FileNotFoundError("eval.mask_source is 'telecommands' but data.telecommand_dir is empty: set it to the "
                                    "raw ESA mission folder that holds telecommands.csv (or use "
                                    "eval.mask_source: rare_events / eval.use_known_event_mask: false)")
        tc_root = _find_telecommand_root(root, cfg.data.telecommand_dir)
    return load_esa_mission(root if parquet_in_root else path.parent, cfg.name, telecommand_root=tc_root,
                            telecommand_priorities=cfg.data.telecommand_priorities)
