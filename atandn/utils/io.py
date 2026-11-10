"""File and device helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, otherwise CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj: Any, path: str | Path) -> Path:
    """Write ``obj`` as indented JSON (numpy types are converted)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
