"""Checkpoint saving and loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from atandn.config import MissionConfig
from atandn.utils.io import _default, resolve_device


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: MissionConfig, extra: dict[str, Any] | None = None) -> Path:
    """Save weights, the full config and JSON-able ``extra`` (e.g. threshold, normalizer state, mask parameters)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    extra_clean = json.loads(json.dumps(extra or {}, default=_default))
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "cfg": cfg.to_dict(), "extra": extra_clean}, p)
    return p


def check_tied_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Raise ``ValueError`` if state_dict keys of one tied tensor hold different values.

    Tied tensors (e.g. the first decoder inverse layer shares beta/Gamma with the first encoder BIGN layer)
    appear under several keys; ``load_state_dict`` would silently let the last key win.
    """
    groups: dict[int, list[str]] = {}
    for key, tensor in model.state_dict(keep_vars=True).items():
        groups.setdefault(id(tensor), []).append(key)
    for keys in groups.values():
        present = [k for k in keys if k in state_dict]
        for other in present[1:]:
            a, b = state_dict[present[0]], state_dict[other]
            if a.shape != b.shape or not torch.equal(a, b):
                raise ValueError(f"Checkpoint is inconsistent: tied tensors '{present[0]}' and '{other}' share storage "
                                 "in the model but have different values in the checkpoint")


def load_checkpoint(path: str | Path, device: str | torch.device = "auto") -> tuple[torch.nn.Module, MissionConfig, dict]:
    """Rebuild the model from the stored config and load its weights.

    Returns:
        ``(model, cfg, extra)`` with the model in eval mode on ``device``.
    """
    from atandn.models import ATANDN   # imported lazily: the model module is optional for data-only use

    dev = resolve_device(device)
    blob = torch.load(Path(path), map_location=dev, weights_only=True)
    cfg_dict = blob["cfg"]
    if "pe_mode" not in (cfg_dict.get("model") or {}):   # checkpoints written before pe_mode existed used the input PE
        cfg_dict.setdefault("model", {})["pe_mode"] = "input"
    cfg = MissionConfig.from_dict(cfg_dict)
    model = ATANDN(**{**cfg.model.__dict__, "conv_filters": tuple(cfg.model.conv_filters)}).to(dev)
    check_tied_state_dict(model, blob["state_dict"])
    model.load_state_dict(blob["state_dict"])
    return model.eval(), cfg, blob.get("extra", {})


def checkpoint_config_conflicts(run_cfg: MissionConfig, ckpt_cfg: MissionConfig, frozen: dict | None = None) -> list[str]:
    """List the settings in which the run's YAML disagrees with what the checkpoint was trained/frozen with.

    Compared: ``data.train_ratio``, ``data.test_stride`` (against the frozen values when present), the
    ``eval`` choices that shape the frozen threshold or the post-processing (``normalizer``, ``alpha_rec``,
    ``use_known_event_mask``, ``min_consecutive``, ``threshold_method``, mask buffers). ``model.window_size`` is not
    compared because the checkpoint's value legitimately comes from the tuned best-params, not the YAML.
    """
    frozen = frozen or {}
    out: list[str] = []

    def cmp(name: str, mine: Any, theirs: Any) -> None:
        if mine != theirs:
            out.append(f"{name}: config={mine!r} checkpoint={theirs!r}")

    cmp("data.train_ratio", run_cfg.data.train_ratio, frozen.get("train_ratio", ckpt_cfg.data.train_ratio))
    cmp("data.test_stride", run_cfg.data.test_stride, frozen.get("test_stride", ckpt_cfg.data.test_stride))
    for k in ("normalizer", "alpha_rec", "use_known_event_mask", "min_consecutive", "threshold_method",
              "mask_buffer_before_minutes", "mask_buffer_after_minutes", "mask_source", "threshold_floor"):
        cmp(f"eval.{k}", getattr(run_cfg.eval, k), getattr(ckpt_cfg.eval, k))
    return out
