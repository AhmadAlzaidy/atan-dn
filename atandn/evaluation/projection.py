"""Theoretical on-board-computer (OBC) latency projection from the analytic FLOPs of one window.

``latency_ms = flops / (efficiency * peak_gflops * 1e9) * 1e3`` and ``fps = 1000 / latency_ms``. This is a
compute-bound estimate: memory bandwidth, data movement and operator support are ignored, and the peak values and
efficiencies come from the hardware table that the caller provides (``configs/obc_hardware.yaml``: copied from a legacy
script, vendor values NOT re-verified).

Viability class (rule inferred from the paper's latency table; the paper does not state it). With a decision period
``T`` (default 1 s, "a conservative 1 Hz decision rate") and a real-time fraction ``f`` (default 0.5):

* ``Real-Time``   : latency <= f * T (the model finishes well inside one decision period; paper: up to 103 ms),
* ``Real-Time*``  : f * T < latency <= T (marginal: still keeps up at the decision rate; paper: 646 ms),
* ``Snapshot``    : latency > T (cannot keep up; usable only for occasional snapshots; paper: 2217 ms, 6748 ms).

The value ``f = 0.5`` is a choice consistent with the paper's rows (any ``f`` in 0.103..0.646 reproduces them).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

REAL_TIME, MARGINAL, SNAPSHOT = "Real-Time", "Real-Time*", "Snapshot"


def latency_ms(flops: float, peak_gflops: float, efficiency: float) -> float:
    """Compute-bound latency in milliseconds: ``flops / (efficiency * peak_gflops * 1e9) * 1e3``.

    Raises:
        ValueError: non-positive FLOPs or peak, or efficiency outside (0, 1].
    """
    if flops <= 0 or peak_gflops <= 0 or not 0 < efficiency <= 1:
        raise ValueError("flops and peak_gflops must be > 0 and efficiency in (0, 1]")
    return float(flops) / (float(efficiency) * float(peak_gflops) * 1e9) * 1e3


def classify_viability(latency: float, decision_period_s: float = 1.0, realtime_fraction: float = 0.5) -> str:
    """Class of a latency (ms) against the decision period (see the module docstring)."""
    period_ms = decision_period_s * 1e3
    if latency <= realtime_fraction * period_ms:
        return REAL_TIME
    return MARGINAL if latency <= period_ms else SNAPSHOT


def project_latency(flops: float, hardware: Mapping[str, Mapping[str, Any]], decision_period_s: float = 1.0,
                    realtime_fraction: float = 0.5) -> list[dict[str, Any]]:
    """Project the latency of one window on every platform of ``hardware``.

    Args:
        flops: FLOPs of one forward pass (window).
        hardware: ``{name: {"peak_gflops": float, "efficiency": float, "type"?: str, "source"?: str}}``.
        decision_period_s: decision period in seconds (default 1 s).
        realtime_fraction: fraction of the period below which the class is ``Real-Time``.

    Returns:
        One dict per platform, in the order of ``hardware``: ``platform``, ``type``, ``peak_gflops``, ``efficiency``,
        ``latency_ms``, ``fps``, ``viability``, ``source``.
    """
    rows = []
    for name, spec in hardware.items():
        lat = latency_ms(flops, float(spec["peak_gflops"]), float(spec["efficiency"]))
        rows.append({"platform": name, "type": spec.get("type", ""), "peak_gflops": float(spec["peak_gflops"]),
                     "efficiency": float(spec["efficiency"]), "latency_ms": lat, "fps": 1000.0 / lat,
                     "viability": classify_viability(lat, decision_period_s, realtime_fraction),
                     "source": spec.get("source", ""), "verified": spec.get("verified")})
    return rows


def load_hardware(path: str | Path) -> tuple[dict[str, dict[str, Any]], float, float]:
    """Read a hardware YAML: ``(hardware dict, decision_period_s, realtime_fraction)``.

    Format: ``decision_period_s``, ``realtime_fraction`` (optional) and ``platforms``: a list of mappings with
    ``name``, ``peak_gflops``, ``efficiency`` (and optionally ``type``, ``source``).

    Raises:
        ValueError: malformed file or a duplicate platform name.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    plats = raw.get("platforms")
    if not isinstance(plats, list) or not plats:
        raise ValueError(f"{path}: 'platforms' must be a non-empty list")
    hw: dict[str, dict[str, Any]] = {}
    for p in plats:
        for k in ("name", "peak_gflops", "efficiency"):
            if k not in p:
                raise ValueError(f"{path}: platform entry without '{k}': {p}")
        if p["name"] in hw:
            raise ValueError(f"{path}: duplicate platform '{p['name']}'")
        hw[p["name"]] = {k: v for k, v in p.items() if k != "name"}
    return hw, float(raw.get("decision_period_s", 1.0)), float(raw.get("realtime_fraction", 0.5))
