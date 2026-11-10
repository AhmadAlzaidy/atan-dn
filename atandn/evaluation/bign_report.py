"""BIGN saturation report: trained Gamma/beta statistics and the empirical check of the output bound."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from atandn.layers.bign import BIGN1D

log = logging.getLogger(__name__)

_TOL = 1e-5  # relative slack for float32 rounding in the bound check


def _as_tensor(X: Any) -> torch.Tensor:
    return torch.as_tensor(X if torch.is_tensor(X) else np.asarray(X), dtype=torch.float32)


@torch.no_grad()
def _empirical_max_abs_y(model: torch.nn.Module, X: Any, batch_size: int) -> dict[int, torch.Tensor]:
    """Per-channel max |y| of every BIGN layer (keyed by ``id(layer)``) over the windows ``X`` (N, C_in, T)."""
    layers = [m for m in model.modules() if isinstance(m, BIGN1D)]
    acc: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer: BIGN1D):
        def hook(_m, _inp, out: torch.Tensor) -> None:
            cur = out.detach().abs().amax(dim=(0, 2)).cpu()
            acc[id(layer)] = cur if id(layer) not in acc else torch.maximum(acc[id(layer)], cur)
        return hook

    for layer in layers:
        hooks.append(layer.register_forward_hook(make_hook(layer)))
    was_training = model.training
    model.eval()
    try:
        dev = next(model.parameters()).device
        x = _as_tensor(X)
        for i in range(0, len(x), batch_size):
            model(x[i:i + batch_size].to(dev))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return acc


@torch.no_grad()
def bign_stats(model: torch.nn.Module, X: Any = None, batch_size: int = 256) -> list[dict[str, Any]]:
    """Statistics of every BIGN layer (Prop. bounded of the paper, ``|y_i| <= gamma_ii^(-1/2)``).

    Args:
        model: an ``ATANDN`` (or any module with ``BIGN1D`` submodules).
        X: optional windows ``(N, C_in, T)`` (already window-z-normalized; use TEST windows only, the numbers are
            descriptive and never fitted). If given, the empirical per-channel max ``|y_i|`` is measured with forward
            hooks and compared with the bound.
        batch_size: forward batch size for ``X``.

    Returns:
        One dict per BIGN layer with ``layer`` (module name), ``C``, ``gamma_ii_min/median/max``,
        ``inv_sqrt_max`` (max of ``gamma_ii^(-1/2)``, the output bound), ``R`` (``sqrt(sum 1/gamma_ii)``),
        ``beta_min``, ``offdiag_max``, ``offdiag_mean``, ``rowsum_max`` (max row sum of Gamma) and, when ``X`` is
        given, ``n_windows``, ``empirical_max_abs_y`` and ``bound_holds`` (every channel's max |y_i| <= its bound).
    """
    named = [(n, m) for n, m in model.named_modules() if isinstance(m, BIGN1D)]
    emp = _empirical_max_abs_y(model, X, batch_size) if X is not None and len(X) > 0 else {}
    out: list[dict[str, Any]] = []
    for name, layer in named:
        g = layer.gamma.detach().double().cpu()
        c = g.shape[0]
        diag = torch.diagonal(g)
        inv_sqrt = diag.rsqrt()
        off = g[~torch.eye(c, dtype=torch.bool)] if c > 1 else torch.zeros(1, dtype=g.dtype)
        row: dict[str, Any] = {
            "layer": name, "C": int(c),
            "gamma_ii_min": float(diag.min()), "gamma_ii_median": float(diag.median()), "gamma_ii_max": float(diag.max()),
            "inv_sqrt_max": float(inv_sqrt.max()), "R": float((1.0 / diag).sum().sqrt()),
            "beta_min": float(layer.beta.detach().min()),
            "offdiag_max": float(off.max()), "offdiag_mean": float(off.mean()),
            "rowsum_max": float(g.sum(dim=1).max()),
        }
        if id(layer) in emp:
            ymax = emp[id(layer)].double()
            row["n_windows"] = int(len(X))
            row["empirical_max_abs_y"] = float(ymax.max())
            row["bound_holds"] = bool(torch.all(ymax <= inv_sqrt * (1.0 + _TOL)).item())
        out.append(row)
    return out
