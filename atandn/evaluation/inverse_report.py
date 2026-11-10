"""Conditioning, clamp rate and solve time of the exact-inverse BIGN decoder layers (a-priori bound kappa <= 2/eps)."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from atandn.layers.bign import ExactInverseBIGN1D

log = logging.getLogger(__name__)


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _system(layer: ExactInverseBIGN1D, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A (B*T,C,C), rhs (B*T,C,1))`` of ``(I - diag(s) Gamma) u = s * beta`` after clamping (float32)."""
    g, b = layer.gamma.detach().float(), layer.beta.detach().float()
    s = y.float() ** 2
    s = torch.minimum(s, ((1.0 - layer.eps) / g.sum(dim=1)).view(1, -1, 1))
    st = s.permute(0, 2, 1).reshape(-1, g.shape[0])
    a = torch.eye(g.shape[0], device=y.device) - st.unsqueeze(-1) * g
    return a, (st * b).unsqueeze(-1)


@torch.no_grad()
def inverse_report(model: torch.nn.Module, X: Any, max_pairs: int = 20000, batch_size: int = 128,
                   seed: int = 0, timing_repeats: int = 5) -> dict[str, Any]:
    """Clamp rate, condition number and solve-only time of every ``ExactInverseBIGN1D`` on windows ``X``.

    Use TEST windows (descriptive only, nothing is fitted). Per layer:

    * ``clamp_rate``: fraction of (window, channel, time) entries whose ``y^2`` hit the clamp ``(1-eps)/rowsum(Gamma)``,
      from ``last_clamp_rate`` of each forward batch (weighted by batch size).
    * ``kappa_p50/p99/max``: ``torch.linalg.cond(I - diag(s) Gamma, p=inf)`` (float64) over a random subsample of at
      most about ``max_pairs`` (window, time) pairs, after clamping.
    * ``solve_ms_per_window``: wall time of ``torch.linalg.solve`` alone (matrices built beforehand, CUDA synchronised,
      median of ``timing_repeats``) for the first batch, divided by its number of windows (all T time steps of one window).
    * ``kappa_bound``: a-priori bound ``2/eps`` (``||A||_inf <= 2 - eps`` and ``||A^-1||_inf <= 1/eps``).

    Args:
        model: ``ATANDN`` with ``inverse="exact"``.
        X: windows ``(N, C_in, T)`` as float array/tensor.
        max_pairs: approximate total number of (window, time) pairs sampled for kappa per layer.
        batch_size: forward batch size.
        seed: RNG seed of the subsampling.
        timing_repeats: repeats of the timed solve.

    Returns:
        ``{"n_windows", "n_pairs", "eps", "kappa_bound", "layers": [per-layer dicts], "kappa_max", "bound_holds"}``.
        With no exact-inverse layer, ``layers`` is empty.
    """
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, ExactInverseBIGN1D)]
    x = torch.as_tensor(X if torch.is_tensor(X) else np.asarray(X), dtype=torch.float32)
    n = len(x)
    rep: dict[str, Any] = {"n_windows": int(n), "n_pairs": 0, "eps": None, "kappa_bound": None, "layers": [],
                           "kappa_max": None, "bound_holds": None}
    if not layers or n == 0:
        return rep
    dev = next(model.parameters()).device
    gen = np.random.default_rng(seed)
    n_batches = (n + batch_size - 1) // batch_size
    per_batch = max(1, -(-max_pairs // n_batches))
    ys: dict[str, torch.Tensor] = {}
    store: dict[str, dict[str, Any]] = {nm: {"rate": 0.0, "kappa": [], "ms": None} for nm, _ in layers}
    hooks = [m.register_forward_hook(lambda _m, inp, _o, nm=nm: ys.__setitem__(nm, inp[0].detach()))
             for nm, m in layers]
    was_training = model.training
    model.eval()
    try:
        for i in range(0, n, batch_size):
            model(x[i:i + batch_size].to(dev))
            for nm, m in layers:
                y = ys[nm]
                bsz, _c, t = y.shape
                st = store[nm]
                st["rate"] += m.last_clamp_rate * bsz
                a, rhs = _system(m, y)
                idx = torch.as_tensor(gen.choice(bsz * t, size=min(per_batch, bsz * t), replace=False), device=a.device)
                st["kappa"].append(torch.linalg.cond(a[idx].double(), p=float("inf")).cpu())
                if st["ms"] is None:   # time the solve of the first batch only
                    torch.linalg.solve(a, rhs)  # warm-up
                    times = []
                    for _r in range(max(1, timing_repeats)):
                        _sync(dev)
                        t_start = time.perf_counter()
                        torch.linalg.solve(a, rhs)
                        _sync(dev)
                        times.append(time.perf_counter() - t_start)
                    st["ms"] = float(np.median(times)) * 1e3 / bsz
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    kmax = 0.0
    for nm, m in layers:
        st = store[nm]
        k = torch.cat(st["kappa"]).numpy()
        kmax = max(kmax, float(k.max()))
        rep["layers"].append({
            "layer": nm, "C": int(m.num_channels), "eps": float(m.eps), "method": m.method,
            "clamp_rate": float(st["rate"] / n), "n_pairs": int(len(k)),
            "kappa_p50": float(np.percentile(k, 50)), "kappa_p99": float(np.percentile(k, 99)), "kappa_max": float(k.max()),
            "solve_ms_per_window": st["ms"], "kappa_bound": 2.0 / float(m.eps),
        })
    eps = layers[0][1].eps
    rep.update({"eps": float(eps), "kappa_bound": 2.0 / float(eps),
                "n_pairs": int(sum(l["n_pairs"] for l in rep["layers"])), "kappa_max": kmax,
                "bound_holds": bool(all(l["kappa_max"] <= l["kappa_bound"] for l in rep["layers"]))})
    return rep
