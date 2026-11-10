"""Parameter count, analytic FLOPs per window and latency."""
from __future__ import annotations

import time

import numpy as np
import torch

from atandn.utils.io import resolve_device


def count_parameters(model: torch.nn.Module, unique: bool = True) -> int:
    """Number of parameters; with ``unique=True`` tensors shared between layers (tying) are counted once."""
    params = model.parameters() if unique else (p for _, p in model.named_parameters(remove_duplicate=False))
    return int(sum(p.numel() for p in params))


def _conv_len(length: int, kernel: int = 5, stride: int = 2, padding: int = 2) -> int:
    return (length + 2 * padding - kernel) // stride + 1


def flops_per_window(
    in_channels: int,
    window_size: int,
    conv_filters: tuple[int, int] = (64, 128),
    kernel: int = 5,
    se_ratio: int = 8,
    classifier_hidden: tuple[int, ...] = (512, 256, 128),
    include_decoder: bool = True,
    inverse_method: str | None = "solve",
    neumann_steps: int = 4,
) -> dict:
    """Analytic multiply-add FLOPs of one forward pass over one window (2 FLOPs per multiply-add).

    Per component (L_l = encoder output length of layer l, ``L_l = (L_{l-1} + 4 - K) // 2 + 1``):
      * encoder Conv1D: ``2 L_l C_{l-1} C_l K``
      * BIGN: ``2 C_l^2 L_l`` per encoder layer
      * MHA: ``8 L C^2 + 4 L^2 C`` for the two encoder blocks and the bottleneck
      * SE: ``4 C^2 / r`` per encoder block (r = ``se_ratio``)
      * classifier: ``2 sum_m n_m n_{m+1}`` for ``C2 L2 -> hidden... -> 1``
      * decoder ConvTranspose1D: ``2 L_in C_in C_out K``
      * inverse BIGN (per decoder layer, ``L`` = output length): ``L ((2/3) C^3 + 3 C^2)`` for ``'solve'`` and
        ``2 K C^2 L`` for ``'neumann'`` (``K`` = ``neumann_steps``)

    Args:
        include_decoder: Add decoder ConvTranspose layers (and, if ``inverse_method`` is not None, the inverse BIGN).
        inverse_method: ``'solve'``, ``'neumann'`` or ``None`` (plain linear decoder, no inverse layer).

    Returns:
        dict of component FLOPs plus ``"total"`` and ``"encoder_total"`` / ``"decoder_total"``.
    """
    c0, (c1, c2) = in_channels, conv_filters
    l1 = _conv_len(window_size, kernel)
    l2 = _conv_len(l1, kernel)
    out: dict[str, float] = {
        "encoder_conv": 2 * l1 * c0 * c1 * kernel + 2 * l2 * c1 * c2 * kernel,
        "bign": 2 * c1 ** 2 * l1 + 2 * c2 ** 2 * l2,
        "mha": (8 * l1 * c1 ** 2 + 4 * l1 ** 2 * c1) + (8 * l2 * c2 ** 2 + 4 * l2 ** 2 * c2)
               + (8 * l2 * c2 ** 2 + 4 * l2 ** 2 * c2),
        "se": 4 * c1 ** 2 / se_ratio + 4 * c2 ** 2 / se_ratio,
    }
    sizes = [c2 * l2, *classifier_hidden, 1]
    out["classifier"] = 2 * sum(a * b for a, b in zip(sizes[:-1], sizes[1:]))
    enc_keys = list(out)
    out["decoder_conv"] = out["inverse_bign"] = 0.0
    if include_decoder:
        lo1, lo2 = 2 * l2, 2 * l2 * 2   # transposed-conv output lengths (stride 2 each); interpolated to T afterwards
        out["decoder_conv"] = 2 * l2 * c2 * c1 * kernel + 2 * l1 * c1 * c0 * kernel
        if inverse_method == "solve":
            out["inverse_bign"] = lo1 * ((2 / 3) * c1 ** 3 + 3 * c1 ** 2) + lo2 * ((2 / 3) * c0 ** 3 + 3 * c0 ** 2)
        elif inverse_method == "neumann":
            out["inverse_bign"] = 2 * neumann_steps * (c1 ** 2 * lo1 + c0 ** 2 * lo2)
        elif inverse_method is not None:
            raise ValueError("inverse_method must be 'solve', 'neumann' or None")
    out["encoder_total"] = float(sum(out[k] for k in enc_keys))
    out["decoder_total"] = float(out["decoder_conv"] + out["inverse_bign"])
    out["total"] = out["encoder_total"] + out["decoder_total"]
    return out


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module, input_shape: tuple[int, ...], device: str | torch.device = "auto", repeats: int = 200, warmup: int = 20
) -> dict:
    """Wall-clock latency of ``model(x)`` for a random ``x`` of ``input_shape`` (e.g. ``(1, C, T)``).

    CUDA is synchronised before and after each timed call. Returns mean/median/p95/std in milliseconds.
    """
    dev = resolve_device(device)
    was_training = model.training
    model.to(dev).eval()
    x = torch.randn(*input_shape, device=dev)
    use_cuda = dev.type == "cuda"
    for _ in range(warmup):
        model(x)
    if use_cuda:
        torch.cuda.synchronize(dev)
    times = []
    for _ in range(repeats):
        if use_cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        model(x)
        if use_cuda:
            torch.cuda.synchronize(dev)
        times.append((time.perf_counter() - t0) * 1e3)
    model.train(was_training)
    a = np.asarray(times)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "std_ms": float(a.std()), "device": str(dev), "repeats": repeats}
