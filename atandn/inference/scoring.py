"""Window scoring: classifier probability, reconstruction error and the dual-evidence score."""
from __future__ import annotations

import numpy as np
import torch

from atandn.utils.io import resolve_device


def _forward_batch(model: torch.nn.Module, xb: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """``(prob, recon_err)`` of one batch (``recon_err`` zeros when the model has no reconstruction head)."""
    recon, logits = model(xb)
    prob = torch.sigmoid(logits.reshape(len(xb), -1)[:, 0]).cpu().numpy()
    if recon is None:
        return prob, np.zeros(len(xb), dtype=np.float32)
    return prob, ((recon - xb) ** 2).mean(dim=(1, 2)).cpu().numpy()


def has_reconstruction_head(model: torch.nn.Module) -> bool:
    """True if ``model`` reconstructs its input (a ``decoder`` attribute that is not ``None``; True if absent)."""
    return getattr(model, "decoder", True) is not None


@torch.no_grad()
def score_windows(model: torch.nn.Module, X: np.ndarray, batch_size: int = 512, device: str | torch.device = "auto") -> dict:
    """Run a frozen model over windows ``X`` of shape ``(N, C, T)``.

    Returns:
        ``{"prob": (N,), "recon_err": (N,)}``; ``recon_err`` is the per-window MSE between input and
        reconstruction (zeros when the model has no reconstruction head).
    """
    dev = resolve_device(device)
    was_training = model.training
    model.to(dev).eval()
    probs, errs = [], []
    for i in range(0, len(X), batch_size):
        p, e = _forward_batch(model, torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=dev))
        probs.append(p)
        errs.append(e)
    model.train(was_training)
    empty = np.zeros(0, dtype=np.float32)
    return {"prob": np.concatenate(probs) if probs else empty, "recon_err": np.concatenate(errs) if errs else empty}


@torch.no_grad()
def score_stream(model: torch.nn.Module, values: np.ndarray, window_size: int, stride: int,
                 batch_size: int = 512, device: str | torch.device = "auto") -> dict:
    """Score every sliding window of a ``(N, C)`` stream in chunks (memory bounded by one batch of windows).

    Windows are generated lazily (:func:`atandn.data.windows.iter_window_batches`), scored and discarded; only the
    per-window scores are kept. The result equals ``score_windows(model, make_test_windows(...)["X"])``.

    Returns:
        ``{"prob", "recon_err", "start", "end_index"}``, each ``(n_windows,)``; ``end_index`` is the index of the
        LAST sample of each window (causal timestamp).
    """
    from atandn.data.windows import iter_window_batches, window_starts

    values = np.asarray(values)
    starts = window_starts(len(values), window_size, stride)
    dev = resolve_device(device)
    was_training = model.training
    model.to(dev).eval()
    probs, errs = [], []
    for xb in iter_window_batches(values, starts, window_size, batch_size):
        p, e = _forward_batch(model, torch.as_tensor(xb, dtype=torch.float32, device=dev))
        probs.append(p)
        errs.append(e)
    model.train(was_training)
    empty = np.zeros(0, dtype=np.float32)
    return {"prob": np.concatenate(probs) if probs else empty, "recon_err": np.concatenate(errs) if errs else empty,
            "start": starts, "end_index": starts + window_size - 1}


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-free average precision ``sum_k (R_k - R_{k-1}) P_k`` (ties handled as one group).

    Returns NaN if ``labels`` has no positive.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels) > 0
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    last = np.r_[s[1:] != s[:-1], True]
    tp_g, k_g = tp[last], (np.arange(len(s)) + 1)[last]
    recall = tp_g / n_pos
    prec = tp_g / k_g
    return float(np.sum(np.diff(np.r_[0.0, recall]) * prec))


class ScoreNormalizer:
    """Maps reconstruction errors to ~[0, 1] using statistics of TRAINING scores only.

    Args:
        method: ``"percentile"`` (1st/99th percentile of the training errors) or ``"minmax"``.
        clip: Clip transformed values to [0, 1] so unseen (test) errors cannot exceed the training range.
    """

    def __init__(self, method: str = "percentile", clip: bool = True, low_q: float = 1.0, high_q: float = 99.0):
        if method not in {"percentile", "minmax"}:
            raise ValueError("method must be 'percentile' or 'minmax'")
        self.method, self.clip, self.low_q, self.high_q = method, clip, low_q, high_q
        self.lo: float | None = None
        self.hi: float | None = None

    def fit(self, recon_err_train: np.ndarray) -> "ScoreNormalizer":
        """Fit on training reconstruction errors."""
        e = np.asarray(recon_err_train, dtype=np.float64)
        if e.size == 0:
            raise ValueError("Cannot fit ScoreNormalizer on empty scores")
        if self.method == "minmax":
            self.lo, self.hi = float(e.min()), float(e.max())
        else:
            self.lo, self.hi = (float(v) for v in np.percentile(e, [self.low_q, self.high_q]))
        return self

    def transform(self, recon_err: np.ndarray) -> np.ndarray:
        """Normalise with the fitted statistics (all zeros if the training range is degenerate)."""
        if self.lo is None or self.hi is None:
            raise RuntimeError("ScoreNormalizer is not fitted")
        e = np.asarray(recon_err, dtype=np.float64)
        if self.hi <= self.lo:
            return np.zeros_like(e)
        out = (e - self.lo) / (self.hi - self.lo)
        return np.clip(out, 0.0, 1.0) if self.clip else out

    def state_dict(self) -> dict:
        """Serialisable state."""
        return {"method": self.method, "clip": self.clip, "low_q": self.low_q, "high_q": self.high_q, "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_state_dict(cls, d: dict) -> "ScoreNormalizer":
        """Rebuild from :meth:`state_dict`."""
        n = cls(d["method"], d["clip"], d["low_q"], d["high_q"])
        n.lo, n.hi = d["lo"], d["hi"]
        return n


def dual_evidence(prob: np.ndarray, recon_err: np.ndarray, alpha_rec: float, norm: ScoreNormalizer) -> np.ndarray:
    """Dual-evidence score ``(1 - a) * prob + a * normalised_recon_err``."""
    return (1.0 - alpha_rec) * np.asarray(prob, dtype=np.float64) + alpha_rec * norm.transform(recon_err)
