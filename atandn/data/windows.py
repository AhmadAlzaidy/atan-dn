"""Windowing, per-window z-score and the training/test window datasets."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def window_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-window, per-channel z-score of a ``(T, C)`` window (population std, as in the legacy code)."""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)


def _starts(n: int, window_size: int, stride: int) -> np.ndarray:
    if n < window_size:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n - window_size + 1, stride, dtype=np.int64)


def augment_window(x: torch.Tensor) -> torch.Tensor:
    """Legacy augmentation of a normalised ``(C, T)`` window.

    With probability 0.10 add Gaussian noise (sigma 0.02), 0.10 zero one channel (if C > 1) and 0.15 roll
    the window by a shift in [-3, 3]. Uses the global torch RNG (seeded by ``set_seed``).
    """
    c = x.shape[0]
    x = x.clone()
    if torch.rand(1).item() < 0.1:
        x = x + torch.randn_like(x) * 0.02
    if torch.rand(1).item() < 0.1 and c > 1:
        x[int(torch.randint(0, c, (1,)))] = 0
    if torch.rand(1).item() < 0.15:
        shift = int(torch.randint(-3, 4, (1,)))
        if shift:
            x = torch.roll(x, shifts=shift, dims=1)
    return x


def _window_labels(labels: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    csum = np.concatenate([[0], np.cumsum(np.asarray(labels).astype(np.int64) > 0)])
    return ((csum[starts + window_size] - csum[starts]) > 0).astype(np.int64)


class WindowDataset(Dataset):
    """Sliding windows over a ``(N, C)`` stream; each window is z-scored on its own.

    Item: ``(x_norm (C, T) float32, y float32)`` with ``y = max`` of the sample labels in the window.
    """

    def __init__(self, values: np.ndarray, labels: np.ndarray, window_size: int, stride: int, augment: bool = False):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if len(values) != len(labels):
            raise ValueError("values and labels must have the same length")
        self.values, self.window_size, self.stride, self.augment = values, int(window_size), int(stride), augment
        self.starts = _starts(len(values), self.window_size, self.stride)
        self.window_labels = _window_labels(labels, self.starts, self.window_size)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self.starts[idx])
        x = torch.from_numpy(window_zscore(self.values[s:s + self.window_size]).T.astype(np.float32).copy())
        if self.augment:
            x = augment_window(x)
        return x, torch.tensor(float(self.window_labels[idx]), dtype=torch.float32)

    def sampler_weights(self) -> list[float]:
        """Inverse-class-frequency weights for ``WeightedRandomSampler`` (legacy ``1/(count+1)``)."""
        counts = np.bincount(self.window_labels, minlength=2)
        w = 1.0 / (counts + 1)
        return [float(w[y]) for y in self.window_labels]


def make_test_windows(values: np.ndarray, labels: np.ndarray, timestamps, window_size: int, stride: int = 10) -> dict:
    """Build all z-scored windows of a stream for scoring.

    Returns:
        dict with ``X`` ``(n, C, T)`` float32, ``y`` ``(n,)`` window labels (max of sample labels), ``ts`` the
        timestamp of the LAST sample of each window (causal) and ``end_index`` the index of that sample.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    starts = _starts(len(values), window_size, stride)
    end_index = starts + window_size - 1
    ts_arr = np.asarray(timestamps)
    if len(starts) == 0:
        return {"X": np.zeros((0, values.shape[1], window_size), np.float32), "y": np.zeros(0, np.int64),
                "ts": ts_arr[:0], "end_index": end_index}
    X = np.stack([window_zscore(values[s:s + window_size]).T for s in starts]).astype(np.float32)
    return {"X": X, "y": _window_labels(labels, starts, window_size), "ts": ts_arr[end_index], "end_index": end_index}


def window_starts(n: int, window_size: int, stride: int) -> np.ndarray:
    """Start indices of the sliding windows of an ``n``-sample stream."""
    return _starts(n, window_size, stride)


def window_labels(labels: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    """Window label = 1 if the window contains at least one anomalous sample."""
    return _window_labels(labels, starts, window_size)


def iter_window_batches(values: np.ndarray, starts: np.ndarray, window_size: int, batch_size: int = 512):
    """Yield z-scored window batches ``(B, C, T)`` float32 without ever holding all windows in memory.

    Only ``batch_size`` windows exist at a time, so the peak extra memory is ``O(batch_size * C * T)`` whatever
    the length of the stream. The batches are bit-identical to the corresponding rows of
    :func:`make_test_windows`.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        yield np.stack([window_zscore(values[s:s + window_size]).T for s in chunk]).astype(np.float32)
