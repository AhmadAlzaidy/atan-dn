"""Adapter to the OFFICIAL ESA-ADB event-wise scorer (optional dependency, imported lazily).

This module never re-implements or approximates the ESA-ADB score. It only forwards the arrays to the
official implementation. The official package is not vendored here; install it from the ESA-ADB repository
(https://github.com/kplabs-pl/ESA-ADB, see its README for the evaluation package / TimeEval-based metrics).

The adapter is intentionally thin and configurable because the entry point of the official package has not
been verified against an installed copy in this repository: pass ``scorer`` (a callable) explicitly, or set
``module`` / ``attribute`` to the official entry point.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

_INSTALL_HELP = (
    "The official ESA-ADB evaluation package is required for event-wise metrics but could not be imported "
    "(module '{module}'). Install it from https://github.com/kplabs-pl/ESA-ADB (follow its README) and make "
    "sure the module is importable, or pass `scorer=` / `module=` / `attribute=` to evaluate_event_wise. "
    "The score is deliberately not re-implemented in this repository."
)


def _load_scorer(module: str, attribute: str) -> Callable[..., Any]:
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(_INSTALL_HELP.format(module=module)) from exc
    try:
        return getattr(mod, attribute)
    except AttributeError as exc:
        raise ImportError(f"Module '{module}' has no attribute '{attribute}'. " + _INSTALL_HELP.format(module=module)) from exc


def evaluate_event_wise(
    y_true_samples: np.ndarray,
    y_pred_samples: np.ndarray,
    timestamps: Any,
    scorer: Callable[..., dict] | None = None,
    module: str = "esa_adb",
    attribute: str = "evaluate",
    **kw: Any,
) -> dict:
    """Event-wise evaluation with the official ESA-ADB scorer.

    Args:
        y_true_samples: ``(N,)`` 0/1 ground truth per sample.
        y_pred_samples: ``(N,)`` 0/1 predictions per sample (e.g. from ``windows_to_samples``).
        timestamps: ``(N,)`` sample timestamps, forwarded to the official scorer.
        scorer: Callable ``scorer(y_true, y_pred, timestamps, **kw) -> dict``; if None it is imported lazily
            from ``module.attribute``.
        **kw: Extra arguments forwarded unchanged to the official scorer.

    Raises:
        ImportError: if the official package is not installed (with install instructions).
    """
    fn = scorer if scorer is not None else _load_scorer(module, attribute)
    return fn(np.asarray(y_true_samples), np.asarray(y_pred_samples), timestamps, **kw)
