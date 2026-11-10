"""Idealized threshold link of the paper (Lemma ``lem:min`` and Proposition ``prop:thr``): a DIAGNOSTIC only.

For ``gamma_f = 0`` the minimizer of the expected weighted, label-smoothed BCE is
``p*(eta) = w t / (w t + 1 - t)`` with ``t = (1 - eps) eta + eps / 2`` (``w`` = ``w_pos``, ``eps`` = ``eps_ls``). Mapping
a threshold ``tau`` on the network output to the posterior gives ``tau_eta`` (:func:`posterior_threshold`), and for
calibrated posteriors the F1-optimal posterior threshold is ``F1* / 2``, which gives the idealized cut-off
``tau*_p`` (:func:`ideal_threshold`). Trained models use ``gamma_f > 0`` and a dual-evidence score, so this closed form
is NOT the deployed threshold: the deployed threshold is the validation-grid optimum. :func:`link_report` compares the
two on the VALIDATION block (training split) and checks the monotone link (Spearman / AP) that the paper relies on.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from atandn.inference.scoring import average_precision
from atandn.inference.threshold import grid_f1_search

log = logging.getLogger(__name__)

DIAGNOSTIC_NOTE = ("diagnostic only: the idealized closed form holds for gamma_f = 0 and calibrated posteriors; "
                   "the deployed threshold is the validation-grid F1 optimum on the dual-evidence score")


def _check(w_pos: float, eps_ls: float) -> None:
    if w_pos <= 0 or not 0.0 <= eps_ls < 1.0:
        raise ValueError("need w_pos > 0 and 0 <= eps_ls < 1")


def smoothed_posterior(eta, eps_ls: float):
    """``tilde eta = (1 - eps) eta + eps / 2`` (label smoothing applied to the posterior)."""
    return (1.0 - eps_ls) * np.asarray(eta, dtype=np.float64) + 0.5 * eps_ls


def population_minimizer(eta, w_pos: float, eps_ls: float):
    """Lemma ``p*(eta)``: minimizer over ``p`` of ``-[w t log p + (1 - t) log(1 - p)]`` with ``t = tilde eta``."""
    _check(w_pos, eps_ls)
    t = smoothed_posterior(eta, eps_ls)
    return w_pos * t / (w_pos * t + 1.0 - t)


def ideal_threshold(f1_star: float, w_pos: float, eps_ls: float) -> float:
    """Idealized F1-optimal cut-off ``tau*_p`` on the network output (Eq. tau-star).

    ``tilde tau* = (1 - eps) F1*/2 + eps/2`` and ``tau*_p = w tilde tau* / (w tilde tau* + 1 - tilde tau*)``.
    With ``eps = 0`` and ``w = 1`` this is ``F1* / 2``.
    """
    _check(w_pos, eps_ls)
    return float(population_minimizer(float(f1_star) / 2.0, w_pos, eps_ls))


def posterior_threshold(tau: float, w_pos: float, eps_ls: float) -> float:
    """Posterior threshold ``tau_eta`` equivalent to the threshold ``tau`` on ``p*`` (Eq. tau-eta).

    ``tilde tau = tau / (w (1 - tau) + tau)`` and ``tau_eta = (tilde tau - eps/2) / (1 - eps)``. The inverse of
    :func:`ideal_threshold` in the sense ``posterior_threshold(ideal_threshold(f, w, e), w, e) == f / 2``.
    """
    _check(w_pos, eps_ls)
    tau = float(tau)
    tilde = tau / (w_pos * (1.0 - tau) + tau)
    return float((tilde - 0.5 * eps_ls) / (1.0 - eps_ls))


def _f1_at(scores: np.ndarray, y: np.ndarray, thr: float) -> float:
    pred = scores > thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    return float(2 * tp / (tp + fp + int(y.sum())))


def link_report(prob_val: np.ndarray, y_val: np.ndarray, w_pos: float, eps_ls: float) -> dict[str, Any]:
    """Compare the idealized closed-form threshold with the validation-grid optimum on the classifier probability.

    Args:
        prob_val: classifier probabilities ``p = sigmoid(z)`` of the VALIDATION windows (alpha = 0, no mask, no filter).
        y_val: 0/1 window labels of those windows.
        w_pos / eps_ls: positive weight and label smoothing of the trained loss.

    Returns:
        dict (labelled ``diagnostic``): ``f1_star_estimate`` (best F1 on the validation grid of ``p``), ``tau_star_p``
        (closed form from that estimate), ``tau_grid`` (grid-optimal threshold on ``p``), ``tau_eta_of_grid`` (its
        posterior threshold), ``abs_diff``, the validation F1 at both thresholds, ``spearman`` and ``average_precision`` of
        ``p`` versus the labels. ``{"computed": False, ...}`` if the block lacks a class. Never used to set the
        deployed threshold.
    """
    p = np.asarray(prob_val, dtype=np.float64)
    y = np.asarray(y_val) > 0
    base: dict[str, Any] = {"diagnostic": True, "note": DIAGNOSTIC_NOTE, "w_pos": float(w_pos),
                            "eps_ls": float(eps_ls), "n_val_windows": int(p.size), "n_val_pos_windows": int(y.sum())}
    if p.size == 0 or not y.any() or y.all() or not np.all(np.isfinite(p)):
        return {**base, "computed": False, "reason": "validation block needs both classes and finite scores"}
    tau_grid, f1_star = grid_f1_search(p, y)
    tau_p = ideal_threshold(f1_star, w_pos, eps_ls)
    rho = pd.Series(p).rank().corr(pd.Series(y.astype(float)).rank())
    return {
        **base, "computed": True, "f1_star_estimate": float(f1_star), "tau_star_p": tau_p, "tau_grid": float(tau_grid),
        "tau_eta_of_grid": posterior_threshold(tau_grid, w_pos, eps_ls), "tau_eta_star": float(f1_star) / 2.0,
        "abs_diff": abs(tau_p - float(tau_grid)),
        "f1_at_tau_star_p": _f1_at(p, y, tau_p), "f1_at_tau_grid": _f1_at(p, y, tau_grid),
        "spearman": None if not np.isfinite(rho) else float(rho), "average_precision": float(average_precision(p, y)),
        "positive_rate": float(y.mean()),
    }
