"""Idealized threshold link (Lemma / Proposition thr): formulas, brute-force minimizer, diagnostic report."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from atandn.inference.threshold import grid_f1_search
from atandn.inference.threshold_link import (ideal_threshold, link_report, population_minimizer, posterior_threshold)


def _brute_minimizer(eta: float, w: float, eps: float) -> float:
    """Argmin over a fine p grid of the expected weighted smoothed BCE (loss is linear in the smoothed label)."""
    t = (1 - eps) * eta + 0.5 * eps
    p = np.linspace(1e-5, 1 - 1e-5, 200001)
    loss = -(w * t * np.log(p) + (1 - t) * np.log(1 - p))
    return float(p[np.argmin(loss)])


def test_lemma_minimizer_by_brute_force_matches_closed_form_and_paper_numbers():
    w, eps = 14.1, 0.05
    assert _brute_minimizer(0.0, w, eps) == pytest.approx(float(population_minimizer(0.0, w, eps)), abs=2e-5)
    assert _brute_minimizer(1.0, w, eps) == pytest.approx(float(population_minimizer(1.0, w, eps)), abs=2e-5)
    assert float(population_minimizer(0.0, w, eps)) == pytest.approx(0.265, abs=1e-3)
    assert float(population_minimizer(1.0, w, eps)) == pytest.approx(0.998, abs=1e-3)
    eta = np.linspace(0, 1, 11)
    assert np.all(np.diff(population_minimizer(eta, w, eps)) > 0)            # strictly increasing in eta


@pytest.mark.parametrize("eps", [0.0, 0.05])
@pytest.mark.parametrize("w", [1.0, 4.7, 14.1])
def test_round_trip_tau_to_posterior_and_back(eps, w):
    for f1 in (0.1, 0.5, 0.9):
        tau = ideal_threshold(f1, w, eps)
        assert 0.0 < tau < 1.0
        assert posterior_threshold(tau, w, eps) == pytest.approx(f1 / 2.0, abs=1e-12)
    for tau in (0.2, 0.5, 0.8):                                               # tau -> tau_eta -> tau
        eta = posterior_threshold(tau, w, eps)
        assert float(population_minimizer(eta, w, eps)) == pytest.approx(tau, abs=1e-12)


def test_unit_weight_without_smoothing_gives_half_f1():
    for f1 in (0.2, 0.64, 1.0):
        assert ideal_threshold(f1, 1.0, 0.0) == pytest.approx(f1 / 2.0)
    with pytest.raises(ValueError):
        ideal_threshold(0.5, 0.0, 0.0)
    with pytest.raises(ValueError):
        posterior_threshold(0.5, 1.0, 1.0)


def test_calibrated_logistic_model_has_validation_optimum_near_the_closed_form():
    """gamma_f = 0, eps = 0, calibrated posterior: logit p* = log w + logit eta, so a tiny logistic model can fit it."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    w, eps, n = 4.0, 0.0, 60000
    x = rng.standard_normal(n)
    eta = 1.0 / (1.0 + np.exp(-(2.0 * x - 2.0)))
    y = (rng.random(n) < eta).astype(np.float32)
    xt, yt = torch.tensor(x[:40000], dtype=torch.float32)[:, None], torch.tensor(y[:40000])
    lin = torch.nn.Linear(1, 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)

    def closure():
        opt.zero_grad()
        z = lin(xt).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")
        loss = (loss * (1 + (w - 1) * yt)).mean()                               # weight w on positives, gamma_f = 0
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p_val = torch.sigmoid(lin(torch.tensor(x[40000:], dtype=torch.float32)[:, None])).squeeze(-1).numpy()
    y_val = y[40000:]
    rep = link_report(p_val, y_val, w, eps)
    assert rep["computed"] and rep["diagnostic"] and "diagnostic only" in rep["note"]
    assert abs(rep["tau_grid"] - rep["tau_star_p"]) < 0.05 and rep["abs_diff"] < 0.05
    assert rep["spearman"] > 0.3 and rep["average_precision"] > float(y_val.mean())
    assert rep["f1_star_estimate"] == pytest.approx(grid_f1_search(p_val, y_val)[1])


def test_link_report_needs_both_classes():
    out = link_report(np.array([0.1, 0.2, 0.3]), np.zeros(3), 5.0, 0.05)
    assert out["computed"] is False and out["diagnostic"] is True
