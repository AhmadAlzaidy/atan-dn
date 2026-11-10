"""Bounded-Influence Gain Normalization (BIGN) and its decoder counterparts.

Notation: x, y have shape (B, C, T). ``Gamma`` is a (C, C) symmetric, entrywise
positive matrix, ``beta`` a (C,) positive vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gamma_from_raw(raw: Tensor) -> Tensor:
    """Symmetric, entrywise positive Gamma = 0.5 * (P + P^T), P = softplus(raw)."""
    p = F.softplus(raw)
    return 0.5 * (p + p.T)


def _fp_dtype(x: Tensor) -> torch.dtype:
    """float32 for half/bfloat16 inputs (mixed precision), otherwise the dtype of ``x`` (bit-identical path)."""
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def _inv_softplus(v: float) -> float:
    """Inverse of softplus: log(exp(v) - 1), computed stably."""
    return float(torch.log(torch.expm1(torch.tensor(float(v), dtype=torch.float64))).item())


def _init_params(c: int, gamma_init: float, mode: str = "diag_dominant",
                 gamma_diag: float = 0.5, gamma_off: float | None = None
                 ) -> tuple[nn.Parameter, nn.Parameter]:
    """Initialisation of (beta_raw, gamma_raw). beta_raw = gamma_init in every mode.

    ``mode == "uniform"`` (legacy): ``gamma_raw = gamma_init * I + N(0, 0.01^2)``, so
    Gamma_ii = softplus(0.1) = 0.744 and Gamma_ij = softplus(0) = 0.693 (~0.7 in EVERY entry), rowsum ~ 0.7 C.

    ``mode == "diag_dominant"``: with P = softplus(raw), Gamma = 0.5 (P + P^T); the
    diagonal of Gamma equals P_ii and an off-diagonal Gamma_ij = 0.5 (P_ij + P_ji).
    Choosing ``raw_ii = softplus^-1(gamma_diag) = log(exp(gamma_diag) - 1)`` and
    ``raw_ij = softplus^-1(gamma_off) = log(exp(gamma_off) - 1)`` (plus N(0, 0.01^2)
    noise, which keeps the matrix generic) gives Gamma_ii ~ gamma_diag and
    Gamma_ij ~ gamma_off. Default ``gamma_off = 0.01 / (C - 1)`` so the row sum is
    ~ gamma_diag + 0.01 < 1.5 for any C (and the clamp is not binding).
    """
    if mode not in ("diag_dominant", "uniform"):
        raise ValueError(f"gamma_init_mode must be 'diag_dominant' or 'uniform', got {mode!r}")
    beta_raw = nn.Parameter(torch.full((c,), float(gamma_init)))
    if mode == "uniform":
        gamma_raw = nn.Parameter(torch.eye(c) * gamma_init + torch.randn(c, c) * 0.01)
    else:
        off = gamma_off if gamma_off is not None else 0.01 / max(c - 1, 1)
        eye = torch.eye(c)
        base = eye * _inv_softplus(gamma_diag) + (1 - eye) * _inv_softplus(off)
        gamma_raw = nn.Parameter(base.float() + torch.randn(c, c) * 0.01)
    return beta_raw, gamma_raw


class BIGN1D(nn.Module):
    """Bounded-Influence Gain Normalization (pointwise in time, cross-channel).

    ``D_i(t) = beta_i + sum_j Gamma_ij x_j(t)^2``, ``y_i(t) = x_i(t) / sqrt(D_i(t))``.
    Because D_i >= beta_i >= beta_min > 0 no epsilon is needed, and because
    D_i >= Gamma_ii x_i^2, ``|y_i| <= Gamma_ii^(-1/2)`` for every input.
    Uses the current time step only (causal). Gamma is symmetric and entrywise
    positive, not necessarily positive semi-definite.

    Args:
        num_channels: C.
        beta_min: positive floor added to softplus(beta_raw).
        gamma_init: initial value of beta_raw (and of gamma_raw's diagonal in "uniform" mode).
        gamma_init_mode: "diag_dominant" (default; Gamma_ii ~ gamma_diag, off-diagonal
            ~ gamma_off, see ``_init_params``) or "uniform" (legacy, all entries ~0.74).
        gamma_diag, gamma_off: targets for "diag_dominant" (gamma_off None -> 0.01/(C-1)).
    """

    def __init__(self, num_channels: int, beta_min: float = 1e-6, gamma_init: float = 0.1,
                 gamma_init_mode: str = "diag_dominant", gamma_diag: float = 0.5,
                 gamma_off: float | None = None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.beta_min = beta_min
        self.beta_raw, self.gamma_raw = _init_params(num_channels, gamma_init, gamma_init_mode, gamma_diag, gamma_off)

    @property
    def beta(self) -> Tensor:
        """Positive vector (C,): softplus(beta_raw) + beta_min."""
        return F.softplus(self.beta_raw) + self.beta_min

    @property
    def gamma(self) -> Tensor:
        """Symmetric, entrywise positive matrix (C, C)."""
        return _gamma_from_raw(self.gamma_raw)

    def denominator(self, x: Tensor) -> Tensor:
        """Return D (B, C, T) = beta + Gamma @ x^2 (channel mixing at each time step)."""
        return self.beta.view(1, -1, 1) + torch.matmul(self.gamma, x * x)

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, C, T) -> (B, C, T).

        Always evaluated outside autocast, in float32 (float32/float64 inputs keep their dtype; half/bfloat16 inputs
        are promoted and the output stays float32, the next autocast layer casts it down again).
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(_fp_dtype(x))
            return x * torch.rsqrt(self.denominator(x))

    def output_bounds(self) -> dict[str, Tensor]:
        """Analytic bounds per channel: ``|y_i| <= Gamma_ii^(-1/2)`` (saturation level)
        and ``|y_i| <= beta_i^(-1/2) |x_i|`` (maximal small-signal gain)."""
        return {
            "gamma_ii_inv_sqrt": torch.diagonal(self.gamma).rsqrt(),
            "beta_inv_sqrt": self.beta.rsqrt(),
        }


class ExactInverseBIGN1D(nn.Module):
    """Algebraic inverse of :class:`BIGN1D`, exact only away from saturation (see ``docs/BIGN_INVERSE.md``).

    Formula. With u = x^2 and s = y^2 the forward map gives ``(I - diag(s) Gamma) u = s * beta``, hence
    ``u = (I - diag(s) Gamma)^-1 (s * beta)`` and ``x_i = sign(y_i) sqrt(u_i)``. For y produced by
    BIGN1D the spectral radius of ``diag(s) Gamma`` is < 1, so the system is solvable.

    Clamp (not changed by design). For arbitrary y (decoder outputs) s is clamped to
    ``s_max_i = (1 - eps) / rowsum_i(Gamma)`` so that ``||diag(s) Gamma||_inf <= 1 - eps`` and the system stays
    well-posed (condition number <~ 2/eps). The clamp is stricter than the range of the forward map
    (``y_i^2 <= 1/Gamma_ii``). When it fires, the inverse is no longer the inverse of the forward map, and the
    clamped entries receive zero gradient.

    When the round trip ``inv(bign(x))`` is exact. Only if no entry of ``y^2`` reaches ``s_max``. For unit-scale
    inputs (x ~ N(0, 1), e.g. after BatchNorm) with the default "diag_dominant" init this holds (measured with
    ``scripts/inverse_diagnostics.py``: float32, x of shape (8, C, 64), seeds 0..7, mean: relative error 2e-7 to
    4e-7, clamp rate 0, C = 1..192). For large |x| the BIGN output approaches its saturation level
    ``|y_i| -> Gamma_ii^(-1/2)``, the clamp fires and the error grows (x ~ N(0, 10^2): clamp rate ~20% and relative
    error 0.37-0.38 for C >= 6, 2e-4 for C = 1; scale 1000: error ~0.97-0.99). With "uniform" init and C >= 6 the
    clamp already binds at unit scale (error 0.60 for C = 6, 0.87 for C = 64, 0.93 for C = 192). Full table and
    conditions: ``docs/BIGN_INVERSE.md``. Use :meth:`round_trip_error` to check your own data range.

    Diagnostics: ``last_clamp_rate`` (fraction of clamped entries of the last call) and
    ``last_max_clamped_ratio`` (max s / s_max, > 1 means some entry was clamped) are properties. The forward
    pass only stores detached tensors; the host synchronisation (``.item()``) happens when the property is read.
    ``last_clamp_rate_tensor`` / ``last_max_clamped_ratio_tensor`` return the stored tensors without a sync.

    Methods: ``"solve"`` (DEFAULT and recommended; batched ``torch.linalg.solve`` over batch and time,
    O(C^3)) or ``"neumann"`` (APPROXIMATION: K fixed-point steps ``u <- s*(beta + Gamma u)`` from
    ``u = s*beta``, O(K C^2)). Under the clamp ``||diag(s) Gamma||_inf`` can be 0.999, so the error ~ rho^K has no
    useful guarantee: measured round-trip relative error at scale 1 (same conditions) with K = 4 is 0.11 (C = 1..192);
    K = 50 gives 2e-4..4e-4 (but 0.49-0.52 at scale 10). Do not read the ``inv_exact_neumann`` ablation as "the same inverse, cheaper".

    Tying: pass ``tied_to=bign``. The inverse then reads beta/Gamma from that layer (the same tensors, no
    copy) and owns no parameters. The tied layer is registered as submodule ``tied``, so ``model.parameters()``
    (which de-duplicates by identity) counts the shared tensors once; ``state_dict`` lists them under both
    prefixes (``atandn.training.checkpoint.load_checkpoint`` raises if the two copies disagree).

    Precision: the solve runs in ``solve_dtype`` (default float32); float32 is adequate for eps >= 1e-3,
    use ``torch.float64`` to remove the issue.
    """

    def __init__(self, num_channels: int | None = None, tied_to: BIGN1D | None = None,
                 method: str = "solve", neumann_steps: int = 4, eps: float = 1e-3,
                 beta_min: float = 1e-6, gamma_init: float = 0.1,
                 solve_dtype: torch.dtype = torch.float32,
                 gamma_init_mode: str = "diag_dominant", gamma_diag: float = 0.5,
                 gamma_off: float | None = None) -> None:
        super().__init__()
        if method not in ("solve", "neumann"):
            raise ValueError(f"method must be 'solve' or 'neumann', got {method!r}")
        if tied_to is None and num_channels is None:
            raise ValueError("give num_channels or tied_to")
        self.method, self.neumann_steps, self.eps = method, neumann_steps, eps
        self.solve_dtype = solve_dtype
        self._clamp_rate_t: Tensor | None = None
        self._max_ratio_t: Tensor | None = None
        if tied_to is not None:
            self.tied: BIGN1D | None = tied_to
            self.num_channels = tied_to.num_channels
        else:
            self.tied = None
            self.num_channels = int(num_channels)  # type: ignore[arg-type]
            self.beta_min = beta_min
            self.beta_raw, self.gamma_raw = _init_params(self.num_channels, gamma_init, gamma_init_mode, gamma_diag, gamma_off)

    @property
    def last_clamp_rate_tensor(self) -> Tensor | None:
        """Fraction of clamped entries of the last forward as a 0-dim tensor (no host sync); None before any call."""
        return self._clamp_rate_t

    @property
    def last_max_clamped_ratio_tensor(self) -> Tensor | None:
        """Max s / s_max of the last forward as a 0-dim tensor (no host sync); None before any call."""
        return self._max_ratio_t

    @property
    def last_clamp_rate(self) -> float:
        """Fraction of clamped entries of the last call (reads the stored tensor: one host sync)."""
        return 0.0 if self._clamp_rate_t is None else float(self._clamp_rate_t.item())

    @property
    def last_max_clamped_ratio(self) -> float:
        """Largest s / s_max of the last call; > 1 means some entry was clamped (one host sync)."""
        return 0.0 if self._max_ratio_t is None else float(self._max_ratio_t.item())

    @property
    def beta(self) -> Tensor:
        """Positive vector (C,) (of the tied layer when tied)."""
        if self.tied is not None:
            return self.tied.beta
        return F.softplus(self.beta_raw) + self.beta_min

    @property
    def gamma(self) -> Tensor:
        """Symmetric positive matrix (C, C) (of the tied layer when tied)."""
        if self.tied is not None:
            return self.tied.gamma
        return _gamma_from_raw(self.gamma_raw)

    def forward(self, y: Tensor) -> Tensor:
        """Map (B, C, T) -> (B, C, T); see class docstring. Runs outside autocast (float32 for half inputs)."""
        with torch.autocast(device_type=y.device.type, enabled=False):
            return self._forward(y.to(_fp_dtype(y)))

    def _forward(self, y: Tensor) -> Tensor:
        gamma, beta = self.gamma.to(self.solve_dtype), self.beta.to(self.solve_dtype)
        s = y.to(self.solve_dtype) ** 2
        s_max = ((1.0 - self.eps) / gamma.sum(dim=1)).view(1, -1, 1)
        with torch.no_grad():  # diagnostics stay on the device; read lazily through the properties
            self._clamp_rate_t = (s > s_max).to(torch.float32).mean()
            self._max_ratio_t = (s / s_max).max()
        s = torch.minimum(s, s_max)
        b = beta.view(1, -1, 1)
        if self.method == "solve":
            st = s.permute(0, 2, 1)                                    # (B, T, C)
            eye = torch.eye(gamma.shape[0], dtype=gamma.dtype, device=gamma.device)
            a = eye - st.unsqueeze(-1) * gamma                         # I - diag(s) Gamma
            rhs = (s * b).permute(0, 2, 1).unsqueeze(-1)               # (B, T, C, 1)
            u = torch.linalg.solve(a, rhs).squeeze(-1).permute(0, 2, 1)
        else:
            u = s * b
            for _ in range(self.neumann_steps):
                u = s * (b + torch.matmul(gamma, u))
        # clamp_min only guards sqrt and its gradient at exact zero (sign(0) = 0 keeps x = 0)
        x = torch.sign(y) * torch.sqrt(u.clamp_min(1e-30))
        return x.to(y.dtype)

    @staticmethod
    @torch.no_grad()
    def round_trip_error(x: Tensor, bign: BIGN1D, method: str = "solve", neumann_steps: int = 4,
                         eps: float = 1e-3, solve_dtype: torch.dtype = torch.float32) -> dict[str, float]:
        """Diagnose the round trip ``inv(bign(x))`` on your own data range.

        Args:
            x: (B, C, T) tensor, e.g. the input of the BIGN layer in your model.
            bign: the forward layer (its beta/Gamma are used by the tied inverse).
            method, neumann_steps, eps, solve_dtype: inverse settings to test.

        Returns:
            dict with ``rel_error`` (``||x_hat - x||_2 / ||x||_2``), ``clamp_rate`` (fraction of entries whose
            y^2 exceeded the clamp level) and ``max_clamped_ratio`` (max s / s_max). Rule of thumb: the round
            trip is exact when ``clamp_rate == 0`` (and method is "solve").
        """
        inv = ExactInverseBIGN1D(tied_to=bign, method=method, neumann_steps=neumann_steps, eps=eps,
                                 solve_dtype=solve_dtype)
        x_hat = inv(bign(x))
        num = torch.linalg.vector_norm((x_hat - x).to(torch.float64))
        den = torch.linalg.vector_norm(x.to(torch.float64)).clamp_min(1e-30)
        return {"rel_error": float(num / den), "clamp_rate": inv.last_clamp_rate,
                "max_clamped_ratio": inv.last_max_clamped_ratio}


class MirrorBIGN1D(nn.Module):
    """Legacy decoder layer: ``x * sqrt(beta + sum_j Gamma_ij x_j^2)``, own parameters.

    It mirrors the form of BIGN but is not its algebraic inverse. Kept for the
    exact-versus-mirror comparison. (The legacy 1e-8 terms are removed.)
    """

    def __init__(self, num_channels: int, beta_min: float = 1e-6, gamma_init: float = 0.1,
                 gamma_init_mode: str = "diag_dominant", gamma_diag: float = 0.5,
                 gamma_off: float | None = None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.beta_min = beta_min
        self.beta_raw, self.gamma_raw = _init_params(num_channels, gamma_init, gamma_init_mode, gamma_diag, gamma_off)

    @property
    def beta(self) -> Tensor:
        """Positive vector (C,)."""
        return F.softplus(self.beta_raw) + self.beta_min

    @property
    def gamma(self) -> Tensor:
        """Symmetric positive matrix (C, C)."""
        return _gamma_from_raw(self.gamma_raw)

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, C, T) -> (B, C, T)."""
        return x * torch.sqrt(self.beta.view(1, -1, 1) + torch.matmul(self.gamma, x * x))
