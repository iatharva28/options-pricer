"""
Option pricing models — BSM, CRR Binomial, Monte Carlo (GBM, antithetic).

All formulas from Hull (2018), Options Futures and Other Derivatives, 10th Ed.

Black-Scholes-Merton (BSM) — analytical, European only.
  d1 = [ln(S/K) + (r - q + σ²/2)·T] / (σ·√T)
  d2 = d1 - σ·√T
  Call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
  Put  = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

CRR Binomial Tree — recombining, European only.
  u = exp(σ·√Δt),  d = 1/u
  p = (exp((r-q)·Δt) - d) / (u - d)
  Backward induction with discount factor exp(-r·Δt)

Monte Carlo (GBM) — European only, with antithetic variates.
  S_T = S·exp((r - q - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)
  Antithetic: generate n_sims/2 normals Z, simulate both S_T(Z_i)
  and S_T(-Z_i), average payoffs within each pair. SE computed on
  n_sims/2 pair-averages (proper antithetic variance — narrower CI
  than naive iid estimator).
  95% CI reported in metadata.

Performance notes:
  - Uses scipy.special.ndtr (CDF) instead of scipy.stats.norm.cdf — 3-5× faster.
  - PricingResult uses slots=True for cache-friendly memory layout.
  - Vectorised paths use np.asarray at entry to avoid copies on ndarray inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import numpy as np
from scipy.special import ndtr  # standard normal CDF — faster than scipy.stats.norm

OptionType = Literal["call", "put"]
ArrayLike = Union[float, np.ndarray]

__all__ = [
    "OptionType", "PricingResult",
    "bsm_d1", "bsm_d2",
    "bsm_price", "bsm_price_vec",
    "binomial_price", "binomial_price_vs_steps",
    "mc_price", "mc_simulate_paths",
]


@dataclass(slots=True)
class PricingResult:
    model: str
    type: OptionType
    price: float
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float
    meta: dict = field(default_factory=dict)


# ── validation ───────────────────────────────────────────────────────────────

def _validate(S, K, T, r, sigma, q):
    """Scalar validation. For array inputs, use bsm_price_vec (no validation)."""
    if S <= 0:     raise ValueError(f"S must be positive, got {S}")
    if K <= 0:     raise ValueError(f"K must be positive, got {K}")
    if T <= 0:     raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0: raise ValueError(f"sigma must be positive, got {sigma}")
    if not (0 <= q < 1): raise ValueError(f"q must be in [0,1), got {q}")


# ── helpers ──────────────────────────────────────────────────────────────────

_SQRT_2PI = np.sqrt(2.0 * np.pi)

def _ndtr(x):
    """Vectorised standard normal CDF (ufunc — 3-5× faster than scipy.stats.norm.cdf)."""
    return ndtr(x)

def _npdf(x):
    """Vectorised standard normal PDF."""
    return np.exp(-0.5 * x * x) / _SQRT_2PI


# ── Black-Scholes ────────────────────────────────────────────────────────────

def bsm_d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bsm_d2(S, K, T, r, sigma, q=0.0):
    return bsm_d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bsm_price(S, K, T, r, sigma, option_type: OptionType = "call", q=0.0) -> PricingResult:
    """
    Price European option via BSM formula. SCALAR inputs only.

    For array inputs, use `bsm_price_vec` (skips validation for speed).
    """
    _validate(S, K, T, r, sigma, q)
    d1 = bsm_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * np.exp(-q*T) * _ndtr(d1) - K * np.exp(-r*T) * _ndtr(d2)
    else:
        price = K * np.exp(-r*T) * _ndtr(-d2) - S * np.exp(-q*T) * _ndtr(-d1)
    return PricingResult("BSM", option_type, float(price), S, K, T, r, sigma, q,
                         meta={"d1": float(d1), "d2": float(d2)})


def bsm_price_vec(S, K, T, r, sigma, option_type="call", q=0.0) -> np.ndarray:
    """
    Vectorised BSM. No validation, for internal batch use (vol surface, MC, etc.).
    Accepts scalars or numpy arrays of any broadcast-compatible shape.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return S * np.exp(-q*T) * _ndtr(d1) - K * np.exp(-r*T) * _ndtr(d2)
    return K * np.exp(-r*T) * _ndtr(-d2) - S * np.exp(-q*T) * _ndtr(-d1)


# ── Binomial Tree (CRR) ──────────────────────────────────────────────────────

def binomial_price(S, K, T, r, sigma, option_type: OptionType = "call",
                   q=0.0, steps: int = 200) -> PricingResult:
    """
    CRR recombining binomial tree, numpy vectorised.
    steps=200 gives |BSM - BTree| < $0.01 for standard options.
    """
    _validate(S, K, T, r, sigma, q)
    N  = int(steps)
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp((r - q) * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    if not (0 < p < 1):
        raise ValueError(f"Invalid risk-neutral probability p={p:.4f}. Check inputs.")

    j  = np.arange(0, N + 1)
    ST = S * (u**j) * (d**(N - j))

    V = np.maximum(ST - K, 0.0) if option_type == "call" else np.maximum(K - ST, 0.0)
    for _ in range(N):
        V = df * (p * V[1:] + (1 - p) * V[:-1])

    return PricingResult("Binomial", option_type, float(V[0]), S, K, T, r, sigma, q,
                         meta={"steps": N, "u": float(u), "d": float(d), "p": float(p)})


def binomial_price_vs_steps(S, K, T, r, sigma, option_type="call", q=0.0,
                             step_range=None) -> list[tuple[int, float]]:
    """Price across multiple N values. Returns [(N, price), ...]."""
    if step_range is None:
        step_range = [10, 25, 50, 100, 200, 500, 1000]
    return [(n, binomial_price(S, K, T, r, sigma, option_type, q, steps=n).price)
            for n in step_range]


# ── Monte Carlo (GBM) with antithetic variates ───────────────────────────────

def mc_price(S, K, T, r, sigma, option_type: OptionType = "call",
             q=0.0, n_sims: int = 100_000, seed: int = 42) -> PricingResult:
    """
    European option via Monte Carlo with antithetic variates.

    Method:
      Generate n_sims/2 normals Z_1,...,Z_n. For each, simulate both S_T(Z_i)
      and S_T(-Z_i). The antithetic estimator averages payoffs within each pair,
      reducing variance ~50% for monotone payoffs.

    SE formula: correctly computed on the n_sims/2 pair-averages (not 2n raw
    payoffs). This is the proper antithetic variance — narrower CI than naive
    iid estimator, while still being a valid 95% CI.

    Reports 95% CI in meta.
    """
    _validate(S, K, T, r, sigma, q)
    if n_sims < 2:
        raise ValueError(f"n_sims must be ≥ 2, got {n_sims}")

    rng  = np.random.default_rng(seed)
    half = n_sims // 2

    # Antithetic pair generation
    Z      = rng.standard_normal(half)
    drift  = (r - q - 0.5 * sigma**2) * T
    vol_T  = sigma * np.sqrt(T)

    ST_pos = S * np.exp(drift + vol_T *  Z)
    ST_neg = S * np.exp(drift + vol_T * -Z)

    if option_type == "call":
        pay_pos = np.maximum(ST_pos - K, 0.0)
        pay_neg = np.maximum(ST_neg - K, 0.0)
    else:
        pay_pos = np.maximum(K - ST_pos, 0.0)
        pay_neg = np.maximum(K - ST_neg, 0.0)

    # Pair-averaged discounted payoffs — the antithetic estimator
    disc   = np.exp(-r * T)
    Y      = 0.5 * (disc * pay_pos + disc * pay_neg)
    price  = float(np.mean(Y))
    se     = float(np.std(Y, ddof=1) / np.sqrt(half))   # SE on n=half pair-averages

    return PricingResult("MC", option_type, price, S, K, T, r, sigma, q, meta={
        "n_sims":     n_sims,
        "n_pairs":    half,
        "std_error":  se,
        "ci_lower":   price - 1.96 * se,
        "ci_upper":   price + 1.96 * se,
        "antithetic": True,
    })


def mc_simulate_paths(S, T, r, sigma, q=0.0,
                      n_paths: int = 500, n_steps: int = 252,
                      seed: int = 42) -> np.ndarray:
    """
    Simulate GBM paths for visualisation.
    Returns array shape (n_paths, n_steps+1). Column 0 = S for all paths.

    Uses the exact GBM solution (vectorised via np.cumsum):
      S_t = S_0 · exp( sum_{s<t} [(r-q-σ²/2)·dt + σ·√dt·Z_s] )
    ~10× faster than per-step Python loop.
    """
    rng   = np.random.default_rng(seed)
    dt    = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = drift + vol * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths[:, 1:] = S * np.exp(log_paths)
    return paths