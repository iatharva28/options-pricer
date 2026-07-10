"""
BSM Greeks — analytical formulas.

FIRST ORDER:
  Delta (Δ) = ∂V/∂S    Call: e^(-qT)·N(d1)       Put: e^(-qT)·(N(d1)−1)
  Vega  (ν) = ∂V/∂σ    S·e^(-qT)·φ(d1)·√T         (same call/put, per 1% σ)
  Theta (Θ) = ∂V/∂t    see formula below           (sign: negative = decay, per calendar day)
  Rho   (ρ) = ∂V/∂r    see formula below           (per 1% r)

SECOND ORDER:
  Gamma (Γ) = ∂²V/∂S²   φ(d1)·e^(-qT) / (S·σ·√T)  (same call/put, always ≥ 0)
  Vanna     = ∂²V/∂S∂σ  −e^(-qT)·φ(d1)·d2/σ
  Volga     = ∂²V/∂σ²   Vega_raw·d1·d2/σ           (per 1% σ, consistent with vega)
  Charm     = ∂²V/∂S∂t  delta decay per calendar day

All formulas from Hull (2018) Options Futures and Other Derivatives, 10th Ed.

NUMERICAL CHECK:
  Every first-order Greek has a numerical_* counterpart using central differences.
  Used by tests to verify analytical formulas. Not intended for production use.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr

from src.models import bsm_d1, bsm_price_vec

__all__ = [
    "Greeks", "all_greeks",
    "delta", "gamma", "theta", "vega", "rho",
    "vanna", "volga", "charm",
    "vega_raw",
    "numerical_delta", "numerical_gamma", "numerical_vega",
    "numerical_theta", "numerical_rho",
]


@dataclass(slots=True)
class Greeks:
    option_type: str
    delta: float
    gamma: float
    theta: float   # $/calendar day
    vega:  float    # $ per 1% σ move
    rho:   float    # $ per 1% r move
    vanna: float
    volga: float
    charm: float    # Δ per calendar day


# ── helpers ──────────────────────────────────────────────────────────────────

_SQRT_2PI = np.sqrt(2.0 * np.pi)

def _phi(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / _SQRT_2PI

def _intermediates(S, K, T, r, sigma, q):
    d1 = bsm_d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2, ndtr(d1), ndtr(d2), _phi(d1)


# ── first order ──────────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    d1, _, Nd1, _, _ = _intermediates(S, K, T, r, sigma, q)
    if option_type == "call":
        return float(np.exp(-q * T) * Nd1)
    return float(np.exp(-q * T) * (Nd1 - 1))

def vega(S, K, T, r, sigma, q=0.0) -> float:
    """Per 1% change in σ."""
    _, _, _, _, nd1 = _intermediates(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * nd1 * np.sqrt(T) / 100.0)

def vega_raw(S, K, T, r, sigma, q=0.0) -> float:
    """Per unit change in σ. Used internally for IV solving."""
    _, _, _, _, nd1 = _intermediates(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * nd1 * np.sqrt(T))

# Backward-compat alias (v2 used _vega_raw; v3 promotes to public vega_raw)
_vega_raw = vega_raw

def theta(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """
    $/calendar day (negative = time decay).

    Call: (-S·e^(-qT)·φ(d1)·σ)/(2√T) − r·K·e^(-rT)·N(d2) + q·S·e^(-qT)·N(d1)
    Put:  (-S·e^(-qT)·φ(d1)·σ)/(2√T) + r·K·e^(-rT)·N(-d2) − q·S·e^(-qT)·N(-d1)
    Divided by 365 to convert annual → per calendar day.
    """
    d1, d2, Nd1, Nd2, nd1 = _intermediates(S, K, T, r, sigma, q)
    common = -S * np.exp(-q * T) * nd1 * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        annual = common - r * K * np.exp(-r * T) * Nd2 + q * S * np.exp(-q * T) * Nd1
    else:
        annual = common + r * K * np.exp(-r * T) * ndtr(-d2) - q * S * np.exp(-q * T) * ndtr(-d1)
    return float(annual / 365.0)

def rho(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """$ per 1% change in r."""
    _, d2, _, Nd2, _ = _intermediates(S, K, T, r, sigma, q)
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * Nd2 / 100.0)
    return float(-K * T * np.exp(-r * T) * ndtr(-d2) / 100.0)


# ── second order ─────────────────────────────────────────────────────────────

def gamma(S, K, T, r, sigma, q=0.0) -> float:
    """Always non-negative. Peaks ATM near expiry."""
    _, _, _, _, nd1 = _intermediates(S, K, T, r, sigma, q)
    return float(np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T)))

def vanna(S, K, T, r, sigma, q=0.0) -> float:
    """∂²V/∂S∂σ = -e^(-qT)·φ(d1)·d2/σ"""
    d1, d2, _, _, nd1 = _intermediates(S, K, T, r, sigma, q)
    return float(-np.exp(-q * T) * nd1 * d2 / sigma)

def volga(S, K, T, r, sigma, q=0.0) -> float:
    """∂²V/∂σ² = Vega·d1·d2/σ  (per 1% σ, consistent with vega)."""
    d1, d2, _, _, _ = _intermediates(S, K, T, r, sigma, q)
    return float(vega_raw(S, K, T, r, sigma, q) * d1 * d2 / sigma / 100.0)

def charm(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """
    ∂²V/∂S∂t = delta decay per calendar day.
    Call: -e^(-qT)·[φ(d1)·((r-q)/(σ√T) - d2/(2T)) - q·N(d1)]  / 365
    Put:  -e^(-qT)·[φ(d1)·((r-q)/(σ√T) - d2/(2T)) + q·N(-d1)]  / 365
    """
    d1, d2, Nd1, _, nd1 = _intermediates(S, K, T, r, sigma, q)
    inner = nd1 * ((r - q) / (sigma * np.sqrt(T)) - d2 / (2 * T))
    if option_type == "call":
        annual = -np.exp(-q * T) * (inner - q * Nd1)
    else:
        annual = -np.exp(-q * T) * (inner + q * ndtr(-d1))
    return float(annual / 365.0)


# ── convenience ──────────────────────────────────────────────────────────────

def all_greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> Greeks:
    """Compute all 8 Greeks at once. Returns Greeks dataclass."""
    return Greeks(
        option_type=option_type,
        delta=delta(S, K, T, r, sigma, option_type, q),
        gamma=gamma(S, K, T, r, sigma, q),
        theta=theta(S, K, T, r, sigma, option_type, q),
        vega=vega(S, K, T, r, sigma, q),
        rho=rho(S, K, T, r, sigma, option_type, q),
        vanna=vanna(S, K, T, r, sigma, q),
        volga=volga(S, K, T, r, sigma, q),
        charm=charm(S, K, T, r, sigma, option_type, q),
    )


# ── numerical cross-check (central differences) ──────────────────────────────
# These exist to validate the analytical formulas in tests.
# Bump sizes: h_S = 0.1% of S,  h_σ = 1%,  h_T = 1 day,  h_r = 1bp

def _p(S, K, T, r, sigma, otype, q):
    return bsm_price_vec(S, K, T, r, sigma, otype, q)

def numerical_delta(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    h = S * 0.001
    return float((_p(S+h, K, T, r, sigma, option_type, q)
                  - _p(S-h, K, T, r, sigma, option_type, q)) / (2*h))

def numerical_gamma(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    h = S * 0.001
    return float((_p(S+h, K, T, r, sigma, option_type, q)
                  - 2*_p(S, K, T, r, sigma, option_type, q)
                  + _p(S-h, K, T, r, sigma, option_type, q)) / h**2)

def numerical_vega(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """Per 1% σ (consistent with analytical vega)."""
    h = 0.01
    return float((_p(S, K, T, r, sigma+h, option_type, q)
                  - _p(S, K, T, r, sigma-h, option_type, q)) / (2*h) / 100.0)

def numerical_theta(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """Per calendar day."""
    h = 1/365
    if T <= h: return 0.0
    return float((_p(S, K, T-h, r, sigma, option_type, q)
                  - _p(S, K, T+h, r, sigma, option_type, q)) / (2*h) / 365.0)

def numerical_rho(S, K, T, r, sigma, option_type="call", q=0.0) -> float:
    """Per 1% r."""
    h = 0.0001
    return float((_p(S, K, T, r+h, sigma, option_type, q)
                  - _p(S, K, T, r-h, sigma, option_type, q)) / (2*h) / 100.0)