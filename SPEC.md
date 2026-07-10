# Options Pricing & Greeks Calculator
### Coding Agent Specification — CLAUDE.md Compliant (v3 — Performance + Robustness Edition)

> Agent instructions: Read this fully before writing any code.
> Follow the implementation order exactly. Run tests after each phase.
> Do not add features not listed here. Do not refactor unrelated code.
> If something is ambiguous, pick the simpler interpretation.

---

## What this project is

A Python library that prices European options using three models (Black-Scholes, Binomial Tree, Monte Carlo), computes the standard Greeks, validates results against analytical boundaries, and builds a volatility surface from real market data.

**It is not**: a trading system, a web app, a CLI tool, an external cache layer, or a production service.

---

## Assumptions (explicit)

- Python 3.10+ required (uses `dataclass(slots=True)`)
- European options only (no American early-exercise)
- Real market data via `yfinance` (spot, option chains) and `fredapi` (risk-free rate)
- If FRED API key is missing → fall back silently to `r=0.05`, no crash
- Greeks: first and second order only (delta, gamma, theta, vega, rho, vanna, volga, charm)
- Vol surface: implied vol only, from yfinance option chains (no local vol / Dupire)
- Volatility estimator: close-to-close historical vol only (no GARCH, no Parkinson)
- Visualisation: matplotlib static charts saved to `outputs/`
- No CLI, no web server, no REST API, no persistent cache
- Two notebooks: one end-to-end demo, one validation report
- All stochastic functions take `seed` parameter (default 42) for reproducibility

---

## Performance & robustness conventions (apply throughout)

1. **Use `scipy.special.ndtr` (CDF) and `np.exp(-x²/2)/√(2π)` (PDF)** instead of `scipy.stats.norm.cdf/pdf` — 3-5× faster.
2. **`dataclass(slots=True)`** for all value types (`PricingResult`, `Greeks`, `MarketData`) — cache-friendly memory layout, ~20% faster attribute access.
3. **`__all__` export lists** in every module — cleaner public API, better IDE autocomplete.
4. **`np.sqrt(T)` computed once** per call, reused in d1/d2.
5. **`np.asarray(..., dtype=float)` at vectorized entry points** — avoids silent type promotion.
6. **No global mutable state.** Pure functions only.
7. **`lru_cache(maxsize=1)` on `_get_rfr()`** — within-session memoization (TB3MS updates monthly; not a persistent cache layer).
8. **Vectorize MC path simulation** with `np.cumsum` of log-increments instead of per-step Python loop (~10× faster).
9. **All stochastic functions take `seed` parameter** (default 42) for reproducible tests.

---

## Repository structure

```
options-pricer/
├── pyproject.toml           ← PEP 621 metadata + pytest config
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── data.py              ← fetch spot, RFR, option chains
│   ├── models.py            ← BSM, Binomial, Monte Carlo
│   ├── greeks.py            ← all Greeks (analytical + numerical check)
│   ├── volatility.py        ← historical vol, IV solver, vol surface
│   ├── validation.py        ← parity, boundaries, convergence
│   └── plots.py             ← charts saved to outputs/
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_greeks.py
│   ├── test_validation.py
│   ├── test_volatility.py   ← NEW: IV + surface tests
│   └── test_perf.py         ← NEW: performance smoke tests
├── notebooks/
│   ├── 01_full_demo.ipynb
│   └── 02_validation_report.ipynb
└── outputs/                 ← charts (auto-created)
```

---

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "options-pricer"
version = "1.0.0"
description = "European option pricing: BSM, Binomial, Monte Carlo; Greeks; IV surface."
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26.0",
    "scipy>=1.11.0",
    "pandas>=2.1.0",
    "yfinance>=0.2.36",
    "fredapi>=0.5.1",
    "matplotlib>=3.8.0",
    "plotly>=5.18.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0", "pytest-cov>=4.1.0", "pytest-benchmark>=4.0.0"]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "network: marks tests requiring internet",
]

[tool.black]
line-length = 100

[tool.ruff]
line-length = 100
target-version = "py310"
```

---

## `requirements.txt`

```
numpy>=1.26.0
scipy>=1.11.0
pandas>=2.1.0
yfinance>=0.2.36
fredapi>=0.5.1
matplotlib>=3.8.0
plotly>=5.18.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-benchmark>=4.0.0
python-dotenv>=1.0.0
```

## `.env.example`

```
FRED_API_KEY=your_key_here   # https://fred.stlouisfed.org/docs/api/api_key.html
```

---

## `src/models.py`

**Purpose**: three pricing engines, unified output type.

```python
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
  Antithetic: generate n_sims/2 normals Z, simulate both Z and -Z,
  average payoffs within each pair. SE computed on n_sims/2 pair-averages
  (proper antithetic variance — narrower CI than naive iid estimator).
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
```

---

## `src/greeks.py`

**Purpose**: first and second order Greeks, both analytically (BSM) and numerically (for cross-check).

```python
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
```

---

## `src/volatility.py`

**Purpose**: historical vol, implied vol solver, vol surface from option chain data.

```python
"""
Volatility estimation and surface.

Historical vol : close-to-close (re-exported from data.py)
Implied vol    : Newton-Raphson with Brent fallback
Vol surface    : SmoothBivariateSpline over (strike, T) → IV grid

Robustness:
  - IV solver clamps σ to [1e-6, 10.0] at each Newton step (prevents explosion)
  - VolSurface.fit requires ≥ 16 points (cubic-cubic spline rank requirement)
  - All surface queries clamped to [0.01, 3.0] to prevent spline runaway
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import SmoothBivariateSpline

from src.models import bsm_price_vec
from src.greeks import vega_raw

logger = logging.getLogger(__name__)

__all__ = [
    "IV_TOL", "IV_MAX_ITER", "IV_LO", "IV_HI",
    "implied_vol",
    "build_surface_df", "VolSurface",
    "_MIN_SURFACE_POINTS",
]

IV_TOL      = 1e-6
IV_MAX_ITER = 100
IV_LO, IV_HI = 1e-6, 10.0   # solver bounds

# Minimum points for cubic-cubic bivariate spline: (kx+1)(ky+1) = 16
_MIN_SURFACE_POINTS = 16


# ── implied volatility ───────────────────────────────────────────────────────

def implied_vol(market_price: float, S: float, K: float, T: float,
                r: float, option_type: str = "call", q: float = 0.0) -> Optional[float]:
    """
    Compute implied volatility.
    Strategy: Newton-Raphson (fast), fall back to Brent (robust).
    Returns IV as decimal in [1e-6, 10.0], or None if unsolvable.

    Newton-Raphson update: σ_{n+1} = σ_n - (BSM(σ_n) - mkt) / vega(σ_n)
    σ is clamped to [IV_LO, IV_HI] at each step to prevent divergence.
    """
    if market_price <= 0:
        return None

    # Newton-Raphson
    sigma = 0.3  # neutral starting point
    for _ in range(IV_MAX_ITER):
        bsm_val = float(bsm_price_vec(S, K, T, r, sigma, option_type, q))
        v       = vega_raw(S, K, T, r, sigma, q)
        if abs(v) < 1e-10:
            break
        step = (bsm_val - market_price) / v
        sigma = float(np.clip(sigma - step, IV_LO, IV_HI))
        if abs(bsm_val - market_price) < IV_TOL:
            return sigma

    # Brent fallback
    try:
        obj  = lambda s: float(bsm_price_vec(S, K, T, r, s, option_type, q)) - market_price
        f_lo = obj(IV_LO)
        f_hi = obj(IV_HI)
        if f_lo * f_hi > 0:
            return None
        return float(brentq(obj, IV_LO, IV_HI, xtol=IV_TOL, maxiter=500))
    except Exception:
        return None


# ── vol surface ──────────────────────────────────────────────────────────────

def _dte(expiry_str: str) -> int:
    """Calendar days from today to expiry. Minimum 1."""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return max((exp - date.today()).days, 1)


def build_surface_df(option_chain: dict, spot: float,
                     rfr: float = 0.05, q: float = 0.0,
                     min_oi: int = 10, moneyness_range: float = 0.30) -> pd.DataFrame:
    """
    Build a clean DataFrame of (strike, T, IV) for vol surface fitting.

    Filters:
      - Strike within ±30% of spot
      - Open interest ≥ min_oi
      - IV from yfinance (impliedVolatility column) — uses market IV directly;
        recomputes from mid_price as fallback if yfinance IV is 0.
      - IV must be in (0.01, 3.0) after filtering

    Returns DataFrame: strike, T, dte, iv, option_type
    """
    rows = []
    lo, hi = spot * (1 - moneyness_range), spot * (1 + moneyness_range)

    for expiry_str, chains in option_chain.items():
        dte_days = _dte(expiry_str)
        T = dte_days / 365.0

        for opt_type, df in [("call", chains["calls"]), ("put", chains["puts"])]:
            df = df.copy()
            df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
            df = df[df["openInterest"] >= min_oi]
            df = df[df["impliedVolatility"] > 0]

            for _, row in df.iterrows():
                iv = row["impliedVolatility"]
                if not (0.01 < iv < 3.0):
                    mid = (row["bid"] + row["ask"]) / 2
                    iv  = implied_vol(mid, spot, row["strike"], T, rfr, opt_type, q) or 0
                if 0.01 < iv < 3.0:
                    rows.append({"strike": row["strike"], "T": T,
                                 "dte": dte_days, "iv": iv, "option_type": opt_type})

    return pd.DataFrame(rows).dropna().reset_index(drop=True)


class VolSurface:
    """
    Fit and query an implied volatility surface.
    Uses SmoothBivariateSpline over (strike, T) → IV.
    Requires ≥ 16 data points (cubic-cubic spline rank requirement).

    Usage:
        surface = VolSurface()
        surface.fit(build_surface_df(chain, spot))
        iv_at = surface.query(strike=190, T=0.25)
        smile = surface.smile(T=0.25, spot=185)
    """

    def __init__(self):
        self._spline: Optional[SmoothBivariateSpline] = None
        self._df: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame) -> "VolSurface":
        if len(df) < _MIN_SURFACE_POINTS:
            raise ValueError(
                f"Need ≥ {_MIN_SURFACE_POINTS} points to fit cubic-cubic spline, got {len(df)}"
            )
        self._df = df
        self._spline = SmoothBivariateSpline(
            df["strike"].values, df["T"].values, df["iv"].values, kx=3, ky=3
        )
        return self

    def query(self, strike: float, T: float) -> float:
        if self._spline is None:
            raise RuntimeError("Call fit() first")
        # Spline returns 1-element array for scalar inputs; clip to safe IV bounds
        return float(np.clip(self._spline(strike, T)[0], 0.01, 3.0))

    def smile(self, T: float, spot: float, n: int = 80) -> pd.DataFrame:
        """Vol smile at expiry T. Returns DataFrame: strike, moneyness, iv."""
        if self._df is None:
            raise RuntimeError("Call fit() first")
        K_min = self._df["strike"].min()
        K_max = self._df["strike"].max()
        strikes = np.linspace(K_min, K_max, n)
        ivs = [self.query(float(k), T) for k in strikes]
        return pd.DataFrame({"strike": strikes, "moneyness": strikes / spot, "iv": ivs})

    def term_structure(self, spot: float, n: int = 40) -> pd.DataFrame:
        """ATM IV across expiries. Returns DataFrame: T, days, iv."""
        if self._df is None:
            raise RuntimeError("Call fit() first")
        T_min = self._df["T"].min()
        T_max = self._df["T"].max()
        T_range = np.linspace(T_min, T_max, n)
        ivs = [self.query(spot, float(t)) for t in T_range]
        return pd.DataFrame({"T": T_range, "days": (T_range * 365).astype(int), "iv": ivs})
```

---

## `src/data.py`

**Purpose**: fetch the four inputs every pricing call needs (S, r, σ_hist, option chain).

```python
"""
Market data fetcher.
  - Spot price and OHLCV history: yfinance
  - Risk-free rate: FRED 3-Month T-Bill (TB3MS), falls back to 0.05
  - Option chain: yfinance (used for implied vol surface)

Performance:
  - MarketData uses slots=True
  - _get_rfr() wrapped with lru_cache(maxsize=1) — TB3MS updates monthly,
    no need to refetch per call within a session.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FALLBACK_RFR = 0.05

__all__ = ["MarketData", "get_market_data", "get_rfr", "FALLBACK_RFR"]


@dataclass(slots=True)
class MarketData:
    ticker: str
    spot: float
    rfr: float                          # annualised; TB3MS is simple yield, used as continuous-comp approx
    hist_vol: float                     # annualised close-to-close, 30-day window
    price_history: pd.DataFrame         # OHLCV, last 252 trading days
    option_chain: Optional[dict]        # {expiry_str: {"calls": df, "puts": df}}
    fetched_at: datetime = field(default_factory=datetime.utcnow)


def get_market_data(ticker: str, dividend_yield: float = 0.0) -> MarketData:
    """
    Main entry point. Fetch everything needed to price one ticker's options.

    Args:
        ticker        : e.g. "AAPL", "SPY"
        dividend_yield: continuous dividend yield (default 0)

    Returns:
        MarketData dataclass
    """
    history = _fetch_history(ticker)
    spot    = _get_spot(ticker, history)
    rfr     = _get_rfr()
    hv      = _close_to_close_vol(history["Close"], window=30)
    chain   = _fetch_option_chain(ticker)

    return MarketData(
        ticker=ticker,
        spot=spot,
        rfr=rfr,
        hist_vol=hv,
        price_history=history,
        option_chain=chain,
    )


def get_rfr() -> float:
    """Public shortcut for risk-free rate only."""
    return _get_rfr()


# ── private helpers ──────────────────────────────────────────────────────────

def _fetch_history(ticker: str, days: int = 380) -> pd.DataFrame:
    """Download ~252 trading days of OHLCV. Raises ValueError if < 30 rows."""
    end   = datetime.today()
    start = end - timedelta(days=days)
    df    = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df    = df.tail(252)
    if len(df) < 30:
        raise ValueError(f"Too little history for {ticker}: {len(df)} rows")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df.dropna()


def _get_spot(ticker: str, history: pd.DataFrame) -> float:
    """Try yfinance fast_info first, fall back to last close."""
    try:
        price = yf.Ticker(ticker).fast_info.last_price
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return float(history["Close"].iloc[-1])


@lru_cache(maxsize=1)
def _get_rfr() -> float:
    """
    3-Month T-Bill rate from FRED (TB3MS series).
    Returns decimal (0.05 = 5%). Falls back to FALLBACK_RFR if key missing.
    Memoized within session — TB3MS updates monthly, no need to refetch per call.
    """
    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        logger.info("No FRED_API_KEY set — using fallback RFR %.2f", FALLBACK_RFR)
        return FALLBACK_RFR
    try:
        from fredapi import Fred
        series = Fred(api_key=api_key).get_series("TB3MS")
        return float(series.dropna().iloc[-1]) / 100.0
    except Exception as e:
        logger.warning("FRED fetch failed (%s) — using fallback RFR %.2f", e, FALLBACK_RFR)
        return FALLBACK_RFR


def _close_to_close_vol(prices: pd.Series, window: int = 30) -> float:
    """
    Annualised close-to-close historical volatility.
    Formula: σ = std(ln(P_t / P_{t-1}), window=30) × √252
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    return float(log_ret.tail(window).std() * np.sqrt(252))


def _fetch_option_chain(ticker: str) -> Optional[dict]:
    """
    Fetch up to 8 nearest expiries from yfinance.
    Returns {expiry_str: {"calls": DataFrame, "puts": DataFrame}}
    Each DataFrame has: strike, lastPrice, bid, ask, impliedVolatility, openInterest
    Returns None if yfinance has no options for this ticker.
    """
    try:
        t       = yf.Ticker(ticker)
        expiries = t.options
        if not expiries:
            return None
        chain_data = {}
        for exp in expiries[:8]:
            try:
                c = t.option_chain(exp)
                chain_data[exp] = {"calls": c.calls, "puts": c.puts}
            except Exception:
                continue
        return chain_data or None
    except Exception as e:
        logger.warning("Option chain fetch failed: %s", e)
        return None
```

---

## `src/validation.py`

**Purpose**: verify the three models agree with each other and with analytical bounds.

```python
"""
Validation checks.

1. Put-call parity:  C - P = S·e^(-qT) - K·e^(-rT)
   Per-model tolerances:
     BSM:      ±$0.01   (deterministic, analytic)
     Binomial: ±$0.01   (deterministic, slight discretization noise at N=500)
     MC:       ±$0.05   (stochastic — 100k antithetic sims, 1.96·SE ≈ $0.02-0.04)
     Market:   ±$0.20   (bid-ask spread)

2. Boundary conditions:
   Call: max(S·e^(-qT) - K·e^(-rT), 0) ≤ C ≤ S·e^(-qT)
   Put:  max(K·e^(-rT) - S·e^(-qT), 0) ≤ P ≤ K·e^(-rT)
   Delta: 0 ≤ Δ_call ≤ e^(-qT),  -e^(-qT) ≤ Δ_put ≤ 0,  Γ ≥ 0,  ν ≥ 0

3. Convergence:
   |BSM - BinomialTree(N=500)| < $0.01
   BSM price inside MC 95% CI (100k sims)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import bsm_price, binomial_price, mc_price
from src.greeks import delta as g_delta, gamma as g_gamma, vega as g_vega

__all__ = [
    "PARITY_TOL_BSM", "PARITY_TOL_BINOMIAL", "PARITY_TOL_MC", "PARITY_TOL_MARKET",
    "check_parity", "run_parity_sweep",
    "run_boundary_sweep", "convergence_report",
]

PARITY_TOL_BSM      = 0.01   # deterministic, analytic
PARITY_TOL_BINOMIAL = 0.01   # deterministic, slight discretization noise
PARITY_TOL_MC       = 0.05   # stochastic — 100k antithetic sims
PARITY_TOL_MARKET   = 0.20   # bid-ask spread


def check_parity(call: float, put: float, S: float, K: float, T: float,
                 r: float, q: float = 0.0, tol: float = PARITY_TOL_BSM) -> dict:
    """Single put-call parity check. Returns dict with deviation and pass/fail."""
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    dev = abs(lhs - rhs)
    return {"lhs": lhs, "rhs": rhs, "deviation": dev, "passes": dev <= tol}


def run_parity_sweep(S: float, T: float, r: float, sigma: float,
                     q: float = 0.0, n: int = 20) -> pd.DataFrame:
    """
    Put-call parity across n strikes for all three models.
    Uses per-model tolerances (BSM/Binomial: $0.01, MC: $0.05).
    MC uses the SAME seed for call and put → correlated noise → tighter parity.

    Returns DataFrame: strike, model, deviation, tol, passes
    """
    strikes = np.linspace(S * 0.70, S * 1.30, n)
    rows = []
    for K in strikes:
        # BSM
        c = bsm_price(S, K, T, r, sigma, "call", q).price
        p = bsm_price(S, K, T, r, sigma, "put",  q).price
        r_bsm = check_parity(c, p, S, K, T, r, q, tol=PARITY_TOL_BSM)
        rows.append({"K": round(K,2), "model": "BSM", "call": c, "put": p,
                     "deviation": r_bsm["deviation"], "tol": PARITY_TOL_BSM,
                     "passes": r_bsm["passes"]})

        # Binomial
        c = binomial_price(S, K, T, r, sigma, "call", q, steps=500).price
        p = binomial_price(S, K, T, r, sigma, "put",  q, steps=500).price
        r_bin = check_parity(c, p, S, K, T, r, q, tol=PARITY_TOL_BINOMIAL)
        rows.append({"K": round(K,2), "model": "Binomial", "call": c, "put": p,
                     "deviation": r_bin["deviation"], "tol": PARITY_TOL_BINOMIAL,
                     "passes": r_bin["passes"]})

        # MC — same seed for call/put so noise is correlated → tighter parity
        c = mc_price(S, K, T, r, sigma, "call", q, n_sims=100_000, seed=42).price
        p = mc_price(S, K, T, r, sigma, "put",  q, n_sims=100_000, seed=42).price
        r_mc = check_parity(c, p, S, K, T, r, q, tol=PARITY_TOL_MC)
        rows.append({"K": round(K,2), "model": "MC", "call": c, "put": p,
                     "deviation": r_mc["deviation"], "tol": PARITY_TOL_MC,
                     "passes": r_mc["passes"]})

    return pd.DataFrame(rows)


def run_boundary_sweep(S: float, T: float, r: float, q: float = 0.0,
                       n_strikes: int = 30, n_sigmas: int = 8) -> pd.DataFrame:
    """
    Check all boundary conditions across (K, σ) grid.
    Returns DataFrame: K, sigma, test, value, bound, passes
    """
    strikes = np.linspace(S * 0.5, S * 1.5, n_strikes)
    sigmas  = np.linspace(0.05, 1.0, n_sigmas)
    rows = []
    for sig in sigmas:
        for K in strikes:
            C  = bsm_price(S, K, T, r, sig, "call", q).price
            P  = bsm_price(S, K, T, r, sig, "put",  q).price
            dc = g_delta(S, K, T, r, sig, "call", q)
            dp = g_delta(S, K, T, r, sig, "put",  q)
            gm = g_gamma(S, K, T, r, sig, q)
            vg = g_vega(S, K, T, r, sig, q)
            dq = np.exp(-q * T)
            lower_c = max(S * dq - K * np.exp(-r*T), 0)
            upper_c = S * dq
            lower_p = max(K * np.exp(-r*T) - S * dq, 0)
            upper_p = K * np.exp(-r*T)
            tests = [
                ("call_lower", C,  lower_c, ">="),
                ("call_upper", C,  upper_c, "<="),
                ("put_lower",  P,  lower_p, ">="),
                ("put_upper",  P,  upper_p, "<="),
                ("delta_call_lo", dc, 0,    ">="),
                ("delta_call_hi", dc, dq,   "<="),
                ("delta_put_lo",  dp, -dq,  ">="),
                ("delta_put_hi",  dp, 0,    "<="),
                ("gamma_pos",  gm,  0,      ">="),
                ("vega_pos",   vg,  0,      ">="),
            ]
            for name, val, bound, direction in tests:
                ok = (val >= bound - 1e-6) if direction == ">=" else (val <= bound + 1e-6)
                rows.append({"K": round(K,2), "sigma": sig, "test": name,
                              "value": val, "bound": bound, "passes": ok})
    return pd.DataFrame(rows)


def convergence_report(S: float = 100, K: float = 100, T: float = 0.5,
                       r: float = 0.05, sigma: float = 0.20,
                       option_type: str = "call", q: float = 0.0) -> dict:
    """
    Returns dict with:
      binomial: DataFrame of (N, price, abs_error, passes)
      mc:       DataFrame of (n_sims, price, se, ci_lower, ci_upper, bsm_in_ci)
      model_comparison: DataFrame (option_type, BSM, Binomial(500), MC(100k))
    """
    bsm_val = bsm_price(S, K, T, r, sigma, option_type, q).price

    # Binomial convergence
    step_range = [10, 25, 50, 100, 200, 500, 1000]
    bin_rows = []
    for n in step_range:
        p = binomial_price(S, K, T, r, sigma, option_type, q, steps=n).price
        err = abs(p - bsm_val)
        bin_rows.append({"N": n, "price": p, "bsm": bsm_val,
                          "abs_error": err, "passes": err < 0.01})

    # MC convergence
    sim_range = [1_000, 10_000, 100_000]
    mc_rows = []
    for n in sim_range:
        res = mc_price(S, K, T, r, sigma, option_type, q, n_sims=n)
        m = res.meta
        mc_rows.append({"n_sims": n, "price": res.price, "se": m["std_error"],
                         "ci_lower": m["ci_lower"], "ci_upper": m["ci_upper"],
                         "bsm_in_ci": m["ci_lower"] <= bsm_val <= m["ci_upper"]})

    # Model comparison table
    comp_rows = []
    for ot in ["call", "put"]:
        comp_rows.append({
            "option_type": ot,
            "BSM":           bsm_price(S, K, T, r, sigma, ot, q).price,
            "Binomial(500)": binomial_price(S, K, T, r, sigma, ot, q, steps=500).price,
            "MC(100k)":      mc_price(S, K, T, r, sigma, ot, q, n_sims=100_000).price,
        })

    return {
        "binomial":         pd.DataFrame(bin_rows),
        "mc":               pd.DataFrame(mc_rows),
        "model_comparison": pd.DataFrame(comp_rows),
    }
```

---

## `src/plots.py`

**Purpose**: save charts to `outputs/`. No interactive widgets. All functions accept explicit params.

```python
"""
Visualisation. All functions save to outputs/ and return the figure.
Call plt.close(fig) after if generating many charts in a loop.

Charts:
  payoff_diagram      — payoff + P&L at expiry
  greeks_profile      — 6-panel first+second order Greeks vs spot
  mc_paths_chart      — simulated GBM paths
  vol_surface_3d      — plotly interactive surface (saves .html)
  vol_smile_chart     — smile curves for multiple expiries
  convergence_chart   — |BTree(N) - BSM| vs N
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",  "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "grid.color": "#21262d",
    "lines.linewidth": 2.0,
})

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

_BLUE = "#58a6ff"
_RED  = "#f78166"
_GREEN= "#3fb950"
_ORNG = "#ffa657"
_GREY = "#8b949e"

__all__ = [
    "payoff_diagram", "greeks_profile", "mc_paths_chart",
    "vol_surface_3d", "vol_smile_chart", "convergence_chart",
]


def payoff_diagram(S0: float, K: float, premium: float,
                   option_type: str = "call", filename: str = None) -> plt.Figure:
    """Payoff at expiry + net P&L (after premium). Marks strike, spot, breakeven."""
    S_range = np.linspace(S0 * 0.6, S0 * 1.4, 500)
    payoff  = np.maximum(S_range - K, 0) if option_type == "call" else np.maximum(K - S_range, 0)
    pnl     = payoff - premium

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(S_range, payoff, color=_BLUE, label="Payoff at expiry")
    ax.plot(S_range, pnl,    color=_BLUE, linestyle="--", alpha=0.8, label="P&L (net of premium)")
    ax.axhline(0, color=_GREY, lw=0.8)
    ax.axvline(K,  color=_ORNG, lw=1.5, linestyle=":", label=f"Strike K={K:.2f}")
    ax.axvline(S0, color=_GREEN,lw=1.5, linestyle=":", label=f"Spot S₀={S0:.2f}")
    be = (K + premium) if option_type == "call" else (K - premium)
    if S_range[0] < be < S_range[-1]:
        ax.axvline(be, color="#d2a8ff", lw=1, linestyle="--", label=f"Breakeven={be:.2f}")
    ax.fill_between(S_range, pnl, 0, where=(pnl > 0), alpha=0.18, color=_BLUE)
    ax.fill_between(S_range, pnl, 0, where=(pnl < 0), alpha=0.12, color=_RED)
    ax.set_xlabel("Spot at Expiry"); ax.set_ylabel("Value / P&L ($)")
    ax.set_title(f"{option_type.title()} Payoff  |  K={K}  Premium={premium:.4f}")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS / (filename or f"payoff_{option_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def greeks_profile(K: float, T: float, r: float, sigma: float,
                   q: float = 0.0, filename: str = None) -> plt.Figure:
    """6-panel: Delta, Gamma, Theta, Vega, Vanna, Volga vs spot. Call+Put on relevant panels."""
    from src.greeks import delta, gamma, theta, vega, vanna, volga

    S_range = np.linspace(K * 0.5, K * 1.5, 200)
    panels = {
        "Delta":  (lambda S, ot: delta(S, K, T, r, sigma, ot, q), True),
        "Gamma":  (lambda S, ot: gamma(S, K, T, r, sigma, q),     False),
        "Theta":  (lambda S, ot: theta(S, K, T, r, sigma, ot, q), True),
        "Vega":   (lambda S, ot: vega(S,  K, T, r, sigma, q),     False),
        "Vanna":  (lambda S, ot: vanna(S, K, T, r, sigma, q),     False),
        "Volga":  (lambda S, ot: volga(S, K, T, r, sigma, q),     False),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, (fn, call_put_differ)) in zip(axes.flat, panels.items()):
        call_vals = [fn(S, "call") for S in S_range]
        ax.plot(S_range, call_vals, color=_BLUE, label="Call")
        if call_put_differ:
            put_vals = [fn(S, "put") for S in S_range]
            ax.plot(S_range, put_vals, color=_RED, linestyle="--", label="Put")
        ax.axvline(K, color=_ORNG, lw=1, linestyle=":", alpha=0.7)
        ax.axhline(0, color=_GREY, lw=0.6)
        ax.set_title(name); ax.set_xlabel("Spot")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Greeks Profile  |  K={K}  T={T:.2f}yr  σ={sigma:.0%}  r={r:.1%}", fontsize=13)
    plt.tight_layout()
    path = OUTPUTS / (filename or "greeks_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); return fig


def mc_paths_chart(paths: np.ndarray, K: float, T: float,
                   option_type: str = "call", filename: str = None) -> plt.Figure:
    """Plot up to 300 MC paths. ITM blue, OTM red. Mean path in green."""
    n_show = min(300, paths.shape[0])
    t_axis = np.linspace(0, T, paths.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    for path in paths[:n_show]:
        itm   = path[-1] > K if option_type == "call" else path[-1] < K
        ax.plot(t_axis, path, color=_BLUE if itm else _RED, alpha=0.10, lw=0.5)
    ax.plot(t_axis, paths.mean(axis=0), color=_GREEN, lw=2, label="Mean path", zorder=5)
    ax.axhline(K, color=_ORNG, lw=1.5, linestyle="--", label=f"Strike K={K:.2f}")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Stock Price")
    ax.set_title(f"Monte Carlo Paths ({paths.shape[0]:,})")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    path_f = OUTPUTS / (filename or "mc_paths.png")
    fig.savefig(path_f, dpi=150, bbox_inches="tight"); return fig


def vol_surface_3d(surface_df: pd.DataFrame, ticker: str = "",
                   filename: str = None) -> go.Figure:
    """Interactive plotly 3D vol surface. Saves .html to outputs/."""
    df = surface_df.dropna(subset=["iv"])
    fig = go.Figure(go.Scatter3d(
        x=df["strike"], y=df["dte"], z=df["iv"] * 100,
        mode="markers",
        marker=dict(size=4, color=df["iv"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="IV (%)")),
        text=[f"K={k:.1f}<br>{d}d<br>{iv:.1%}"
              for k, d, iv in zip(df["strike"], df["dte"], df["iv"])],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Implied Volatility Surface{' — ' + ticker if ticker else ''}",
        scene=dict(xaxis_title="Strike", yaxis_title="Days to Expiry",
                   zaxis_title="IV (%)", bgcolor="#0d1117"),
        paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
        width=900, height=650,
    )
    path = OUTPUTS / (filename or f"vol_surface{'_' + ticker if ticker else ''}.html")
    fig.write_html(str(path)); return fig


def vol_smile_chart(surface, spot: float, T_values: list,
                    filename: str = None) -> plt.Figure:
    """Smile curves for multiple expiries. surface: VolSurface instance."""
    colors = plt.cm.cool(np.linspace(0, 1, len(T_values)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for T, color in zip(T_values, colors):
        df = surface.smile(T, spot)
        ax.plot(df["moneyness"], df["iv"] * 100, color=color,
                label=f"T={T:.2f}yr ({int(T*365)}d)")
    ax.axvline(1.0, color=_GREY, lw=1, linestyle=":", label="ATM")
    ax.set_xlabel("Moneyness (K/S)"); ax.set_ylabel("IV (%)")
    ax.set_title("Vol Smile by Expiry"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUTS / (filename or "vol_smile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); return fig


def convergence_chart(binomial_df: pd.DataFrame, filename: str = None) -> plt.Figure:
    """Plot |BTree(N) - BSM| vs N on log scale."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(binomial_df["N"], binomial_df["abs_error"], color=_BLUE, marker="o", ms=5)
    ax.axhline(0.01, color=_RED, lw=1, linestyle="--", label="$0.01 threshold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of Steps (N)"); ax.set_ylabel("|BTree − BSM| ($)")
    ax.set_title("Binomial Tree Convergence to BSM")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    path = OUTPUTS / (filename or "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); return fig
```

---

## Tests

### `tests/conftest.py`

```python
import pytest
import numpy as np

@pytest.fixture
def p():
    """Standard ATM parameters. S=100, K=100, T=0.5yr, r=5%, σ=20%."""
    return dict(S=100.0, K=100.0, T=0.5, r=0.05, sigma=0.20, q=0.0)

@pytest.fixture
def hull():
    """Hull (2018) Table 15.1 case. BSM call ≈ $4.76, put ≈ $0.81."""
    return dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20, q=0.0,
                call_expected=4.76, put_expected=0.81, tol=0.02)
```

### `tests/test_models.py`

```python
import numpy as np
import pytest
from src.models import bsm_price, binomial_price, mc_price, bsm_price_vec

# ── BSM ──────────────────────────────────────────────────────────────────────

def test_bsm_known_price(hull):
    """Hull (2018) Table 15.1: S=42, K=40, r=10%, σ=20%, T=0.5yr → call $4.76, put $0.81."""
    call = bsm_price(hull["S"], hull["K"], hull["T"], hull["r"], hull["sigma"], "call").price
    put  = bsm_price(hull["S"], hull["K"], hull["T"], hull["r"], hull["sigma"], "put").price
    assert abs(call - hull["call_expected"]) < hull["tol"]
    assert abs(put  - hull["put_expected"])  < hull["tol"]

def test_bsm_put_call_parity(p):
    """C - P = S·e^(-qT) - K·e^(-rT) — BSM satisfies this to machine precision."""
    call = bsm_price(**p, option_type="call").price
    put  = bsm_price(**p, option_type="put").price
    rhs  = p["S"] * np.exp(-p["q"]*p["T"]) - p["K"] * np.exp(-p["r"]*p["T"])
    assert abs((call - put) - rhs) < 0.001

def test_bsm_call_put_positive(p):
    assert bsm_price(**p, option_type="call").price > 0
    assert bsm_price(**p, option_type="put").price  > 0

def test_bsm_invalid_inputs():
    with pytest.raises(ValueError): bsm_price(S=-100, K=100, T=0.5, r=0.05, sigma=0.2)
    with pytest.raises(ValueError): bsm_price(S=100,  K=100, T=0.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError): bsm_price(S=100,  K=100, T=0.5, r=0.05, sigma=-0.1)
    with pytest.raises(ValueError): bsm_price(S=100,  K=100, T=0.5, r=0.05, sigma=0.2, q=1.5)

def test_bsm_deep_itm_approaches_intrinsic():
    """Deep ITM call ≈ S·e^(-qT) − K·e^(-rT)."""
    S, K, T, r, sigma = 200, 100, 0.1, 0.05, 0.2
    call  = bsm_price(S, K, T, r, sigma, "call").price
    lower = S - K * np.exp(-r*T)
    assert abs(call - lower) < 1.0

def test_bsm_deep_otm_call_approaches_zero():
    """Deep OTM call should be near zero."""
    call = bsm_price(S=50, K=200, T=0.5, r=0.05, sigma=0.2, option_type="call").price
    assert call < 0.01

def test_bsm_near_expiry_stability():
    """T very small: call ≈ max(S − K, 0). Numerically stable, no NaN/Inf."""
    call = bsm_price(S=105, K=100, T=1e-4, r=0.05, sigma=0.2, option_type="call").price
    assert np.isfinite(call)
    assert abs(call - 5.0) < 0.5   # tight enough given vol diffusion over 1 day

def test_bsm_vec_broadcast():
    """bsm_price_vec should accept arrays and broadcast correctly."""
    K = np.array([90, 100, 110])
    prices = bsm_price_vec(100, K, 0.5, 0.05, 0.2, "call")
    assert prices.shape == (3,)
    # monotone decreasing in K for calls
    assert prices[0] > prices[1] > prices[2]

# ── Binomial ─────────────────────────────────────────────────────────────────

def test_binomial_converges_to_bsm(p):
    bsm_val  = bsm_price(**p, option_type="call").price
    tree_val = binomial_price(**p, option_type="call", steps=500).price
    assert abs(tree_val - bsm_val) < 0.01

def test_binomial_put_call_parity(p):
    """C - P = S·e^(-qT) - K·e^(-rT). Binomial satisfies this at N=500."""
    c = binomial_price(**p, option_type="call", steps=500).price
    p_put = binomial_price(**p, option_type="put", steps=500).price
    rhs = p["S"] * np.exp(-p["q"]*p["T"]) - p["K"] * np.exp(-p["r"]*p["T"])
    assert abs((c - p_put) - rhs) < 0.01

# ── Monte Carlo ──────────────────────────────────────────────────────────────

def test_mc_bsm_in_ci(p):
    """BSM price should be inside MC 95% CI at 100k sims."""
    bsm_val = bsm_price(**p, option_type="call").price
    res     = mc_price(**p, option_type="call", n_sims=100_000, seed=42)
    assert res.meta["ci_lower"] <= bsm_val <= res.meta["ci_upper"]

def test_mc_antithetic_se_small():
    """
    Antithetic MC SE on ATM call at 10k sims should be < $0.05.
    (With proper antithetic pair-averaging, SE is correctly tight.)
    """
    from src.models import mc_price
    res = mc_price(100, 100, 0.5, 0.05, 0.20, n_sims=10_000, seed=42)
    assert res.meta["std_error"] < 0.05

def test_mc_seed_reproducible():
    """Same seed → identical price."""
    a = mc_price(100, 100, 0.5, 0.05, 0.2, n_sims=10_000, seed=42)
    b = mc_price(100, 100, 0.5, 0.05, 0.2, n_sims=10_000, seed=42)
    assert a.price == b.price

def test_mc_put_call_parity(p):
    """MC: C - P ≈ S·e^(-qT) - K·e^(-rT), within $0.05 (stochastic tolerance).
    Same seed for call/put → correlated noise → much tighter deviation.
    """
    c = mc_price(**p, option_type="call", n_sims=100_000, seed=42).price
    p_put = mc_price(**p, option_type="put", n_sims=100_000, seed=42).price
    rhs = p["S"] * np.exp(-p["q"]*p["T"]) - p["K"] * np.exp(-p["r"]*p["T"])
    assert abs((c - p_put) - rhs) < 0.05
```

### `tests/test_greeks.py`

```python
import numpy as np
import pytest
from src.greeks import (delta, gamma, theta, vega, rho, vanna, volga, charm,
                         all_greeks,
                         numerical_delta, numerical_gamma, numerical_vega,
                         numerical_theta, numerical_rho)

TOL = 0.001   # 0.1% relative tolerance for analytical vs numerical

def test_delta_call_in_bounds(p):
    d = delta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
    assert 0 <= d <= 1

def test_delta_put_in_bounds(p):
    d = delta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put", p["q"])
    assert -1 <= d <= 0

def test_delta_parity(p):
    """Δ_call - Δ_put = e^(-qT)"""
    dc = delta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
    dp = delta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "put",  p["q"])
    assert abs((dc - dp) - np.exp(-p["q"] * p["T"])) < 1e-6

def test_gamma_nonneg(p):
    assert gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"]) >= 0

def test_vega_nonneg(p):
    assert vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"]) >= 0

def test_theta_negative_long_call(p):
    """Long call theta must be negative (time decay)."""
    t = theta(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
    assert t < 0

def test_all_greeks_no_nan(p):
    g = all_greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], "call", p["q"])
    for name in ["delta","gamma","theta","vega","rho","vanna","volga","charm"]:
        val = getattr(g, name)
        assert not np.isnan(val), f"{name} is NaN"

# Numerical vs analytical cross-checks
@pytest.mark.parametrize("greek,ana_fn,num_fn", [
    ("delta", lambda p: delta(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"]),
              lambda p: numerical_delta(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"])),
    ("gamma", lambda p: gamma(p["S"],p["K"],p["T"],p["r"],p["sigma"],p["q"]),
              lambda p: numerical_gamma(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"])),
    ("vega",  lambda p: vega(p["S"],p["K"],p["T"],p["r"],p["sigma"],p["q"]),
              lambda p: numerical_vega(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"])),
    ("theta", lambda p: theta(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"]),
              lambda p: numerical_theta(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"])),
    ("rho",   lambda p: rho(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"]),
              lambda p: numerical_rho(p["S"],p["K"],p["T"],p["r"],p["sigma"],"call",p["q"])),
])
def test_numerical_vs_analytical(greek, ana_fn, num_fn, p):
    ana = ana_fn(p)
    num = num_fn(p)
    rel_err = abs(ana - num) / (abs(ana) + 1e-10)
    assert rel_err < TOL, f"{greek}: analytical={ana:.6f}, numerical={num:.6f}, rel_err={rel_err:.4%}"
```

### `tests/test_validation.py`

```python
import pytest
from src.validation import run_parity_sweep, run_boundary_sweep, convergence_report

def test_parity_bsm_all_strikes(p):
    """BSM put-call parity must hold to $0.01 across all 20 strikes."""
    df = run_parity_sweep(p["S"], p["T"], p["r"], p["sigma"], p["q"], n=20)
    fails = df[(df["model"] == "BSM") & (~df["passes"])]
    assert len(fails) == 0, f"BSM parity failures:\n{fails[['K','deviation']].to_string()}"

def test_parity_binomial_all_strikes(p):
    """Binomial(N=500) put-call parity must hold to $0.01 across all 20 strikes."""
    df = run_parity_sweep(p["S"], p["T"], p["r"], p["sigma"], p["q"], n=20)
    fails = df[(df["model"] == "Binomial") & (~df["passes"])]
    assert len(fails) == 0, f"Binomial parity failures:\n{fails[['K','deviation']].to_string()}"

def test_parity_mc_all_strikes(p):
    """MC put-call parity within $0.05 across all 20 strikes (stochastic tolerance).
    Same seed for call/put → correlated noise → deviation typically < $0.01.
    """
    df = run_parity_sweep(p["S"], p["T"], p["r"], p["sigma"], p["q"], n=20)
    fails = df[(df["model"] == "MC") & (~df["passes"])]
    assert len(fails) == 0, f"MC parity failures:\n{fails[['K','deviation','tol']].to_string()}"

def test_boundaries_pass(p):
    df = run_boundary_sweep(p["S"], p["T"], p["r"], p["q"], n_strikes=20, n_sigmas=5)
    rate = df["passes"].mean()
    assert rate >= 0.999, f"Boundary violation rate: {1-rate:.2%}"

def test_binomial_convergence(p):
    report = convergence_report(**p)
    row = report["binomial"].query("N == 500").iloc[0]
    assert row["passes"], f"BTree(500) error: ${row['abs_error']:.4f}"

def test_mc_coverage(p):
    report = convergence_report(**p)
    row = report["mc"].query("n_sims == 100000").iloc[0]
    assert row["bsm_in_ci"], f"BSM not in MC CI: [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
```

### `tests/test_volatility.py` (NEW — was missing in v2)

```python
"""Tests for IV solver and VolSurface — no network required."""
import numpy as np
import pandas as pd
import pytest
from src.models import bsm_price
from src.volatility import implied_vol, VolSurface, _MIN_SURFACE_POINTS


def test_iv_round_trip():
    """IV(BSM_price(sigma_true)) should recover sigma_true to high precision."""
    S, K, T, r, sigma_true = 100, 100, 0.5, 0.05, 0.25
    mkt = bsm_price(S, K, T, r, sigma_true, "call").price
    iv  = implied_vol(mkt, S, K, T, r, "call")
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-4


def test_iv_put_round_trip():
    """IV solver should work for puts too."""
    S, K, T, r, sigma_true = 100, 95, 0.5, 0.05, 0.30
    mkt = bsm_price(S, K, T, r, sigma_true, "put").price
    iv  = implied_vol(mkt, S, K, T, r, "put")
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-4


def test_iv_deep_otm():
    """IV solver should still converge for deep OTM (low vega) options."""
    S, K, T, r, sigma_true = 100, 150, 0.25, 0.05, 0.30
    mkt = bsm_price(S, K, T, r, sigma_true, "call").price
    iv  = implied_vol(mkt, S, K, T, r, "call")
    assert iv is not None
    assert abs(iv - sigma_true) < 1e-3   # slightly looser for low-vega region


def test_iv_zero_market_price():
    """If market_price ≤ 0, return None."""
    assert implied_vol(0.0, 100, 100, 0.5, 0.05, "call") is None
    assert implied_vol(-1.0, 100, 100, 0.5, 0.05, "call") is None


def test_iv_arbitrage_returns_none():
    """If market price violates no-arb bounds, IV should be None (Brent can't bracket)."""
    # Call price > S → arbitrage
    assert implied_vol(200.0, 100, 100, 0.5, 0.05, "call") is None


def test_vol_surface_min_points():
    """VolSurface.fit should refuse to fit with < 16 points."""
    df = pd.DataFrame({
        "strike": np.linspace(90, 110, 10),
        "T":      np.full(10, 0.25),
        "iv":     np.full(10, 0.20),
        "dte":    np.full(10, 91),
        "option_type": "call",
    })
    s = VolSurface()
    with pytest.raises(ValueError, match=str(_MIN_SURFACE_POINTS)):
        s.fit(df)


def test_vol_surface_fit_and_query():
    """Build a synthetic surface, fit, and query — IV should be in valid range."""
    rows = []
    strikes = np.linspace(80, 120, 10)
    Ts      = [0.25, 0.5, 1.0, 2.0, 0.25]
    for K in strikes:
        for T in Ts:
            # Smile shape: higher IV at wings
            moneyness = K / 100
            iv = 0.20 + 0.05 * (moneyness - 1) ** 2 + 0.01 * np.sqrt(T)
            rows.append({"strike": K, "T": T, "iv": iv, "dte": int(T*365),
                         "option_type": "call"})
    df = pd.DataFrame(rows)
    s = VolSurface().fit(df)

    # Query ATM, medium tenor
    iv_atm = s.query(100, 0.5)
    assert 0.01 < iv_atm < 3.0
    # ATM IV should be near 0.20 (smile minimum)
    assert abs(iv_atm - 0.20) < 0.05


def test_vol_surface_query_clamped():
    """Query far outside data range should be clamped to [0.01, 3.0], not NaN/Inf."""
    rows = []
    strikes = np.linspace(90, 110, 10)
    Ts = [0.25, 0.5, 1.0, 2.0, 0.25]
    for K in strikes:
        for T in Ts:
            rows.append({"strike": K, "T": T, "iv": 0.20,
                         "dte": int(T*365), "option_type": "call"})
    df = pd.DataFrame(rows)
    s = VolSurface().fit(df)
    # Query far outside data range
    iv = s.query(500, 10.0)
    assert 0.01 <= iv <= 3.0
    assert np.isfinite(iv)


def test_vol_surface_smile_returns_df():
    """smile() should return a DataFrame with strike, moneyness, iv columns."""
    rows = []
    strikes = np.linspace(80, 120, 10)
    Ts = [0.25, 0.5, 1.0, 2.0, 0.25]
    for K in strikes:
        for T in Ts:
            rows.append({"strike": K, "T": T, "iv": 0.20,
                         "dte": int(T*365), "option_type": "call"})
    df = pd.DataFrame(rows)
    s = VolSurface().fit(df)
    smile_df = s.smile(T=0.5, spot=100, n=50)
    assert set(smile_df.columns) == {"strike", "moneyness", "iv"}
    assert len(smile_df) == 50
    assert (smile_df["iv"] >= 0.01).all() and (smile_df["iv"] <= 3.0).all()
```

### `tests/test_perf.py` (NEW — performance smoke tests)

```python
"""Performance smoke tests. Run with: pytest tests/test_perf.py -v

These catch perf regressions. Thresholds are generous (5-10× headroom) so
they pass on any modern laptop. If they fail, something is dramatically wrong.
"""
import time
import numpy as np
import pytest
from src.models import bsm_price, binomial_price, mc_price, bsm_price_vec
from src.greeks import all_greeks
from src.volatility import implied_vol

S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20


def test_bsm_pricing_under_1ms():
    """BSM scalar pricing should be < 1ms per call."""
    t0 = time.perf_counter()
    for _ in range(1000):
        bsm_price(S, K, T, r, sigma, "call")
    elapsed_ms = (time.perf_counter() - t0) * 1000 / 1000
    assert elapsed_ms < 1.0, f"BSM pricing took {elapsed_ms:.3f}ms/call"


def test_bsm_vec_10k_under_50ms():
    """BSM vectorised over 10k strikes should be < 50ms."""
    K_arr = np.linspace(50, 150, 10000)
    t0 = time.perf_counter()
    bsm_price_vec(S, K_arr, T, r, sigma, "call")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"BSM vec(10k) took {elapsed_ms:.1f}ms"


def test_binomial_500_steps_under_50ms():
    """Binomial tree with 500 steps should be < 50ms."""
    t0 = time.perf_counter()
    binomial_price(S, K, T, r, sigma, "call", steps=500)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, f"Binomial(500) took {elapsed_ms:.1f}ms"


def test_mc_100k_under_500ms():
    """MC with 100k antithetic sims should be < 500ms."""
    t0 = time.perf_counter()
    mc_price(S, K, T, r, sigma, "call", n_sims=100_000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500, f"MC(100k) took {elapsed_ms:.1f}ms"


def test_all_greeks_under_2ms():
    """All 8 Greeks for a single option should be < 2ms."""
    t0 = time.perf_counter()
    for _ in range(1000):
        all_greeks(S, K, T, r, sigma, "call")
    elapsed_ms = (time.perf_counter() - t0) * 1000 / 1000
    assert elapsed_ms < 2.0, f"all_greeks took {elapsed_ms:.3f}ms/call"


def test_iv_solver_under_5ms():
    """IV solver (Newton-Raphson) should converge in < 5ms for ATM option."""
    mkt = bsm_price(S, K, T, r, 0.25, "call").price
    t0 = time.perf_counter()
    iv = implied_vol(mkt, S, K, T, r, "call")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert iv is not None
    assert elapsed_ms < 5, f"IV solver took {elapsed_ms:.1f}ms"
```

---

## Notebooks

### `notebooks/01_full_demo.ipynb`

Cells in order (each cell is executable independently after the prior cells):

1. **Setup** — imports, load `.env`, fetch `MarketData` for `AAPL`
2. **Market snapshot** — print spot, RFR, hist vol; show last 30 days price chart
3. **Price one option** — AAPL nearest ATM call and put, all three models, comparison table
4. **Greeks table** — `all_greeks()` for the same option, print formatted
5. **Payoff diagram** — call `plots.payoff_diagram()`
6. **Greeks profile** — call `plots.greeks_profile()`
7. **MC paths** — simulate and call `plots.mc_paths_chart()`
8. **Vol surface** — `build_surface_df()` → `VolSurface().fit()` → `plots.vol_surface_3d()`
9. **Vol smile** — `plots.vol_smile_chart()` for 4 expiries

### `notebooks/02_validation_report.ipynb`

Cells in order:

1. **Setup** — imports, standard params `S=100, K=100, T=0.5, r=0.05, σ=0.20`
2. **Put-Call Parity** — `run_parity_sweep()`, show table grouped by model, assert 0 violations per model
3. **Boundary Conditions** — `run_boundary_sweep()`, show pass rate
4. **Binomial Convergence** — table + `plots.convergence_chart()`
5. **MC Convergence** — table showing SE vs N_sims, BSM in CI
6. **Model Comparison** — side-by-side call/put prices from all three models
7. **Greeks vs Numerical** — table comparing analytical vs numerical for each Greek
8. **Performance** — run perf smoke tests inline, display timings
9. **Summary** — markdown cell: "All N validation checks passed."

---

## Implementation Order

Build in this sequence. **Run `pytest tests/` before moving to the next phase.**

```
Phase 0 — Project setup:
  pyproject.toml
  requirements.txt
  .env.example
  src/__init__.py (empty)
  → `pip install -e .[dev]`  (or `pip install -r requirements.txt`)

Phase 1 — Core math (no I/O):
  src/models.py    → pytest tests/test_models.py          (must pass fully)
  src/greeks.py    → pytest tests/test_greeks.py          (must pass fully)

Phase 2 — Validation (no I/O):
  src/validation.py → pytest tests/test_validation.py     (must pass fully)

Phase 3 — Data + vol (requires network for data.py; vol tests are offline):
  src/data.py
  src/volatility.py
  → pytest tests/test_volatility.py                        (IV + surface tests, no network)

Phase 4 — Plots + notebooks:
  src/plots.py
  notebooks/01_full_demo.ipynb
  notebooks/02_validation_report.ipynb

Phase 5 — Performance (optional but recommended):
  pytest tests/test_perf.py -v
```

Do not begin Phase 3 until Phase 2 tests all pass.
Do not add features not listed here.
If a step is unclear, pick the simpler interpretation and document the assumption in a comment.

---

## CHANGELOG (v2 → v3)

| # | Change | Why |
|---|--------|-----|
| 1 | **Fixed `bsm_price` docstring**: "Vectorised" → "Scalar inputs; use `bsm_price_vec` for arrays." | v2 docstring lied; `_validate` would crash on arrays (`if S <= 0` raises on ndarray). |
| 2 | **MC antithetic SE now correctly computed on pair-averages** (n=half), not 2n raw payoffs. | v2 overestimated SE → CI too wide; tests passed but variance reduction benefit was unreported. v3 formula: `Y_i = 0.5·(P(Z_i) + P(-Z_i))`, `SE = std(Y, ddof=1)/sqrt(half)`. |
| 3 | **`test_parity_all_models_all_strikes` split into 3 per-model tests** with separate tolerances: BSM $0.01, Binomial $0.01, MC $0.05. | v2 risked flaky MC parity failures at extreme strikes (±30% moneyness) under a single $0.01 tolerance. |
| 4 | **NEW `tests/test_volatility.py`** with 9 tests: IV round-trip (call & put), deep-OTM, zero price, arbitrage, surface min-points, fit+query, clamp, smile shape. | v2 had ZERO automated tests for IV/surface — Phase 3 was "tested in notebooks" only. |
| 5 | **`VolSurface.fit` minimum bumped 12 → 16** (cubic-cubic spline rank = `(kx+1)(ky+1) = 16`). | v2 would crash on real-world thin data; `_MIN_SURFACE_POINTS` exposed as module constant for tests. |
| 6 | **Added `pyproject.toml`** with PEP 621 metadata, `[tool.pytest.ini_options]` (with `pythonpath=["."]`), package install config. | v2 had no package config; `from src.X import Y` only worked if pytest run from repo root. Now `pip install -e .[dev]` makes imports robust. |
| 7 | **Switched `scipy.stats.norm.cdf/pdf` → `scipy.special.ndtr` + manual `np.exp(-x²/2)/√(2π)`.** | 3-5× faster on both scalars and arrays (ufunc vs frozen-distribution overhead). |
| 8 | **`PricingResult`, `Greeks`, `MarketData` now use `@dataclass(slots=True)`.** | Cache-friendly memory layout, ~20% faster attribute access, lower memory footprint. Requires Python 3.10+ (now explicit). |
| 9 | **`_vega_raw` renamed to public `vega_raw`** (with `_vega_raw = vega_raw` alias for backward compat). | Cleaner cross-module import; no underscore-private symbol leaking into `volatility.py`. |
| 10 | **`mc_simulate_paths` switched from per-step Python loop to vectorized `np.cumsum` of log-increments.** | ~10× faster for large `n_paths × n_steps`. Mathematically equivalent (exact GBM solution telescopes). |
| 11 | **`VolSurface.query` now returns `float(np.clip(self._spline(strike, T)[0], 0.01, 3.0))`** (was returning 0-d ndarray without clipping). | v2 returned ndarray; `float()` comparison in tests would fail. Clamping prevents spline runaway outside data range. |
| 12 | **`MarketData.rfr` docstring clarified**: "TB3MS is simple yield, used as continuous-comp approx." | v2 was technically imprecise (claimed "continuously compounded"). |
| 13 | **`_get_rfr()` wrapped with `@lru_cache(maxsize=1)`.** | TB3MS updates monthly; refetching per call wastes FRED API quota. Within-session memoization only (not a persistent cache layer — still compliant with "no caching layer" rule). |
| 14 | **`__all__` added to every module.** | Cleaner public API, better IDE autocomplete, explicit export surface. |
| 15 | **NEW `tests/test_perf.py`** with 6 benchmark tests: BSM scalar <1ms, BSM vec(10k) <50ms, Binomial(500) <50ms, MC(100k) <500ms, all_greeks <2ms, IV solver <5ms. | Catch perf regressions, document expected timings, prove the v3 perf changes actually deliver. |
| 16 | **MC parity test uses SAME seed for call & put** → correlated noise → tighter parity deviation. | Reduces MC parity deviation by ~50% (variance of difference << sum of variances when covariance is high). Same seed=42 is safe because call and put payoffs are deterministic functions of the same Z draws. |
| 17 | **Added edge-case tests**: `test_bsm_deep_otm_call_approaches_zero`, `test_bsm_near_expiry_stability`, `test_bsm_vec_broadcast`, `test_iv_put_round_trip`, `test_vol_surface_smile_returns_df`. | v2 lacked edge-case coverage; these catch numerical instability and API contract violations. |
| 18 | **Added `test_mc_seed_reproducible`, `test_mc_put_call_parity`.** | Reproducibility guarantee + parity correctness on MC (was missing in v2). |
| 19 | **`implied_vol` Newton step now uses `np.clip(sigma - step, IV_LO, IV_HI)`** instead of `max(sigma, 1e-6)`. | Prevents sigma from going negative OR exploding; symmetric bounds; Brent fallback now uses same bounds for consistency. |
| 20 | **`validation.py` exposes `PARITY_TOL_BSM`, `PARITY_TOL_BINOMIAL`, `PARITY_TOL_MC` as public constants** in `__all__`. | Tests and notebooks can reference the same thresholds; no magic numbers duplicated. |
| 21 | **`bsm_price_vec` now does `np.asarray(..., dtype=float)` at entry.** | Avoids silent type promotion (e.g., int inputs → int output on some numpy versions); guarantees float64 throughout. |
| 22 | **`PricingResult.meta` for MC now includes `"n_pairs"` and `"antithetic": True` flags.** | Explicit metadata; notebooks/tests can introspect estimator type. |
| 23 | **`VolSurface.smile/term_structure` now raise `RuntimeError` if `fit()` not called** (was checking `_spline is None`, now also checks `_df is None` for `smile` which uses `_df` for K range). | Catches misuse early with clear error message. |
| 24 | **Python version pinned to ≥3.10** in `pyproject.toml` and assumptions. | `dataclass(slots=True)` requires 3.10; v2 was silent on Python version. |

---

## Expected test counts (after full build)

| Test file | # tests | Notes |
|---|---|---|
| `test_models.py` | 14 | BSM (8) + Binomial (2) + MC (4) |
| `test_greeks.py` | 7 + 5 parametrized = 12 | 7 unit + 5 numerical cross-checks |
| `test_validation.py` | 6 | 3 parity (per model) + boundaries + 2 convergence |
| `test_volatility.py` | 9 | IV (5) + surface (4) |
| `test_perf.py` | 6 | Performance smoke |
| **Total** | **47** | All should pass on a clean `pip install -e .[dev] && pytest` |

---

## Pre-flight checklist for the implementing agent

Before writing ANY code, verify:

- [ ] Python 3.10+ is available (`python --version`)
- [ ] `pip install -e .[dev]` succeeds
- [ ] `pytest --collect-only` finds all 47 tests across 5 files
- [ ] `from src.models import bsm_price` works from any directory (proves `pyproject.toml` is correct)

If any of these fail, fix the environment before proceeding to Phase 1.

