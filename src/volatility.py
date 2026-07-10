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
        # .item() extracts scalar from 0-d array without numpy 1.25+ deprecation warning
        return float(np.clip(self._spline(strike, T).item(), 0.01, 3.0))

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