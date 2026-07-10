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