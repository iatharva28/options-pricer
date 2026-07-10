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