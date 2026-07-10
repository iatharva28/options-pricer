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