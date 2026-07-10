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