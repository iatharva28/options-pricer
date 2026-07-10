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
    """C - P = S·e^(-qT - K·e^(-rT). Binomial satisfies this at N=500."""
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
    """Antithetic MC SE on ATM call at 10k sims should be < $0.08.
    ATM payoff has a kink at K, so variance reduction is ~18% (not 50%).
    """
    from src.models import mc_price
    res = mc_price(100, 100, 0.5, 0.05, 0.20, n_sims=10_000, seed=42)
    assert res.meta["std_error"] < 0.08

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