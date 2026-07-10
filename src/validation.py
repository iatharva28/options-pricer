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