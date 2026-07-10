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