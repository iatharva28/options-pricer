"""
make_all_charts.py
==================

One-shot script that regenerates every chart in the project into ``outputs/``.

Run it from the project root (the folder containing pyproject.toml):

    python examples/make_all_charts.py

What it produces
----------------
Static charts (no internet required):
    outputs/payoff_call.png         – call payoff & P&L at expiry
    outputs/payoff_put.png          – put payoff & P&L at expiry
    outputs/greeks_profile.png      – 6-panel Greeks vs spot
    outputs/mc_paths.png            – 300 Monte Carlo GBM paths
    outputs/convergence.png         – |BTree(N) - BSM| vs N (log-log)

Live-data charts (require internet + yfinance):
    outputs/vol_surface_AAPL.html   – plotly 3D implied-vol surface
    outputs/vol_smile.png           – IV smile curves for several expiries

The live-data section is wrapped in a try/except so the script still
produces the static charts even if yfinance is down or there's no
internet connection.
"""
from __future__ import annotations

# Make the project root importable so `from src...` works no matter which
# directory you run this script from.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ────────────────────────────────────────────────────────────────────────────
# 1. Setup
# ────────────────────────────────────────────────────────────────────────────
# Standard ATM call parameters used throughout the script.
#   S     = spot price
#   K     = strike
#   T     = time to expiry (in years; 0.5 = 6 months)
#   r     = risk-free rate (5%)
#   sigma = volatility (20%)
S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

print("=" * 60)
print("options-pricer — generating all charts")
print("=" * 60)


# ────────────────────────────────────────────────────────────────────────────
# 2. Payoff diagrams — what an option is worth at expiry
# ────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Payoff diagrams (call + put)...")
from src.models import bsm_price
from src.plots import payoff_diagram

call_price = bsm_price(S, K, T, r, sigma, "call").price
put_price  = bsm_price(S, K, T, r, sigma, "put").price

payoff_diagram(S0=S, K=K, premium=call_price, option_type="call")
payoff_diagram(S0=S, K=K, premium=put_price,  option_type="put")
print(f"    call premium = ${call_price:.4f}   put premium = ${put_price:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 3. Greeks profile — 6-panel: Delta, Gamma, Theta, Vega, Vanna, Volga
# ────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Greeks profile (6-panel)...")
from src.plots import greeks_profile
greeks_profile(K=K, T=T, r=r, sigma=sigma)


# ────────────────────────────────────────────────────────────────────────────
# 4. Monte Carlo paths — visualise GBM simulation
# ────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Monte Carlo GBM paths...")
from src.models import mc_simulate_paths
from src.plots import mc_paths_chart

paths = mc_simulate_paths(
    S=S, T=T, r=r, sigma=sigma,
    n_paths=500,      # 500 simulated paths
    n_steps=252,      # daily steps for 1 year
    seed=42,          # reproducible
)
mc_paths_chart(paths, K=K, T=T, option_type="call")


# ────────────────────────────────────────────────────────────────────────────
# 5. Binomial tree convergence — |BTree(N) - BSM| vs N
# ────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Binomial convergence to BSM...")
from src.validation import convergence_report
from src.plots import convergence_chart

report = convergence_report(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call")
convergence_chart(report["binomial"])


# ────────────────────────────────────────────────────────────────────────────
# 6. Live-data charts — implied vol surface + smile (need internet)
# ────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Implied vol surface + smile (live AAPL data)...")
print("      This needs an internet connection. Will skip if it fails.")
try:
    from src.data import get_market_data
    from src.volatility import build_surface_df, VolSurface
    from src.plots import vol_surface_3d, vol_smile_chart

    md = get_market_data("AAPL")
    print(f"      AAPL spot = ${md.spot:.2f}    RFR = {md.rfr:.2%}")

    surface_df = build_surface_df(
        md.option_chain, md.spot, md.rfr, min_oi=10, moneyness_range=0.30
    )
    print(f"      Fetched {len(surface_df)} option quotes for the surface")

    # 3D plotly surface (interactive HTML)
    vol_surface_3d(surface_df, ticker="AAPL")

    # Smile curves for 3 expiries: ~30d, ~90d, ~180d
    surface = VolSurface().fit(surface_df)
    vol_smile_chart(
        surface=surface,
        spot=md.spot,
        T_values=[30 / 365, 90 / 365, 180 / 365],
    )
    print("      ✓ vol_surface_AAPL.html + vol_smile.png written")

except Exception as exc:
    print(f"      ✗ Skipped live-data section: {exc}")


# ────────────────────────────────────────────────────────────────────────────
# 7. Done
# ────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Done. Charts are in the outputs/ folder.")
print("=" * 60)
