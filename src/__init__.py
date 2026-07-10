"""
options-pricer — European option pricing, Greeks, and implied-vol surface.

Public API:
    Pricing engines : bsm_price, binomial_price, mc_price, bsm_price_vec
    Value types     : PricingResult, Greeks, MarketData
    Greeks          : all_greeks, delta, gamma, theta, vega, rho, vanna, volga, charm
    Volatility      : implied_vol, VolSurface, build_surface_df
    Validation      : run_parity_sweep, run_boundary_sweep, convergence_report
    Data            : get_market_data, get_rfr
    Plots           : payoff_diagram, greeks_profile, mc_paths_chart,
                      vol_surface_3d, vol_smile_chart, convergence_chart

Usage:
    from src import bsm_price, all_greeks, get_market_data
"""
from src.models import (
    PricingResult, OptionType,
    bsm_d1, bsm_d2,
    bsm_price, bsm_price_vec,
    binomial_price, binomial_price_vs_steps,
    mc_price, mc_simulate_paths,
)
from src.greeks import (
    Greeks, all_greeks,
    delta, gamma, theta, vega, rho,
    vanna, volga, charm, vega_raw,
    numerical_delta, numerical_gamma, numerical_vega,
    numerical_theta, numerical_rho,
)
from src.volatility import (
    IV_TOL, IV_MAX_ITER, IV_LO, IV_HI,
    implied_vol, build_surface_df, VolSurface,
)
from src.validation import (
    PARITY_TOL_BSM, PARITY_TOL_BINOMIAL, PARITY_TOL_MC, PARITY_TOL_MARKET,
    check_parity, run_parity_sweep, run_boundary_sweep, convergence_report,
)
from src.data import MarketData, get_market_data, get_rfr, FALLBACK_RFR
from src.plots import (
    payoff_diagram, greeks_profile, mc_paths_chart,
    vol_surface_3d, vol_smile_chart, convergence_chart,
)

__version__ = "1.0.0"

__all__ = [
    # version
    "__version__",
    # models
    "PricingResult", "OptionType",
    "bsm_d1", "bsm_d2", "bsm_price", "bsm_price_vec",
    "binomial_price", "binomial_price_vs_steps",
    "mc_price", "mc_simulate_paths",
    # greeks
    "Greeks", "all_greeks",
    "delta", "gamma", "theta", "vega", "rho",
    "vanna", "volga", "charm", "vega_raw",
    "numerical_delta", "numerical_gamma", "numerical_vega",
    "numerical_theta", "numerical_rho",
    # volatility
    "IV_TOL", "IV_MAX_ITER", "IV_LO", "IV_HI",
    "implied_vol", "build_surface_df", "VolSurface",
    # validation
    "PARITY_TOL_BSM", "PARITY_TOL_BINOMIAL", "PARITY_TOL_MC", "PARITY_TOL_MARKET",
    "check_parity", "run_parity_sweep", "run_boundary_sweep", "convergence_report",
    # data
    "MarketData", "get_market_data", "get_rfr", "FALLBACK_RFR",
    # plots
    "payoff_diagram", "greeks_profile", "mc_paths_chart",
    "vol_surface_3d", "vol_smile_chart", "convergence_chart",
]
