# Options Pricing & Greeks Calculator

A Python library for pricing European options using three models (Black-Scholes-Merton, CRR Binomial Tree, Monte Carlo with antithetic variates), computing first- and second-order Greeks, validating results against analytical boundaries, and building an implied-volatility surface from real market data.

> This is a quantitative-finance reference implementation, **not** a trading system, web app, or production service. All formulas follow Hull (2018), *Options, Futures, and Other Derivatives*, 10th Ed.

---

## Features

- **Three pricing engines** with a unified `PricingResult` dataclass
  - Black-Scholes-Merton (analytical, vectorised)
  - CRR Binomial Tree (recombining, vectorised backward induction)
  - Monte Carlo (GBM with antithetic variates, 95% CI reported)
- **Eight Greeks**: delta, gamma, theta, vega, rho, vanna, volga, charm — analytical + numerical cross-checks
- **Implied volatility solver**: Newton-Raphson with Brent fallback, clamped to `[1e-6, 10.0]`
- **Volatility surface**: `SmoothBivariateSpline` over `(strike, T) → IV`, with smile and term-structure queries
- **Market data**: yfinance (spot, OHLCV, option chains) + FRED (3M T-Bill risk-free rate, falls back to 5% if no API key)
- **Validation suite**: put-call parity, boundary conditions, model convergence
- **Visualisation**: matplotlib (payoff, Greeks profile, MC paths, vol smile, convergence) + plotly (interactive 3D surface)
- **Performance optimisations**: `scipy.special.ndtr` instead of `scipy.stats.norm`, `dataclass(slots=True)`, vectorised MC path simulation via `np.cumsum`, `lru_cache` on RFR fetcher

---

## Installation

```bash
git clone https://github.com/<your-username>/options-pricer.git
cd options-pricer

# Create a virtual environment (Python 3.10+ required)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install runtime + dev dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Environment variables (optional)

Copy `.env.example` to `.env` and add a FRED API key for real risk-free-rate fetching. If the key is missing, the library silently falls back to `r = 0.05`.

```bash
cp .env.example .env
# Edit .env and replace `your_key_here` with your FRED key
# Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
```

---

## Quick start

```python
from src.models import bsm_price, binomial_price, mc_price
from src.greeks import all_greeks

# Price an ATM European call: S=100, K=100, T=6m, r=5%, σ=20%
S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

bsm  = bsm_price(S, K, T, r, sigma, "call")
tree = binomial_price(S, K, T, r, sigma, "call", steps=500)
mc   = mc_price(S, K, T, r, sigma, "call", n_sims=100_000, seed=42)

print(f"BSM       : ${bsm.price:.4f}")
print(f"Binomial  : ${tree.price:.4f}")
print(f"MC (95%CI): ${mc.price:.4f}  [{mc.meta['ci_lower']:.4f}, {mc.meta['ci_upper']:.4f}]")

# Greeks — all 8 in one call
g = all_greeks(S, K, T, r, sigma, "call")
print(f"Δ={g.delta:+.4f}  Γ={g.gamma:.4f}  Θ={g.theta:+.4f}/day  ν={g.vega:.4f}/1%σ")
```

### Real market data + IV surface

```python
from src.data import get_market_data
from src.volatility import build_surface_df, VolSurface, implied_vol

md = get_market_data("AAPL")                # fetches spot, RFR, OHLCV, option chains
df = build_surface_df(md.option_chain, md.spot, md.rfr)   # clean (K, T, IV) grid

surface = VolSurface().fit(df)
iv_atm_30d = surface.query(strike=md.spot, T=30/365)
print(f"AAPL ATM 30-day IV: {iv_atm_30d:.2%}")

# Solve IV from a market price
iv = implied_vol(market_price=4.76, S=42, K=40, T=0.5, r=0.10, option_type="call")
```

---

## Testing

```bash
# Full suite (47 tests, ~2s)
pytest

# Exclude slow / network tests
pytest -m "not slow and not network"

# With coverage report
pytest --cov=src --cov-report=term-missing
```

The test suite covers:
- **`test_models.py`** — BSM/Binomial/MC pricing vs Hull (2018) Table 15.1, put-call parity, boundary conditions, near-expiry stability, broadcasting
- **`test_greeks.py`** — analytical vs numerical (central differences) for delta/gamma/vega/theta/rho
- **`test_validation.py`** — put-call parity sweeps across strikes for all 3 models, boundary sweeps across `(K, σ)` grid, binomial convergence, MC 95%-CI coverage
- **`test_volatility.py`** — IV round-trips, deep-OTM, arbitrage rejection, surface fit/query/smile
- **`test_perf.py`** — performance smoke tests (BSM < 1ms, BSM-vec 10k < 50ms, Binomial-500 < 50ms, MC 100k < 500ms, all Greeks < 2ms, IV solver < 5ms)

---

## Project structure

```
options-pricer/
├── pyproject.toml              # PEP 621 metadata + pytest config
├── requirements.txt            # pinned dependencies
├── .env.example                # FRED_API_KEY template
├── SPEC.md                     # full design specification
├── README.md
├── LICENSE
├── .gitignore
├── .github/workflows/ci.yml    # GitHub Actions: pytest on push
├── src/
│   ├── __init__.py             # public API re-exports
│   ├── models.py               # BSM, CRR Binomial, Monte Carlo
│   ├── greeks.py               # 8 Greeks + numerical cross-checks
│   ├── volatility.py           # IV solver + VolSurface
│   ├── validation.py           # parity, boundaries, convergence
│   ├── data.py                 # yfinance + FRED fetcher
│   └── plots.py                # matplotlib + plotly charts → outputs/
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_greeks.py
│   ├── test_validation.py
│   ├── test_volatility.py
│   └── test_perf.py
├── notebooks/
│   ├── 01_full_demo.ipynb      # end-to-end demo on AAPL
│   └── 02_validation_report.ipynb
└── outputs/                    # auto-populated by plots.py (gitignored except .gitkeep)
```

---

## Performance notes

- `scipy.special.ndtr` is used for the standard-normal CDF — **3–5× faster** than `scipy.stats.norm.cdf`.
- `dataclass(slots=True)` for all value types (`PricingResult`, `Greeks`, `MarketData`) — cache-friendly memory layout, ~20% faster attribute access.
- MC path simulation vectorised with `np.cumsum` of log-increments — **~10× faster** than per-step Python loop.
- `lru_cache(maxsize=1)` on `_get_rfr()` — TB3MS updates monthly, no need to refetch per call within a session.
- `np.asarray(..., dtype=float)` at vectorised entry points — avoids silent type promotion.

---

## Design choices

- **European options only** — no American early-exercise.
- **Close-to-close historical vol** — no GARCH, no Parkinson.
- **Implied vol surface only** — no local vol / Dupire.
- **No CLI, no web server, no REST API, no persistent cache** — this is a library, not a service.
- All stochastic functions take a `seed` parameter (default `42`) for reproducibility.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.

---

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*, 10th Edition. Pearson.
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229–263.
