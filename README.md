# options-pricer

Python library for pricing European options. I built this mostly to teach myself the math properly — writing out BSM, CRR binomial, and a Monte Carlo with antithetic variates from scratch, then sanity-checking each one against put-call parity and the standard boundary conditions.

Formulas are straight out of Hull (2018), *Options, Futures, and Other Derivatives*, 10th Ed. Nothing exotic — but everything actually works and the test suite (47 tests) proves it.

## What's in here

- **Three pricing models** behind a single `PricingResult` dataclass:
  - Black-Scholes-Merton (analytical, vectorised)
  - CRR binomial tree (recombining, vectorised backward induction)
  - Monte Carlo with antithetic variates — reports a 95% CI on the price
- **Greeks**: delta, gamma, theta, vega, rho, plus vanna / volga / charm. Each analytical formula has a central-difference counterpart used by the tests, so the analyticals are actually verified, not just asserted.
- **Implied vol solver** — Newton-Raphson with a Brent fallback. σ gets clamped to `[1e-6, 10.0]` every step so it can't explode on weird inputs.
- **Vol surface** — `SmoothBivariateSpline` over `(strike, T) → IV`, fitted from yfinance option chains.
- **Market data** — yfinance for spot / OHLCV / option chains, FRED for the 3M T-Bill rate. If no FRED key is set, it silently falls back to `r = 0.05` rather than crashing.
- **Charts** — matplotlib for payoff / Greeks / MC paths / vol smile / convergence, plotly for an interactive 3D IV surface.

A few performance choices worth flagging: I use `scipy.special.ndtr` instead of `scipy.stats.norm.cdf` (about 3–5× faster), `dataclass(slots=True)` for the value types, and the MC paths are simulated with `np.cumsum` of log-increments rather than a Python loop (about 10× faster). The risk-free-rate fetcher is memoised with `lru_cache(maxsize=1)` because TB3MS only updates monthly — refetching per call would just be waste.

## Install

```bash
git clone https://github.com/<your-username>/options-pricer.git
cd options-pricer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e ".[dev]"
```

Needs Python 3.10+ (uses `dataclass(slots=True)`).

### FRED API key (optional)

```bash
cp .env.example .env
# put your key in .env — free at https://fred.stlouisfed.org/docs/api/api_key.html
```

Without a key the library just uses `r = 0.05` and prints a one-line info log.

## Quick start

```python
from src.models import bsm_price, binomial_price, mc_price
from src.greeks import all_greeks

S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20

bsm  = bsm_price(S, K, T, r, sigma, "call")
tree = binomial_price(S, K, T, r, sigma, "call", steps=500)
mc   = mc_price(S, K, T, r, sigma, "call", n_sims=100_000, seed=42)

print(f"BSM       : ${bsm.price:.4f}")
print(f"Binomial  : ${tree.price:.4f}")
print(f"MC 95% CI : ${mc.price:.4f}  [{mc.meta['ci_lower']:.4f}, {mc.meta['ci_upper']:.4f}]")

g = all_greeks(S, K, T, r, sigma, "call")
print(f"Δ={g.delta:+.4f}  Γ={g.gamma:.4f}  Θ={g.theta:+.4f}/day  ν={g.vega:.4f}/1%σ")
```

Real market data + IV surface:

```python
from src.data import get_market_data
from src.volatility import build_surface_df, VolSurface, implied_vol

md = get_market_data("AAPL")
df = build_surface_df(md.option_chain, md.spot, md.rfr)

surface = VolSurface().fit(df)
print(f"AAPL ATM 30d IV: {surface.query(strike=md.spot, T=30/365):.2%}")

# or back out IV from a quoted price
iv = implied_vol(market_price=4.76, S=42, K=40, T=0.5, r=0.10, option_type="call")
```

## Tests

```bash
pytest                                       # full suite, ~2s
pytest -m "not slow and not network"         # skip slow / network tests
pytest --cov=src --cov-report=term-missing   # with coverage
```

What each file covers:

- `test_models.py` — pricing vs Hull Table 15.1, put-call parity, boundary conditions, near-expiry stability, broadcasting
- `test_greeks.py` — analytical vs numerical (central differences) for delta / gamma / vega / theta / rho
- `test_validation.py` — parity sweeps across strikes for all three models, boundary sweeps on a `(K, σ)` grid, binomial convergence, MC 95%-CI coverage
- `test_volatility.py` — IV round-trips, deep-OTM, arbitrage rejection, surface fit / query / smile
- `test_perf.py` — smoke tests for performance regressions (BSM < 1ms, MC 100k < 500ms, IV solver < 5ms, etc.)

## Project layout

```
options-pricer/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── SPEC.md                    # the original design doc — kept for reference
├── src/
│   ├── models.py              # BSM, CRR binomial, Monte Carlo
│   ├── greeks.py              # 8 Greeks + numerical cross-checks
│   ├── volatility.py          # IV solver + VolSurface
│   ├── validation.py          # parity, boundaries, convergence
│   ├── data.py                # yfinance + FRED fetcher
│   └── plots.py               # matplotlib + plotly charts → outputs/
├── tests/
├── notebooks/
│   ├── 01_full_demo.ipynb
│   └── 02_validation_report.ipynb
└── outputs/                   # populated by plots.py at runtime (gitignored)
```

## Scope — what this is *not*

I deliberately kept this scoped to a library. There is no CLI, no web server, no REST API, no persistent cache. American options aren't supported (no early-exercise), historical vol is close-to-close only (no GARCH / Parkinson), and the vol surface is implied vol only — no local vol / Dupire. All stochastic functions take a `seed` parameter (default `42`) so tests are reproducible.

## TODO

- [ ] American options via Black's approximation (or a proper tree with early-exercise)
- [ ] Dividend yield auto-estimation from yfinance (right now `q` defaults to 0)
- [ ] GARCH(1,1) historical vol as an alternative estimator
- [ ] Add a couple of test cases that hit the live yfinance + FRED path (currently marked `network`, not run in CI)
- [ ] Type-check the public surface with mypy / pyright

## License

MIT — see [`LICENSE`](LICENSE).
