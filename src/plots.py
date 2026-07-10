"""
Visualisation. All functions save to outputs/ and return the figure.
Call plt.close(fig) after if generating many charts in a loop.

Charts:
  payoff_diagram      — payoff + P&L at expiry
  greeks_profile      — 6-panel first+second order Greeks vs spot
  mc_paths_chart      — simulated GBM paths
  vol_surface_3d      — plotly interactive surface (saves .html)
  vol_smile_chart     — smile curves for multiple expiries
  convergence_chart   — |BTree(N) - BSM| vs N
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

matplotlib.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",   "text.color": "#c9d1d9",
    "axes.labelcolor": "#c9d1d9",  "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",      "grid.color": "#21262d",
    "lines.linewidth": 2.0,
})

OUTPUTS = Path("outputs")
OUTPUTS.mkdir(exist_ok=True)

_BLUE = "#58a6ff"
_RED  = "#f78166"
_GREEN= "#3fb950"
_ORNG = "#ffa657"
_GREY = "#8b949e"

__all__ = [
    "payoff_diagram", "greeks_profile", "mc_paths_chart",
    "vol_surface_3d", "vol_smile_chart", "convergence_chart",
]


def payoff_diagram(S0: float, K: float, premium: float,
                   option_type: str = "call", filename: str = None) -> plt.Figure:
    """Payoff at expiry + net P&L (after premium). Marks strike, spot, breakeven."""
    S_range = np.linspace(S0 * 0.6, S0 * 1.4, 500)
    payoff  = np.maximum(S_range - K, 0) if option_type == "call" else np.maximum(K - S_range, 0)
    pnl     = payoff - premium

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(S_range, payoff, color=_BLUE, label="Payoff at expiry")
    ax.plot(S_range, pnl,    color=_BLUE, linestyle="--", alpha=0.8, label="P&L (net of premium)")
    ax.axhline(0, color=_GREY, lw=0.8)
    ax.axvline(K,  color=_ORNG, lw=1.5, linestyle=":", label=f"Strike K={K:.2f}")
    ax.axvline(S0, color=_GREEN,lw=1.5, linestyle=":", label=f"Spot S₀={S0:.2f}")
    be = (K + premium) if option_type == "call" else (K - premium)
    if S_range[0] < be < S_range[-1]:
        ax.axvline(be, color="#d2a8ff", lw=1, linestyle="--", label=f"Breakeven={be:.2f}")
    ax.fill_between(S_range, pnl, 0, where=(pnl > 0), alpha=0.18, color=_BLUE)
    ax.fill_between(S_range, pnl, 0, where=(pnl < 0), alpha=0.12, color=_RED)
    ax.set_xlabel("Spot at Expiry"); ax.set_ylabel("Value / P&L ($)")
    ax.set_title(f"{option_type.title()} Payoff  |  K={K}  Premium={premium:.4f}")
    ax.legend(); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    path = OUTPUTS / (filename or f"payoff_{option_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def greeks_profile(K: float, T: float, r: float, sigma: float,
                   q: float = 0.0, filename: str = None) -> plt.Figure:
    """6-panel: Delta, Gamma, Theta, Vega, Vanna, Volga vs spot. Call+Put on relevant panels."""
    from src.greeks import delta, gamma, theta, vega, vanna, volga

    S_range = np.linspace(K * 0.5, K * 1.5, 200)
    panels = {
        "Delta":  (lambda S, ot: delta(S, K, T, r, sigma, ot, q), True),
        "Gamma":  (lambda S, ot: gamma(S, K, T, r, sigma, q),     False),
        "Theta":  (lambda S, ot: theta(S, K, T, r, sigma, ot, q), True),
        "Vega":   (lambda S, ot: vega(S,  K, T, r, sigma, q),     False),
        "Vanna":  (lambda S, ot: vanna(S, K, T, r, sigma, q),     False),
        "Volga":  (lambda S, ot: volga(S, K, T, r, sigma, q),     False),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (name, (fn, call_put_differ)) in zip(axes.flat, panels.items()):
        call_vals = [fn(S, "call") for S in S_range]
        ax.plot(S_range, call_vals, color=_BLUE, label="Call")
        if call_put_differ:
            put_vals = [fn(S, "put") for S in S_range]
            ax.plot(S_range, put_vals, color=_RED, linestyle="--", label="Put")
        ax.axvline(K, color=_ORNG, lw=1, linestyle=":", alpha=0.7)
        ax.axhline(0, color=_GREY, lw=0.6)
        ax.set_title(name); ax.set_xlabel("Spot")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Greeks Profile  |  K={K}  T={T:.2f}yr  σ={sigma:.0%}  r={r:.1%}", fontsize=13)
    plt.tight_layout()
    path = OUTPUTS / (filename or "greeks_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); return fig


def mc_paths_chart(paths: np.ndarray, K: float, T: float,
                   option_type: str = "call", filename: str = None) -> plt.Figure:
    """Plot up to 300 MC paths. ITM blue, OTM red. Mean path in green."""
    n_show = min(300, paths.shape[0])
    t_axis = np.linspace(0, T, paths.shape[1])
    fig, ax = plt.subplots(figsize=(12, 5))
    for path in paths[:n_show]:
        itm   = path[-1] > K if option_type == "call" else path[-1] < K
        ax.plot(t_axis, path, color=_BLUE if itm else _RED, alpha=0.10, lw=0.5)
    ax.plot(t_axis, paths.mean(axis=0), color=_GREEN, lw=2, label="Mean path", zorder=5)
    ax.axhline(K, color=_ORNG, lw=1.5, linestyle="--", label=f"Strike K={K:.2f}")
    ax.set_xlabel("Time (years)"); ax.set_ylabel("Stock Price")
    ax.set_title(f"Monte Carlo Paths ({paths.shape[0]:,})")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    path_f = OUTPUTS / (filename or "mc_paths.png")
    fig.savefig(path_f, dpi=150, bbox_inches="tight"); return fig


def vol_surface_3d(surface_df: pd.DataFrame, ticker: str = "",
                   filename: str = None) -> go.Figure:
    """Interactive plotly 3D vol surface. Saves .html to outputs/."""
    df = surface_df.dropna(subset=["iv"])
    fig = go.Figure(go.Scatter3d(
        x=df["strike"], y=df["dte"], z=df["iv"] * 100,
        mode="markers",
        marker=dict(size=4, color=df["iv"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="IV (%)")),
        text=[f"K={k:.1f}<br>{d}d<br>{iv:.1%}"
              for k, d, iv in zip(df["strike"], df["dte"], df["iv"])],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Implied Volatility Surface{' — ' + ticker if ticker else ''}",
        scene=dict(xaxis_title="Strike", yaxis_title="Days to Expiry",
                   zaxis_title="IV (%)", bgcolor="#0d1117"),
        paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"),
        width=900, height=650,
    )
    path = OUTPUTS / (filename or f"vol_surface{'_' + ticker if ticker else ''}.html")
    fig.write_html(str(path)); return fig


def vol_smile_chart(surface, spot: float, T_values: list,
                    filename: str = None) -> plt.Figure:
    """Smile curves for multiple expiries. surface: VolSurface instance."""
    colors = plt.cm.cool(np.linspace(0, 1, len(T_values)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for T, color in zip(T_values, colors):
        df = surface.smile(T, spot)
        ax.plot(df["moneyness"], df["iv"] * 100, color=color,
                label=f"T={T:.2f}yr ({int(T*365)}d)")
    ax.axvline(1.0, color=_GREY, lw=1, linestyle=":", label="ATM")
    ax.set_xlabel("Moneyness (K/S)"); ax.set_ylabel("IV (%)")
    ax.set_title("Vol Smile by Expiry"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUTPUTS / (filename or "vol_smile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); return fig


def convergence_chart(binomial_df: pd.DataFrame, filename: str = None) -> plt.Figure:
    """Plot |BTree(N) - BSM| vs N on log scale."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(binomial_df["N"], binomial_df["abs_error"], color=_BLUE, marker="o", ms=5)
    ax.axhline(0.01, color=_RED, lw=1, linestyle="--", label="$0.01 threshold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of Steps (N)"); ax.set_ylabel("|BTree − BSM| ($)")
    ax.set_title("Binomial Tree Convergence to BSM")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    path = OUTPUTS / (filename or "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); return fig