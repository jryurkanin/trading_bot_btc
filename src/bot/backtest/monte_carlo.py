"""Monte Carlo simulation on equity curves via stationary block bootstrap."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._bootstrap import stationary_block_bootstrap
from .metrics import (
    _to_returns,
    compute_cagr,
    compute_sharpe,
    compute_sortino,
    max_drawdown,
)


@dataclass
class MonteCarloConfig:
    n_simulations: int = 1000
    block_length: Optional[int] = None  # default: int(sqrt(n))
    percentiles: tuple = (5.0, 25.0, 50.0, 75.0, 95.0)
    seed: int = 42
    freq_per_year: int = 8760


@dataclass
class MonteCarloResult:
    percentile_labels: List[float]
    cagr: Dict[str, float] = field(default_factory=dict)
    sharpe: Dict[str, float] = field(default_factory=dict)
    sortino: Dict[str, float] = field(default_factory=dict)
    max_drawdown: Dict[str, float] = field(default_factory=dict)
    terminal_wealth: Dict[str, float] = field(default_factory=dict)
    n_simulations: int = 0
    block_length_used: int = 0


def _key(p: float) -> str:
    """Format percentile as a dictionary key like 'p5', 'p25', etc."""
    if p == int(p):
        return f"p{int(p)}"
    return f"p{p}"


def run_monte_carlo(
    equity_curve: pd.Series,
    config: MonteCarloConfig | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation on an equity curve.

    Resamples returns via stationary block bootstrap, reconstructs equity
    curves, and computes percentile-based metric distributions.
    """
    if config is None:
        config = MonteCarloConfig()

    returns = _to_returns(equity_curve).values.astype(float)
    n = len(returns)

    block_length = config.block_length
    if block_length is None:
        block_length = max(1, int(np.sqrt(n)))

    initial_value = float(equity_curve.iloc[0]) if len(equity_curve) else 1.0

    rng = np.random.default_rng(config.seed)

    sim_cagr = np.empty(config.n_simulations, dtype=float)
    sim_sharpe = np.empty(config.n_simulations, dtype=float)
    sim_sortino = np.empty(config.n_simulations, dtype=float)
    sim_mdd = np.empty(config.n_simulations, dtype=float)
    sim_tw = np.empty(config.n_simulations, dtype=float)

    for i in range(config.n_simulations):
        sim_returns = stationary_block_bootstrap(
            returns, n, block_length, rng,
        )
        # Reconstruct equity from bootstrapped returns
        equity_values = initial_value * np.cumprod(1.0 + sim_returns)
        equity_values = np.concatenate([[initial_value], equity_values])
        eq_series = pd.Series(equity_values)

        ret_series = pd.Series(sim_returns)

        sim_cagr[i] = compute_cagr(eq_series, periods_per_year=config.freq_per_year)
        sim_sharpe[i] = compute_sharpe(ret_series, periods_per_year=config.freq_per_year)
        sim_sortino[i] = compute_sortino(ret_series, periods_per_year=config.freq_per_year)
        sim_mdd[i] = max_drawdown(eq_series)
        sim_tw[i] = float(equity_values[-1])

    # Compute percentiles
    result = MonteCarloResult(
        percentile_labels=list(config.percentiles),
        n_simulations=config.n_simulations,
        block_length_used=block_length,
    )

    for p in config.percentiles:
        k = _key(p)
        result.cagr[k] = float(np.percentile(sim_cagr, p))
        result.sharpe[k] = float(np.percentile(sim_sharpe, p))
        result.sortino[k] = float(np.percentile(sim_sortino, p))
        result.max_drawdown[k] = float(np.percentile(sim_mdd, p))
        result.terminal_wealth[k] = float(np.percentile(sim_tw, p))

    return result
