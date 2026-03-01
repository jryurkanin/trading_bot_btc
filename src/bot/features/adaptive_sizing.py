"""Adaptive position sizing with Kelly criterion (8.6).

Planned approaches:
- Full Kelly fraction: f* = (p * b - q) / b where p=win rate, b=avg win/avg loss
- Half-Kelly for conservative sizing (standard in practice)
- Regime-conditional Kelly: separate estimates per detected regime
- Bayesian Kelly with confidence intervals on win rate / payoff ratio
- Integration with existing vol_sizing.py for combined signal

The output is a fractional multiplier [0, 1] applied to the base position size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KellyConfig:
    enabled: bool = False
    fraction: float = 0.5  # Half-Kelly by default
    min_trades: int = 30  # minimum trades before Kelly estimate is trusted
    lookback_trades: int = 200  # rolling window of trades for estimation
    max_kelly_fraction: float = 0.5  # hard cap regardless of Kelly output


def compute_kelly_fraction(
    trade_returns: pd.Series,
    config: KellyConfig,
) -> float:
    """Compute Kelly-optimal position fraction from recent trade returns.

    Returns a fraction in [0, max_kelly_fraction].
    Currently a no-op placeholder returning config.fraction.
    """
    if not config.enabled:
        return 1.0

    returns = trade_returns.dropna().tail(config.lookback_trades)
    if len(returns) < config.min_trades:
        return config.fraction  # not enough data, use default

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    if len(wins) == 0 or len(losses) == 0:
        return config.fraction

    p = len(wins) / len(returns)  # win rate
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(np.abs(losses)))

    if avg_loss <= 0:
        return config.fraction

    b = avg_win / avg_loss  # payoff ratio
    kelly = (p * b - (1 - p)) / b

    # Apply fractional Kelly and cap
    scaled = max(0.0, kelly * config.fraction)
    return min(scaled, config.max_kelly_fraction)
