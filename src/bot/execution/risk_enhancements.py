"""Risk management enhancements (8.10).

Planned enhancements:
- Conditional VaR (CVaR/Expected Shortfall) for tail risk monitoring
- Regime-conditional risk limits: tighter limits in HIGH_VOL regime
- Correlation-based position limits: reduce size when BTC correlates with risk assets
- Intraday VaR tracking with real-time P&L
- Stress testing framework: apply historical drawdown scenarios to current portfolio
- Dynamic stop-loss: adjust stop levels based on current regime and vol
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CVaRConfig:
    enabled: bool = False
    confidence_level: float = 0.95  # 95% CVaR
    lookback_days: int = 252
    max_cvar_pct: float = 0.05  # max 5% expected shortfall


def compute_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    Returns the average loss in the worst (1 - confidence) fraction of scenarios.
    """
    returns = returns.dropna()
    if len(returns) < 10:
        return 0.0
    cutoff = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= cutoff]
    if len(tail) == 0:
        return float(cutoff)
    return float(np.mean(tail))


def compute_var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Compute Value at Risk at given confidence level."""
    returns = returns.dropna()
    if len(returns) < 10:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


@dataclass
class StressScenario:
    name: str
    drawdown_pct: float  # e.g. 0.30 for a 30% drawdown scenario
    duration_days: int
    description: str = ""


# Historical BTC stress scenarios for reference
HISTORICAL_STRESS_SCENARIOS = [
    StressScenario("covid_crash_2020", 0.50, 30, "March 2020 COVID crash"),
    StressScenario("china_ban_2021", 0.55, 90, "May-July 2021 China mining ban"),
    StressScenario("ftx_collapse_2022", 0.25, 14, "Nov 2022 FTX collapse"),
    StressScenario("luna_crash_2022", 0.40, 30, "May 2022 LUNA/UST depegging"),
]


def stress_test_portfolio(
    current_equity: float,
    current_exposure: float,
    scenarios: list[StressScenario] | None = None,
) -> dict[str, float]:
    """Estimate portfolio loss under each stress scenario.

    Returns dict of scenario_name -> estimated_loss_usd.
    """
    if scenarios is None:
        scenarios = HISTORICAL_STRESS_SCENARIOS
    results = {}
    for s in scenarios:
        btc_value = current_equity * current_exposure
        loss = btc_value * s.drawdown_pct
        results[s.name] = loss
    return results
