from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.backtest.monte_carlo import run_monte_carlo, MonteCarloConfig


def _make_equity(n: int = 1000, initial: float = 10_000.0, daily_return: float = 0.0001, seed: int = 0) -> pd.Series:
    """Simple equity curve with drift + noise."""
    rng = np.random.default_rng(seed)
    returns = daily_return + rng.normal(0, 0.005, n)
    prices = initial * np.cumprod(1.0 + returns)
    prices = np.concatenate([[initial], prices])
    return pd.Series(prices)


def _make_constant_equity(n: int = 500, initial: float = 10_000.0, rate: float = 0.0001) -> pd.Series:
    """Constant-return equity curve (no noise)."""
    prices = initial * (1.0 + rate) ** np.arange(n + 1)
    return pd.Series(prices)


class TestMonteCarloReproducibility:
    def test_same_seed_same_result(self):
        eq = _make_equity(n=500, seed=1)
        r1 = run_monte_carlo(eq, MonteCarloConfig(n_simulations=200, seed=42))
        r2 = run_monte_carlo(eq, MonteCarloConfig(n_simulations=200, seed=42))
        assert r1.cagr == r2.cagr
        assert r1.sharpe == r2.sharpe
        assert r1.terminal_wealth == r2.terminal_wealth


class TestMonteCarloConstantReturn:
    """Constant-return curve should produce tightly clustered percentile bands."""

    def test_tight_bands(self):
        eq = _make_constant_equity(n=500, rate=0.0001)
        result = run_monte_carlo(eq, MonteCarloConfig(n_simulations=500, seed=42))
        # For constant returns, all bootstrapped curves are identical
        tw_values = list(result.terminal_wealth.values())
        spread = max(tw_values) - min(tw_values)
        mean_tw = np.mean(tw_values)
        # Relative spread should be very small
        assert spread / mean_tw < 0.01, f"Spread too large: {spread / mean_tw}"


class TestMonteCarloAutoBlockLength:
    def test_auto_block_length(self):
        n = 400
        eq = _make_equity(n=n, seed=5)
        result = run_monte_carlo(eq, MonteCarloConfig(n_simulations=50, seed=1))
        expected_bl = int(np.sqrt(n))
        assert result.block_length_used == expected_bl


class TestMonteCarloPercentileOrdering:
    """p5 <= p25 <= p50 <= p75 <= p95 for all metrics."""

    def test_ordering(self):
        eq = _make_equity(n=800, seed=3)
        result = run_monte_carlo(eq, MonteCarloConfig(n_simulations=500, seed=42))

        for metric_name in ["cagr", "sharpe", "sortino", "terminal_wealth"]:
            metric_dict = getattr(result, metric_name)
            values = [metric_dict[f"p{int(p)}"] for p in [5, 25, 50, 75, 95]]
            for i in range(len(values) - 1):
                assert values[i] <= values[i + 1] + 1e-10, (
                    f"{metric_name}: p{[5,25,50,75,95][i]}={values[i]} > "
                    f"p{[5,25,50,75,95][i+1]}={values[i+1]}"
                )

    def test_max_drawdown_ordering(self):
        """Max drawdown is negative, so p5 is most negative (worst)."""
        eq = _make_equity(n=800, seed=3)
        result = run_monte_carlo(eq, MonteCarloConfig(n_simulations=500, seed=42))
        values = [result.max_drawdown[f"p{int(p)}"] for p in [5, 25, 50, 75, 95]]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10


class TestMonteCarloEdgeCases:
    def test_short_equity(self):
        eq = pd.Series([10000.0, 10010.0, 10020.0])
        result = run_monte_carlo(eq, MonteCarloConfig(n_simulations=50, seed=1))
        assert result.n_simulations == 50
        assert len(result.cagr) == 5  # 5 default percentiles

    def test_custom_percentiles(self):
        eq = _make_equity(n=200, seed=0)
        result = run_monte_carlo(eq, MonteCarloConfig(
            n_simulations=50, seed=1, percentiles=(10.0, 50.0, 90.0),
        ))
        assert result.percentile_labels == [10.0, 50.0, 90.0]
        assert "p10" in result.cagr
        assert "p50" in result.cagr
        assert "p90" in result.cagr


def test_n_simulations_stored():
    eq = _make_equity(n=200, seed=0)
    result = run_monte_carlo(eq, MonteCarloConfig(n_simulations=123, seed=1))
    assert result.n_simulations == 123
