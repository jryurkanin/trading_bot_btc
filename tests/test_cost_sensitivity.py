from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.config import BotConfig
from bot.backtest.engine import BacktestEngine
from bot.backtest.cost_sensitivity import (
    run_cost_sensitivity,
    CostSensitivityConfig,
    _interpolate_breakeven,
)


def make_oscillating_candles(n=400, product="BTC-USD"):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    ts = pd.date_range(start, periods=n, freq="h")
    close = 30000 + (pd.Series(range(n)) % 20 - 10).astype(float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 5
    low = close - 5
    vol = 100.0
    df = pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )
    dstart = pd.Timestamp("2025-01-01", tz="UTC")
    dts = pd.date_range(dstart, periods=max(2, n // 24), freq="D")
    daily_close = pd.Series(30000 + (pd.Series(range(len(dts))) % 20 - 10).astype(float).values, index=dts)
    daily = pd.DataFrame(
        {"timestamp": dts, "open": daily_close, "high": daily_close + 10, "low": daily_close - 10, "close": daily_close, "volume": 1000.0}
    )
    return df, daily


class TestCostSensitivityOutput:
    def setup_method(self):
        self.hourly, self.daily = make_oscillating_candles(n=400)
        self.cfg = BotConfig()
        self.cfg.data.product = "BTC-USD"
        self.cfg.backtest.initial_equity = 10000
        self.cfg.backtest.start = "2025-01-01"
        self.cfg.backtest.end = str(self.hourly["timestamp"].iloc[-1].date())

    def test_output_has_correct_rows(self):
        n_steps = 3
        result = run_cost_sensitivity(
            self.hourly, self.daily, self.cfg,
            CostSensitivityConfig(n_steps=n_steps),
        )
        assert len(result.combined_sweep) == n_steps + 1

    def test_required_columns(self):
        result = run_cost_sensitivity(
            self.hourly, self.daily, self.cfg,
            CostSensitivityConfig(n_steps=3),
        )
        df = result.combined_sweep
        for col in ["cost_multiplier", "cagr", "sharpe", "sortino", "max_drawdown", "turnover", "trade_count"]:
            assert col in df.columns, f"Missing column: {col}"


class TestCostSensitivityMonotonicity:
    """Sharpe and CAGR at multiplier=0 should be >= at multiplier=3."""

    def test_degradation(self):
        hourly, daily = make_oscillating_candles(n=400)
        cfg = BotConfig()
        cfg.data.product = "BTC-USD"
        cfg.backtest.initial_equity = 10000
        cfg.backtest.start = "2025-01-01"
        cfg.backtest.end = str(hourly["timestamp"].iloc[-1].date())

        result = run_cost_sensitivity(
            hourly, daily, cfg,
            CostSensitivityConfig(multiplier_min=0.0, multiplier_max=3.0, n_steps=3),
        )
        df = result.combined_sweep
        sharpe_0 = df.loc[df["cost_multiplier"] == 0.0, "sharpe"].iloc[0]
        sharpe_max = df.loc[df["cost_multiplier"] == df["cost_multiplier"].max(), "sharpe"].iloc[0]
        assert sharpe_0 >= sharpe_max


class TestBreakevenInterpolation:
    def test_known_crossing(self):
        mults = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([2.0, 1.0, -0.5, -1.5])
        be = _interpolate_breakeven(mults, values, 0.0)
        assert be is not None
        # Crossing between mult=1 (val=1.0) and mult=2 (val=-0.5)
        # Expected: 1 + 1.0 / 1.5 = 1.6667
        assert be == pytest.approx(1.6667, abs=0.01)

    def test_no_crossing(self):
        mults = np.array([0.0, 1.0, 2.0])
        values = np.array([3.0, 2.0, 1.0])
        be = _interpolate_breakeven(mults, values, 0.0)
        assert be is None

    def test_exact_boundary(self):
        mults = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 0.0, -1.0])
        be = _interpolate_breakeven(mults, values, 0.0)
        assert be is not None
        assert be == pytest.approx(1.0, abs=0.01)
