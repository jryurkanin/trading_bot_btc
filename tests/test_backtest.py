from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bot.config import BotConfig
from bot.backtest.engine import BacktestEngine


def make_oscillating_candles(n=400, product="BTC-USD"):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    ts = pd.date_range(start, periods=n, freq="h")
    # simple oscillation to give trend/range signals
    close = 30000 + (pd.Series(range(n)) % 20 - 10).astype(float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 5
    low = close - 5
    vol = 100.0
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )
    dstart = pd.Timestamp("2025-01-01", tz="UTC")
    dts = pd.date_range(dstart, periods=max(2, n // 24), freq="D")
    daily_close = pd.Series(30000 + (pd.Series(range(len(dts))) % 20 - 10).astype(float).values, index=dts)
    daily = pd.DataFrame(
        {
            "timestamp": dts,
            "open": daily_close,
            "high": daily_close + 10,
            "low": daily_close - 10,
            "close": daily_close,
            "volume": 1000.0,
        }
    )
    return df, daily


def test_backtest_runs_and_outputs_metrics():
    hourly, daily = make_oscillating_candles()
    cfg = BotConfig()
    cfg.backtest.initial_equity = 10000

    eng = BacktestEngine(
        product="BTC-USD",
        hourly_candles=hourly,
        daily_candles=daily,
        start=hourly["timestamp"].iloc[0].to_pydatetime(),
        end=hourly["timestamp"].iloc[-1].to_pydatetime(),
        config=cfg.backtest,
        fees=(0.0001, 0.00025),
        slippage_bps=1.0,
    )

    res = eng.run()
    assert not res.equity_curve.empty
    assert "equity" in res.equity_curve.columns
    assert "cagr" in res.metrics
    assert "sharpe" in res.metrics
    assert res.metrics["max_drawdown"] <= 0.0


def test_unsupported_strategy_rejected_by_engine():
    hourly, daily = make_oscillating_candles()
    cfg = BotConfig()
    cfg.backtest.initial_equity = 10000
    cfg.backtest.strategy = "macro_gate_benchmark"

    eng = BacktestEngine(
        product="BTC-USD",
        hourly_candles=hourly,
        daily_candles=daily,
        start=hourly["timestamp"].iloc[0].to_pydatetime(),
        end=hourly["timestamp"].iloc[-1].to_pydatetime(),
        config=cfg.backtest,
        fees=(0.0001, 0.00025),
        slippage_bps=1.0,
    )
    res = eng.run()
    assert not res.decisions.empty

    cfg = BotConfig()
    cfg.backtest.initial_equity = 10000
    cfg.backtest.strategy = "v5_adaptive"

    eng = BacktestEngine(
        product="BTC-USD",
        hourly_candles=hourly,
        daily_candles=daily,
        start=hourly["timestamp"].iloc[0].to_pydatetime(),
        end=hourly["timestamp"].iloc[-1].to_pydatetime(),
        config=cfg.backtest,
        fees=(0.0001, 0.00025),
        slippage_bps=1.0,
    )

    with pytest.raises(ValueError):
        eng.run()


def test_backtest_uses_prestart_daily_history_for_macro_warmup():
    # Build long, gently trending data so daily macro score should be positive
    # once sufficient history is available.
    n_days = 320
    hourly_ts = pd.date_range("2024-01-01", periods=n_days * 24, freq="h", tz="UTC")
    base = np.arange(len(hourly_ts), dtype=float)
    close = 20_000.0 + base * 1.5
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 5.0
    low = np.minimum(open_, close) - 5.0

    hourly = pd.DataFrame(
        {
            "timestamp": hourly_ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        }
    )

    daily_ts = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    dbase = np.arange(n_days, dtype=float)
    dclose = 20_000.0 + dbase * 20.0
    daily = pd.DataFrame(
        {
            "timestamp": daily_ts,
            "open": dclose,
            "high": dclose + 20.0,
            "low": dclose - 20.0,
            "close": dclose,
            "volume": 1000.0,
        }
    )

    cfg = BotConfig()
    cfg.backtest.strategy = "macro_gate_benchmark"
    cfg.regime.v4_macro_confirm_days = 1
    cfg.regime.v4_macro_min_off_days = 1
    cfg.regime.v4_macro_min_on_days = 1

    eval_start = hourly_ts[24 * 250].to_pydatetime()
    eval_end = hourly_ts[-1].to_pydatetime()

    eng = BacktestEngine(
        product="BTC-USD",
        hourly_candles=hourly,
        daily_candles=daily,
        start=eval_start,
        end=eval_end,
        config=cfg.backtest,
        fees=(0.0001, 0.00025),
        slippage_bps=1.0,
        regime_config=cfg.regime,
    )

    res = eng.run()
    assert not res.decisions.empty
    # With preserved pre-start daily history, macro gate should turn on.
    assert float(res.decisions["macro_multiplier"].max()) > 0.0
