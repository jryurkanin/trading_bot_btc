from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

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
