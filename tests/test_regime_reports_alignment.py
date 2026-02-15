from __future__ import annotations

import pandas as pd

from bot.backtest.regime_reports import performance_by_regime


def test_performance_by_regime_handles_datetime_precision_mismatch():
    ts = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")

    ts_us = pd.to_datetime((ts.view("int64") // 1_000).astype("int64"), unit="us", utc=True)
    ts_s = pd.to_datetime((ts.view("int64") // 1_000_000_000).astype("int64"), unit="s", utc=True)

    equity_curve = pd.DataFrame(
        {
            "timestamp": ts_us,
            "equity": [10000, 10010, 10030, 10020, 10040, 10035, 10060, 10080],
            "micro_regime": ["TREND", "TREND", "RANGE", "RANGE", "TREND", "TREND", "RANGE", "RANGE"],
        }
    ).set_index("timestamp")

    decisions = pd.DataFrame(
        {
            "timestamp": ts_s,
            "decision_applies_at": ts_s,
            "micro_regime": ["TREND", "TREND", "RANGE", "RANGE", "TREND", "TREND", "RANGE", "RANGE"],
        }
    ).set_index("timestamp")

    trades = pd.DataFrame(
        {
            "ts": ts_s,
            "notional": [100, 50, 75, 60, 110, 40, 80, 30],
            "fee": [0.1] * 8,
            "side": ["BUY", "SELL", "BUY", "SELL", "BUY", "SELL", "BUY", "SELL"],
        }
    )

    by_regime = performance_by_regime(equity_curve, trades_df=trades, decisions_df=decisions)
    assert isinstance(by_regime, dict)
    assert "TREND" in by_regime or "RANGE" in by_regime
