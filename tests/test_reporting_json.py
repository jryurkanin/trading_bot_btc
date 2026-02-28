from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from bot.backtest.engine import BacktestEngine
from bot.backtest.reporting import write_strict_json
from bot.config import BotConfig


def _make_candles(n: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    base = pd.Series(range(n), dtype=float)
    close = 20_000.0 + (base % 96 - 48) * 8.0 + (base / 12.0)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 10.0
    low = pd.concat([open_, close], axis=1).min(axis=1) - 10.0

    hourly = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        }
    )

    dts = pd.date_range("2022-01-01", periods=max(420, n // 24 + 10), freq="D", tz="UTC")
    dbase = pd.Series(range(len(dts)), dtype=float)
    dclose = 20_000.0 + (dbase % 20 - 10) * 50.0 + dbase * 6.0
    daily = pd.DataFrame(
        {
            "timestamp": dts,
            "open": dclose,
            "high": dclose + 30.0,
            "low": dclose - 30.0,
            "close": dclose,
            "volume": 1000.0,
        }
    )
    return hourly, daily


def _assert_finite_or_none(v):
    if v is None:
        return
    assert isinstance(v, (int, float))
    assert math.isfinite(float(v))


def test_report_json_is_strict_and_profit_factor_consistent(tmp_path: Path):
    hourly, daily = _make_candles()
    cfg = BotConfig()
    cfg.backtest.initial_equity = 10_000.0
    cfg.regime.macro_mode = "binary"
    cfg.regime.trend_boost_enabled = False

    engine = BacktestEngine(
        product="BTC-USD",
        hourly_candles=hourly,
        daily_candles=daily,
        start=hourly["timestamp"].iloc[0].to_pydatetime(),
        end=hourly["timestamp"].iloc[-1].to_pydatetime(),
        config=cfg.backtest,
        fees=(0.001, 0.0025),
        slippage_bps=5.0,
        use_spread_slippage=True,
        regime_config=cfg.regime,
        risk_config=cfg.risk,
        execution_config=cfg.execution,
    )

    result = engine.run()

    report = {
        "metrics": result.metrics,
        "regime_metrics": result.regime_stats,
        "diagnostics": result.diagnostics,
    }

    out = write_strict_json(tmp_path / "report.json", report)
    text = out.read_text(encoding="utf-8")

    assert "Infinity" not in text
    assert "NaN" not in text

    parsed = json.loads(text)
    assert isinstance(parsed, dict)

    overall_pf = parsed["metrics"].get("profit_factor")
    _assert_finite_or_none(overall_pf)

    by_regime = parsed["metrics"].get("by_regime", {})
    assert by_regime == parsed["regime_metrics"].get("performance_by_regime", {})

    for _, m in by_regime.items():
        pf = m.get("profit_factor")
        _assert_finite_or_none(pf)
        if pf is not None:
            assert float(pf) >= 0.0
