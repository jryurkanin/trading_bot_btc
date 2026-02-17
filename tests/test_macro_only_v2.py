from __future__ import annotations

import json

import pandas as pd
import pytest

from bot.config import RegimeConfig
from bot.features.macro_signals import (
    MacroStrength,
    macro_mom_6_12_signal,
    macro_sma200_band_signal,
)
from bot.features.vol_sizing import sized_weight
from bot.features.macro_score import MacroState
from bot.strategy.macro_gate_state import MacroGateV2
from bot.strategy.drawdown_breaker import DrawdownBreaker
from bot.backtest.macro_attribution import compute_macro_bucket_attribution
from bot.strategy.macro_only_v2 import MacroOnlyV2Strategy
from bot.strategy.regime_switching_orchestrator import RegimeDecisionBundle



def _daily_df_for_sma_test(start: str = "2024-01-01", n: int = 230, base: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = float(base) + pd.Series([0.02 * (i % 5) for i in range(n)], dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        }
    )


def _daily_df_for_mom_test(n: int = 390, first: float = 500.0, trough: float = 150.0) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    first_segment = pd.Series([first - (first - trough) * (i / 209) for i in range(210)], dtype=float)
    second_segment = pd.Series([trough + (trough + 40) * ((i + 1) / 180) for i in range(180)])
    close = pd.concat([first_segment, second_segment], ignore_index=True)
    close = close.iloc[:n]
    return pd.DataFrame(
        {
            "timestamp": ts[: n],
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        }
    )


# ---------------------------------------------------------------------------
# 1. Macro signals
# ---------------------------------------------------------------------------

def test_macro_signal_sma200_band_modes():
    cfg = RegimeConfig(sma200_entry_band=0.05, sma200_exit_band=0.0)
    base = _daily_df_for_sma_test(n=240)

    off_df = base.copy()
    off_df.loc[off_df.index[-1], "close"] = 98.0
    assert macro_sma200_band_signal(off_df, cfg) == MacroStrength.OFF

    half_df = base.copy()
    half_df.loc[half_df.index[-1], "close"] = 104.0
    assert macro_sma200_band_signal(half_df, cfg) == MacroStrength.ON_HALF

    full_df = base.copy()
    full_df.loc[full_df.index[-1], "close"] = 108.0
    assert macro_sma200_band_signal(full_df, cfg) == MacroStrength.ON_FULL


def test_macro_signal_momentum_modes():
    cfg = RegimeConfig(mom_6m_days=180, mom_12m_days=365)
    # both positive -> ON_FULL
    up = _daily_df_for_sma_test(n=390, base=100.0)
    up.loc[up.index[-1], "close"] = 1800.0
    assert macro_mom_6_12_signal(up, cfg) == MacroStrength.ON_FULL

    # only 6m positive -> ON_HALF
    mixed = _daily_df_for_mom_test()
    assert macro_mom_6_12_signal(mixed, cfg) == MacroStrength.ON_HALF

    # none positive -> OFF
    down = _daily_df_for_sma_test(n=390, base=500.0)
    down["close"] = down["close"].iloc[::-1].values
    assert macro_mom_6_12_signal(down, cfg) == MacroStrength.OFF


# ---------------------------------------------------------------------------
# 2. Macro gate hysteresis
# ---------------------------------------------------------------------------


def test_macro_gate_v2_hysteresis_confirm_and_recovery():
    gate = MacroGateV2(confirm_days=2, min_on_days=2, min_off_days=1)

    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    # stay OFF while signal not confirmed
    assert gate.step(MacroStrength.OFF, t0) == MacroState.OFF
    assert gate.step(MacroStrength.ON_HALF, t0 + pd.Timedelta(days=1)) == MacroState.OFF
    assert gate.step(MacroStrength.ON_HALF, t0 + pd.Timedelta(days=2)) == MacroState.ON_HALF

    # de-escalate requires consecutive weak signal + age constraint
    assert gate.step(MacroStrength.ON_FULL, t0 + pd.Timedelta(days=3)) == MacroState.ON_HALF
    assert gate.step(MacroStrength.ON_FULL, t0 + pd.Timedelta(days=4)) == MacroState.ON_FULL

    # clear off signal long enough to exit
    assert gate.step(MacroStrength.OFF, t0 + pd.Timedelta(days=5)) == MacroState.ON_FULL
    assert gate.step(MacroStrength.OFF, t0 + pd.Timedelta(days=6)) == MacroState.OFF


# ---------------------------------------------------------------------------
# 3. Vol sizing
# ---------------------------------------------------------------------------


def test_vol_sizing_clamps_no_leverage():
    cfg = RegimeConfig(
        macro2_target_ann_vol_half=0.30,
        macro2_target_ann_vol_full=0.60,
        macro2_vol_floor=0.05,
        macro2_weight_half=0.50,
        macro2_weight_full=1.00,
        macro2_weight_off=0.0,
    )

    # direct state weights (no vol targeting)
    assert sized_weight(state=MacroState.OFF, realized_vol=0.10, mode="none", cfg=cfg) == pytest.approx(0.0)
    assert sized_weight(state=MacroState.ON_HALF, realized_vol=0.10, mode="none", cfg=cfg) == pytest.approx(0.50)
    assert sized_weight(state=MacroState.ON_FULL, realized_vol=0.10, mode="none", cfg=cfg) == pytest.approx(1.0)

    # inverse-vol mode clamps at 1.0 on low realized vol
    assert sized_weight(state=MacroState.ON_FULL, realized_vol=0.01, mode="inverse_vol", cfg=cfg) == pytest.approx(1.0)

    # inverse-vol mode maps full/half targets
    assert sized_weight(state=MacroState.ON_HALF, realized_vol=0.60, mode="inverse_vol", cfg=cfg) == pytest.approx(0.5)
    assert sized_weight(state=MacroState.ON_FULL, realized_vol=0.60, mode="inverse_vol", cfg=cfg) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 4. Drawdown breaker
# ---------------------------------------------------------------------------


def test_drawdown_breaker_activate_and_reentry_logic():
    breaker = DrawdownBreaker(
        enabled=True,
        threshold=0.10,
        cooldown_days=2,
        reentry_confirm_days=2,
        safe_weight=0.0,
    )

    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    # start above hwm
    raw = breaker.step(equity=1.0, daily_ts=t0, macro_state=MacroState.ON_FULL, raw_target=0.8)
    assert raw == 0.8
    assert not breaker.active

    # drawdown breaches threshold -> activate
    v = breaker.step(equity=0.88, daily_ts=t0 + pd.Timedelta(days=1), macro_state=MacroState.ON_FULL, raw_target=0.8)
    assert v == pytest.approx(0.0)
    assert breaker.active

    # still cooling down
    v = breaker.step(equity=0.89, daily_ts=t0 + pd.Timedelta(days=2), macro_state=MacroState.ON_FULL, raw_target=0.8)
    assert v == pytest.approx(0.0)
    assert breaker.active

    # reentry confirm allows recovery
    v = breaker.step(equity=0.90, daily_ts=t0 + pd.Timedelta(days=3), macro_state=MacroState.ON_FULL, raw_target=0.8)
    assert v == 0.8
    assert not breaker.active


# ---------------------------------------------------------------------------
# 5. Attribution
# ---------------------------------------------------------------------------


def test_macro_attribution_uses_decision_asof_for_trade_buckets():
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="h", tz="UTC")

    equity = pd.DataFrame(
        {
            "timestamp": ts,
            "equity": [10_000, 10_020, 10_030, 10_015, 10_060, 10_090],
            "exposure": [0.0, 0.1, 0.5, 0.45, 1.0, 0.9],
        }
    ).set_index("timestamp")

    decisions = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[2], ts[4]],
            "decision_applies_at": [ts[0], ts[2], ts[4]],
            "macro_state": ["OFF", "ON_HALF", "ON_FULL"],
            "macro_multiplier": [0.0, 0.5, 1.0],
        }
    ).set_index("timestamp")

    trades = pd.DataFrame(
        {
            "ts": [ts[0] + pd.Timedelta(minutes=30), ts[2] + pd.Timedelta(minutes=30), ts[4] + pd.Timedelta(minutes=30)],
            "fee": [1.0, 2.0, 3.0],
            "notional": [100.0, 200.0, 300.0],
        }
    )

    report, _ = compute_macro_bucket_attribution(
        equity,
        decisions,
        trades,
        initial_equity=10_000.0,
    )

    buckets = report.get("buckets", {})
    assert buckets["OFF"]["trade_count"] == 1
    assert buckets["ON_HALF"]["trade_count"] == 1
    assert buckets["ON_FULL"]["trade_count"] == 1
    assert buckets["OFF"]["fees"] == pytest.approx(1.0)
    assert buckets["ON_HALF"]["fees"] == pytest.approx(2.0)
    assert buckets["ON_FULL"]["fees"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 6. Strategy end-to-end
# ---------------------------------------------------------------------------


def _make_hourly_and_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_dates = pd.date_range("2024-01-01", periods=320, freq="D", tz="UTC")
    daily_close = 10000 + pd.Series(range(320), dtype=float)
    daily = pd.DataFrame(
        {
            "timestamp": daily_dates,
            "open": daily_close,
            "high": daily_close + 2,
            "low": daily_close - 2,
            "close": daily_close,
            "volume": 1000.0,
        }
    )

    hourly_dates = pd.date_range("2024-01-01", periods=320 * 24, freq="h", tz="UTC")
    hourly_close = 10000 + pd.Series(range(len(hourly_dates)), dtype=float) / 2
    hourly = pd.DataFrame(
        {
            "timestamp": hourly_dates,
            "open": hourly_close,
            "high": hourly_close + 1,
            "low": hourly_close - 1,
            "close": hourly_close,
            "volume": 100.0,
        }
    )
    return hourly.set_index("timestamp"), daily.set_index("timestamp")


def test_macro_only_v2_strategy_end_to_end_bundle_fields():
    hourly, daily = _make_hourly_and_daily()
    cfg = RegimeConfig(
        macro2_signal_mode="sma200_and_mom",
        macro2_confirm_days=1,
        macro2_min_on_days=1,
        macro2_min_off_days=1,
        macro2_vol_mode="none",
        macro2_weight_half=0.5,
        macro2_weight_full=1.0,
        macro2_dd_enabled=False,
    )

    strategy = MacroOnlyV2Strategy(cfg)

    ts = hourly.index[24 * 220]
    bundle = strategy.compute_target_position(ts, hourly, daily, 0.0, hourly_idx=250)
    assert isinstance(bundle, RegimeDecisionBundle)
    assert bundle.strategy_name == "macro_only_v2"
    assert bundle.micro_regime.value == "NEUTRAL"
    assert bundle.regime_multiplier >= 0.0
    assert 0.0 <= bundle.final_target <= 1.0

    ts2 = hourly.index[24 * 220 + 1]
    bundle2 = strategy.compute_target_position(ts2, hourly, daily, bundle.final_target, hourly_idx=251)
    assert bundle2.final_target <= bundle.final_target

    metadata = bundle.metadata
    assert isinstance(metadata, dict)
    assert json.loads(json.dumps(metadata))
    assert metadata["signal_mode"] == "sma200_and_mom"
    assert "macro_state" in metadata
    assert "macro_multiplier" in metadata
    assert "macro_score_raw" in metadata
    assert "macro_score_after_fred" in metadata
    assert float(metadata["macro_score_after_fred"]) <= float(metadata["macro_score_raw"]) + 1e-12
    assert "macro2_target_ann_vol" in metadata
