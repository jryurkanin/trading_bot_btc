from __future__ import annotations

import pandas as pd
import pytest

import bot.strategy.regime_switching_orchestrator as orch_mod
from bot.config import RegimeConfig
from bot.features.macro_score import MacroScoreResult
from bot.features.regime import RegimeState
from bot.strategy.regime_switching_orchestrator import RegimeSwitchingOrchestrator


def _hourly_df() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=72, freq="h", tz="UTC")
    close = pd.Series([100.0 + i * 0.1 for i in range(len(ts))])
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


def _daily_df() -> pd.DataFrame:
    ts = pd.date_range("2025-10-01", periods=260, freq="D", tz="UTC")
    close = pd.Series([80.0 + i * 0.25 for i in range(len(ts))])
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 10_000.0,
        }
    )


def test_v3_stateful_gate_bucket_scaling_and_directional_boost(monkeypatch):
    cfg = RegimeConfig()
    cfg.macro_mode = "stateful_gate"
    cfg.trend_boost_enabled = True
    cfg.trend_boost_multiplier = 1.5
    cfg.trend_boost_confirm_days = 1
    cfg.trend_boost_min_on_days = 1
    cfg.trend_boost_min_off_days = 1
    cfg.macro_confirm_days = 1
    cfg.macro_min_on_days = 1
    cfg.macro_min_off_days = 1
    cfg.macro_enter_threshold = 0.75
    cfg.macro_half_threshold = 0.75
    cfg.macro_full_threshold = 1.0
    cfg.target_ann_vol = 0.25
    cfg.max_position_fraction = 1.0
    cfg.enable_hourly_overlay = False

    orch = RegimeSwitchingOrchestrator(cfg)

    scores = iter([0.75, 1.0])  # ON_HALF then ON_FULL

    def fake_macro_result(_daily, _cfg):
        s = float(next(scores))
        return MacroScoreResult(score=s, components={"mock": s}, multiplier=s, enabled_components=["mock"])

    monkeypatch.setattr(orch_mod, "macro_result", fake_macro_result)
    monkeypatch.setattr(orch, "_micro_regime", lambda _h, _idx: RegimeState.TREND)
    monkeypatch.setattr(orch, "_realized_vol", lambda h: pd.Series([0.25] * len(h)))
    monkeypatch.setattr(orch, "_core_momentum_ratio", lambda _d: 1.0)
    monkeypatch.setattr(
        orch,
        "_daily_directional_inputs",
        lambda _d: {
            "daily_adx": 40.0,
            "plus_di": 30.0,
            "minus_di": 10.0,
            "sma200": 100.0,
            "sma50": 110.0,
            "sma50_slope": 1.0,
            "daily_close": 120.0,
        },
    )

    hourly = _hourly_df()
    daily = _daily_df()

    # First timestamp -> latest closed daily bar is day N-1, score 0.75 => ON_HALF, no boost.
    b1 = orch.compute_target_position(
        timestamp=pd.Timestamp("2026-05-15T12:00:00Z"),
        hourly_df=hourly,
        daily_df=daily,
        current_exposure=0.0,
    )
    assert b1.metadata.get("macro_state") == "ON_HALF"
    assert float(b1.final_target) == pytest.approx(0.5, rel=1e-9, abs=1e-9)
    assert int(b1.metadata.get("trend_boost_active", 0)) == 0

    # Next day -> new closed daily bar, score 1.0 => ON_FULL, boost condition true.
    b2 = orch.compute_target_position(
        timestamp=pd.Timestamp("2026-05-16T12:00:00Z"),
        hourly_df=hourly,
        daily_df=daily,
        current_exposure=0.0,
    )
    assert b2.metadata.get("macro_state") == "ON_FULL"
    assert int(b2.metadata.get("trend_boost_active", 0)) == 1
    assert float(b2.metadata.get("boost_multiplier_applied", 1.0)) == 1.5
    assert 0.0 <= float(b2.final_target) <= 1.0
    assert float(b2.final_target) == pytest.approx(1.0, rel=1e-9, abs=1e-9)  # clamped after boost


def test_orchestrator_runtime_state_roundtrip():
    cfg = RegimeConfig()
    cfg.macro_mode = "stateful_gate"
    cfg.macro_confirm_days = 1
    cfg.macro_min_on_days = 1
    cfg.macro_min_off_days = 1

    o1 = RegimeSwitchingOrchestrator(cfg)
    d0 = pd.Timestamp("2026-05-01", tz="UTC")
    o1._macro_gate.step(1.0, d0)

    state = o1.runtime_state()

    o2 = RegimeSwitchingOrchestrator(cfg)
    o2.load_runtime_state(state)

    s1 = o1.runtime_state()["macro_gate"]
    s2 = o2.runtime_state()["macro_gate"]

    assert s1["state"] == s2["state"]
    assert s1["state_age_days"] == s2["state_age_days"]
    assert s1["last_daily_ts"] == s2["last_daily_ts"]
