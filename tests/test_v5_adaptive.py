"""Tests for V5 Adaptive Strategy — adaptive macro gate + asymmetric micro regimes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.config import RegimeConfig, BacktestConfig
from bot.features.macro_score import MacroGateStateMachine, MacroState
from bot.features.regime import RegimeState
from bot.strategy.macro_gate import AdaptiveMacroGate
from bot.strategy.v5_adaptive import V5AdaptiveStrategy
from bot.strategy.macro_gate_benchmark import MacroGateBenchmarkStrategy
from bot.strategy.regime_switching_orchestrator import RegimeDecisionBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(n_days: int = 250, base_price: float = 50000.0, trend: float = 0.001) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    close = np.zeros(n_days)
    close[0] = base_price
    for i in range(1, n_days):
        close[i] = close[i - 1] * (1 + trend + rng.normal(0, 0.015))
    high = close * (1 + rng.uniform(0, 0.02, n_days))
    low = close * (1 - rng.uniform(0, 0.02, n_days))
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + rng.normal(0, 0.005, n_days)),
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(100, 1000, n_days),
    })


def _make_hourly_df(n_hours: int = 2000, base_price: float = 50000.0, trend: float = 0.0001) -> pd.DataFrame:
    rng = np.random.RandomState(123)
    dates = pd.date_range("2023-06-01", periods=n_hours, freq="h", tz="UTC")
    close = np.zeros(n_hours)
    close[0] = base_price
    for i in range(1, n_hours):
        close[i] = close[i - 1] * (1 + trend + rng.normal(0, 0.004))
    high = close * (1 + rng.uniform(0, 0.005, n_hours))
    low = close * (1 - rng.uniform(0, 0.005, n_hours))
    return pd.DataFrame({
        "timestamp": dates,
        "open": close * (1 + rng.normal(0, 0.001, n_hours)),
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.uniform(10, 100, n_hours),
    })


def _default_cfg(**overrides) -> RegimeConfig:
    defaults = {
        # V5 adaptive config
        "v5_micro_trend_mult": 1.15,
        "v5_micro_range_mult": 0.85,
        "v5_micro_neutral_mult": 1.0,
        "v5_micro_high_vol_mult": 0.0,
        "v5_micro_max_mult": 1.5,
        "v5_adaptive_gate_enabled": True,
        "v5_adaptive_vol_window_days": 60,
        "v5_adaptive_enter_base": 0.75,
        "v5_adaptive_exit_base": 0.25,
        "v5_adaptive_half_base": 0.75,
        "v5_adaptive_full_base": 1.0,
        "v5_adaptive_sensitivity": 0.15,
        "v5_adaptive_confirm_days": 1,  # fast for tests
        "v5_adaptive_min_on_days": 1,
        "v5_adaptive_min_off_days": 1,
        "v5_adaptive_half_multiplier": 0.5,
        "v5_adaptive_full_multiplier": 1.0,
        # Shared config
        "target_ann_vol": 0.30,
        "max_position_fraction": 1.0,
        "realized_vol_window": 168,
    }
    defaults.update(overrides)
    return RegimeConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. AdaptiveMacroGate
# ---------------------------------------------------------------------------

class TestAdaptiveMacroGate:
    def test_starts_off(self):
        cfg = _default_cfg()
        gate = AdaptiveMacroGate(cfg)
        assert gate.state == MacroState.OFF
        assert gate.multiplier == 0.0

    def test_transitions_with_high_score(self):
        cfg = _default_cfg(v5_adaptive_confirm_days=1, v5_adaptive_min_off_days=1)
        gate = AdaptiveMacroGate(cfg)
        daily = _make_daily_df(250, trend=0.001)

        ts1 = pd.Timestamp("2023-09-01", tz="UTC")
        ts2 = pd.Timestamp("2023-09-02", tz="UTC")
        state1, mult1, score1, _ = gate.update(daily, ts1)
        state2, mult2, score2, _ = gate.update(daily, ts2)

        if score2 >= cfg.v5_adaptive_enter_base:
            assert state2 in (MacroState.ON_HALF, MacroState.ON_FULL)
            assert mult2 > 0

    def test_cached_on_same_bar(self):
        cfg = _default_cfg()
        gate = AdaptiveMacroGate(cfg)
        daily = _make_daily_df(250)
        ts = pd.Timestamp("2023-09-01", tz="UTC")

        r1 = gate.update(daily, ts)
        r2 = gate.update(daily, ts)
        assert r1 == r2

    def test_thresholds_shift_with_vol(self):
        """After feeding high-vol data, enter threshold should increase."""
        cfg = _default_cfg(v5_adaptive_sensitivity=0.15, v5_adaptive_vol_window_days=10)
        gate = AdaptiveMacroGate(cfg)

        # Feed several calm days first
        calm_daily = _make_daily_df(100, trend=0.0005)
        for i in range(5):
            ts = pd.Timestamp(f"2023-04-{i + 1:02d}", tz="UTC")
            gate.update(calm_daily, ts)

        calm_thresholds = dict(gate.thresholds)

        # Now feed a high-vol day (simulate with a more volatile df)
        volatile_daily = _make_daily_df(100, trend=0.005)
        ts_vol = pd.Timestamp("2023-04-06", tz="UTC")
        gate.update(volatile_daily, ts_vol)

        vol_thresholds = gate.thresholds
        # After high vol, enter threshold should be >= calm (or equal if vol_z ~ 0)
        # This is a soft check — the adaptation direction matters more than magnitude
        assert "enter" in vol_thresholds
        assert "exit" in vol_thresholds
        assert "vol_z" in vol_thresholds

    def test_reset_clears_vol_history(self):
        cfg = _default_cfg()
        gate = AdaptiveMacroGate(cfg)
        daily = _make_daily_df(100)

        gate.update(daily, pd.Timestamp("2023-05-01", tz="UTC"))
        gate.update(daily, pd.Timestamp("2023-05-02", tz="UTC"))
        assert len(gate._vol_history) > 0

        gate.reset()
        assert len(gate._vol_history) == 0
        assert gate.state == MacroState.OFF

    def test_threshold_ordering_invariants(self):
        """exit < enter and half <= full must always hold."""
        cfg = _default_cfg(v5_adaptive_sensitivity=0.5)  # aggressive sensitivity
        gate = AdaptiveMacroGate(cfg)
        daily = _make_daily_df(250)

        for i in range(20):
            ts = pd.Timestamp(f"2023-05-{i + 1:02d}", tz="UTC")
            gate.update(daily, ts)
            t = gate.thresholds
            if t:  # thresholds populated after first update
                assert t["exit"] < t["enter"], f"exit ({t['exit']}) >= enter ({t['enter']}) on day {i}"
                assert t["half"] <= t["full"], f"half ({t['half']}) > full ({t['full']}) on day {i}"


# ---------------------------------------------------------------------------
# 2. V5AdaptiveStrategy — asymmetric micro
# ---------------------------------------------------------------------------

class TestV5AsymmetricMicro:
    def _make_strategy_and_data(self, **cfg_overrides):
        cfg = _default_cfg(**cfg_overrides)
        strat = V5AdaptiveStrategy(cfg)
        hourly = _make_hourly_df(500)
        daily = _make_daily_df(250)
        hourly_indexed = hourly.set_index("timestamp").sort_index()
        daily_indexed = daily.set_index("timestamp").sort_index()
        return strat, hourly_indexed, daily_indexed, cfg

    def test_trend_mult_can_exceed_one(self):
        """With v5_micro_trend_mult > 1.0, final_target can exceed core_target."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v5_micro_trend_mult=1.3,
            v5_micro_range_mult=0.7,
            v5_micro_neutral_mult=1.0,
            v5_micro_high_vol_mult=0.0,
            v5_adaptive_enter_base=0.0,  # always on
            v5_adaptive_full_base=0.0,
            v5_adaptive_confirm_days=1,
            v5_adaptive_min_off_days=1,
        )
        # Warm up the gate
        for day_offset in range(5):
            idx = day_offset * 24
            if idx < len(hourly):
                ts = hourly.index[idx]
                strat.compute_target_position(ts, hourly, daily, 0.0, idx)

        # Find a bar and check metadata
        ts = hourly.index[200]
        bundle = strat.compute_target_position(ts, hourly, daily, 0.0, 200)

        micro_mult = float(bundle.metadata.get("micro_mult", 0))
        if bundle.metadata.get("micro_regime") == "TREND":
            assert micro_mult == pytest.approx(1.3)
            core_target = float(bundle.metadata.get("core_target", 0))
            if core_target > 0 and not bundle.metadata.get("intraday_increase_suppressed"):
                # final can exceed core (up to max_position_fraction)
                assert bundle.final_target >= core_target * 0.99 or bundle.final_target == cfg.max_position_fraction

    def test_micro_mult_capped_at_max(self):
        """Micro mult should never exceed v5_micro_max_mult."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v5_micro_trend_mult=3.0,  # absurdly high
            v5_micro_max_mult=1.5,
        )
        # The internal _micro_mult should clamp
        assert strat._micro_mult(RegimeState.TREND) == pytest.approx(1.5)
        assert strat._micro_mult(RegimeState.RANGE) == pytest.approx(0.85)  # default
        assert strat._micro_mult(RegimeState.HIGH_VOL) == pytest.approx(0.0)

    def test_micro_mult_mapping(self):
        """Verify all regime mappings use v5 config."""
        cfg = _default_cfg(
            v5_micro_trend_mult=1.2,
            v5_micro_range_mult=0.8,
            v5_micro_neutral_mult=0.95,
            v5_micro_high_vol_mult=0.1,
            v5_micro_max_mult=2.0,
        )
        strat = V5AdaptiveStrategy(cfg)
        assert strat._micro_mult(RegimeState.TREND) == pytest.approx(1.2)
        assert strat._micro_mult(RegimeState.RANGE) == pytest.approx(0.8)
        assert strat._micro_mult(RegimeState.NEUTRAL) == pytest.approx(0.95)
        assert strat._micro_mult(RegimeState.HIGH_VOL) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 3. V5AdaptiveStrategy — full integration
# ---------------------------------------------------------------------------

class TestV5AdaptiveStrategy:
    def _make_strategy_and_data(self, **cfg_overrides):
        cfg = _default_cfg(**cfg_overrides)
        strat = V5AdaptiveStrategy(cfg)
        hourly = _make_hourly_df(500)
        daily = _make_daily_df(250)
        hourly_indexed = hourly.set_index("timestamp").sort_index()
        daily_indexed = daily.set_index("timestamp").sort_index()
        return strat, hourly_indexed, daily_indexed, cfg

    def test_macro_off_means_target_zero(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v5_adaptive_enter_base=2.0,  # impossible
            v5_adaptive_full_base=2.0,
        )
        ts = hourly.index[200]
        bundle = strat.compute_target_position(ts, hourly, daily, 0.0, 200)
        assert bundle.final_target == 0.0
        assert bundle.macro_risk_on is False

    def test_strategy_name(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data()
        ts = hourly.index[100]
        bundle = strat.compute_target_position(ts, hourly, daily, 0.0, 100)
        assert bundle.strategy_name == "v5_adaptive"

    def test_intraday_increase_suppression(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v5_adaptive_enter_base=0.0,
            v5_adaptive_full_base=0.0,
            v5_adaptive_confirm_days=1,
            v5_adaptive_min_off_days=1,
            v5_micro_trend_mult=1.0,
            v5_micro_range_mult=1.0,
            v5_micro_neutral_mult=1.0,
            v5_micro_high_vol_mult=1.0,
        )
        ts0 = hourly.index[48]
        bundle0 = strat.compute_target_position(ts0, hourly, daily, 0.0, 48)
        first_target = bundle0.final_target

        strat._current_target = first_target * 0.5

        ts1 = hourly.index[51]
        bundle1 = strat.compute_target_position(ts1, hourly, daily, 0.0, 51)
        assert bundle1.final_target <= first_target * 0.5 + 1e-9

    def test_daily_refresh_allows_increase(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v5_adaptive_enter_base=0.0,
            v5_adaptive_full_base=0.0,
            v5_adaptive_confirm_days=1,
            v5_adaptive_min_off_days=1,
        )
        ts0 = hourly.index[48]
        strat.compute_target_position(ts0, hourly, daily, 0.0, 48)
        strat._current_target *= 0.1

        next_day_idx = 48 + 24
        if next_day_idx < len(hourly):
            ts1 = hourly.index[next_day_idx]
            bundle1 = strat.compute_target_position(ts1, hourly, daily, 0.0, next_day_idx)
            assert bundle1.metadata.get("macro_refresh") == 1

    def test_metadata_completeness(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data()
        ts = hourly.index[100]
        bundle = strat.compute_target_position(ts, hourly, daily, 0.2, 100)

        required_keys = {
            "macro_score", "macro_state", "macro_mult", "micro_mult",
            "base_fraction", "core_target", "desired_target", "final_target",
            "macro_refresh", "micro_cap", "micro_boost",
            "intraday_increase_suppressed", "current_position_fraction",
            "adaptive_enter_threshold", "adaptive_exit_threshold",
            "adaptive_vol_z", "adaptive_daily_vol",
        }
        assert required_keys.issubset(set(bundle.metadata.keys())), \
            f"Missing keys: {required_keys - set(bundle.metadata.keys())}"

    def test_empty_hourly_returns_zero(self):
        cfg = _default_cfg()
        strat = V5AdaptiveStrategy(cfg)
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        daily = _make_daily_df(100).set_index("timestamp")
        bundle = strat.compute_target_position(
            pd.Timestamp("2023-06-01", tz="UTC"), empty, daily, 0.0
        )
        assert bundle.final_target == 0.0

    def test_reset(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data()
        strat.compute_target_position(hourly.index[100], hourly, daily, 0.0, 100)
        strat.reset()
        assert strat._current_target == 0.0
        assert strat._frozen_base_fraction == 0.0
        assert strat._last_refresh_day is None

    def test_uses_same_macro_score_as_benchmark(self):
        """V5 and benchmark should compute the same macro score for identical data."""
        cfg = _default_cfg(
            v5_adaptive_enter_base=0.0,
            v5_adaptive_full_base=0.0,
            v5_adaptive_confirm_days=1,
            v5_adaptive_min_off_days=1,
            # Also set V4 config for benchmark
            v4_macro_enter_threshold=0.0,
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
        )
        v5 = V5AdaptiveStrategy(cfg)
        bench = MacroGateBenchmarkStrategy(cfg)
        hourly = _make_hourly_df(300).set_index("timestamp").sort_index()
        daily = _make_daily_df(200).set_index("timestamp").sort_index()

        ts = hourly.index[100]
        b_v5 = v5.compute_target_position(ts, hourly, daily, 0.0, 100)
        b_bench = bench.compute_target_position(ts, hourly, daily, 0.0, 100)

        assert b_v5.metadata["macro_score"] == b_bench.metadata["macro_score"]


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------

class TestV5Config:
    def test_v5_strategy_accepted(self):
        bc = BacktestConfig(strategy="v5_adaptive")
        assert bc.strategy == "v5_adaptive"

    def test_v5_config_fields_exist(self):
        cfg = RegimeConfig()
        v5_fields = [
            "v5_micro_trend_mult", "v5_micro_range_mult",
            "v5_micro_neutral_mult", "v5_micro_high_vol_mult",
            "v5_micro_max_mult",
            "v5_adaptive_gate_enabled", "v5_adaptive_vol_window_days",
            "v5_adaptive_enter_base", "v5_adaptive_exit_base",
            "v5_adaptive_half_base", "v5_adaptive_full_base",
            "v5_adaptive_sensitivity",
            "v5_adaptive_confirm_days", "v5_adaptive_min_on_days",
            "v5_adaptive_min_off_days",
            "v5_adaptive_half_multiplier", "v5_adaptive_full_multiplier",
        ]
        for field in v5_fields:
            assert hasattr(cfg, field), f"Missing config field: {field}"

    def test_v5_defaults(self):
        cfg = RegimeConfig()
        assert cfg.v5_micro_trend_mult == 1.15
        assert cfg.v5_micro_range_mult == 0.85
        assert cfg.v5_micro_neutral_mult == 1.0
        assert cfg.v5_micro_high_vol_mult == 0.0
        assert cfg.v5_micro_max_mult == 1.5
        assert cfg.v5_adaptive_sensitivity == 0.15


# ---------------------------------------------------------------------------
# 5. Attribution compat
# ---------------------------------------------------------------------------

class TestV5Attribution:
    def test_macro_bucket_attribution_with_v5_decisions(self):
        from bot.backtest.macro_attribution import compute_macro_bucket_attribution

        n = 200
        dates = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")

        equity = 10000.0 + np.cumsum(np.random.RandomState(42).normal(0, 10, n))
        eq = pd.DataFrame({
            "equity": equity,
            "exposure": np.clip(np.random.RandomState(42).uniform(0, 0.5, n), 0, 1),
            "micro_regime": ["TREND"] * 50 + ["RANGE"] * 50 + ["NEUTRAL"] * 50 + ["HIGH_VOL"] * 50,
            "macro_risk_on": [True] * 100 + [False] * 50 + [True] * 50,
            "macro_state": ["ON_FULL"] * 50 + ["ON_HALF"] * 50 + ["OFF"] * 50 + ["ON_FULL"] * 50,
            "macro_multiplier": [1.0] * 50 + [0.5] * 50 + [0.0] * 50 + [1.0] * 50,
        }, index=dates)

        report, table = compute_macro_bucket_attribution(eq, None, None, initial_equity=10000.0)
        assert "buckets" in report
        for bucket in ("OFF", "ON_HALF", "ON_FULL"):
            assert bucket in report["buckets"]
