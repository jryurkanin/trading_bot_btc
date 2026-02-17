"""Comprehensive tests for V4 core strategy, macro gate benchmark, and macro gate module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.config import RegimeConfig
from bot.features.macro_score import MacroGateStateMachine, MacroState
from bot.features.regime import RegimeState
from bot.strategy.macro_gate import V4MacroGate
from bot.strategy.regime_switching_v4_core import V4CoreStrategy
from bot.strategy.macro_gate_benchmark import MacroGateBenchmarkStrategy
from bot.strategy.regime_switching_orchestrator import RegimeDecisionBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(n_days: int = 250, base_price: float = 50000.0, trend: float = 0.001) -> pd.DataFrame:
    """Create a synthetic daily OHLCV DataFrame with a mild uptrend."""
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
    """Create a synthetic hourly OHLCV DataFrame."""
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
        "v4_macro_enter_threshold": 0.75,
        "v4_macro_exit_threshold": 0.25,
        "v4_macro_half_threshold": 0.75,
        "v4_macro_full_threshold": 1.0,
        "v4_macro_confirm_days": 1,  # fast for tests
        "v4_macro_min_on_days": 1,
        "v4_macro_min_off_days": 1,
        "v4_macro_half_multiplier": 0.5,
        "v4_macro_full_multiplier": 1.0,
        "v4_micro_mult_trend": 1.0,
        "v4_micro_mult_range": 0.75,
        "v4_micro_mult_neutral": 0.5,
        "v4_micro_mult_high_vol": 0.0,
        "target_ann_vol": 0.30,
        "max_position_fraction": 1.0,
        "realized_vol_window": 168,
    }
    defaults.update(overrides)
    return RegimeConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. MacroGate state machine transitions with v4 config
# ---------------------------------------------------------------------------

class TestV4MacroGateTransitions:
    def test_starts_off(self):
        cfg = _default_cfg()
        gate = V4MacroGate(cfg)
        assert gate.state == MacroState.OFF
        assert gate.multiplier == 0.0

    def test_transition_off_to_on_half(self):
        cfg = _default_cfg(v4_macro_confirm_days=1, v4_macro_min_off_days=1)
        gate = V4MacroGate(cfg)
        # Create a daily df that yields score >= enter_threshold but < full_threshold
        daily = _make_daily_df(250, trend=0.001)
        ts1 = pd.Timestamp("2023-09-01", tz="UTC")
        ts2 = pd.Timestamp("2023-09-02", tz="UTC")

        state1, mult1, score1, _ = gate.update(daily, ts1)
        # With uptrend, score should be reasonably high
        # After first day, could be OFF still (age check)
        state2, mult2, score2, _ = gate.update(daily, ts2)
        # After confirm_days=1 and min_off_days=1, if score >= enter_threshold, should move
        if score2 >= cfg.v4_macro_enter_threshold:
            assert state2 in (MacroState.ON_HALF, MacroState.ON_FULL)
            assert mult2 > 0

    def test_cached_on_same_daily_bar(self):
        cfg = _default_cfg()
        gate = V4MacroGate(cfg)
        daily = _make_daily_df(250)
        ts = pd.Timestamp("2023-09-01", tz="UTC")

        state1, mult1, score1, comp1 = gate.update(daily, ts)
        state2, mult2, score2, comp2 = gate.update(daily, ts)
        assert state1 == state2
        assert mult1 == mult2
        assert score1 == score2

    def test_reset(self):
        cfg = _default_cfg()
        gate = V4MacroGate(cfg)
        daily = _make_daily_df(250)
        ts = pd.Timestamp("2023-09-01", tz="UTC")
        gate.update(daily, ts)
        gate.reset()
        assert gate.state == MacroState.OFF
        assert gate.multiplier == 0.0
        assert gate.score == 0.0

    def test_multiplier_values(self):
        cfg = _default_cfg(v4_macro_half_multiplier=0.4, v4_macro_full_multiplier=0.9)
        gate = V4MacroGate(cfg)
        # Check multiplier static method
        assert MacroGateStateMachine.multiplier(MacroState.OFF) == 0.0
        assert MacroGateStateMachine.multiplier(MacroState.ON_HALF, half_multiplier=0.4) == 0.4
        assert MacroGateStateMachine.multiplier(MacroState.ON_FULL, full_multiplier=0.9) == 0.9


# ---------------------------------------------------------------------------
# 2. V4 core target logic
# ---------------------------------------------------------------------------

class TestV4CoreStrategy:
    def _make_strategy_and_data(self, **cfg_overrides):
        cfg = _default_cfg(**cfg_overrides)
        strat = V4CoreStrategy(cfg)
        hourly = _make_hourly_df(500)
        daily = _make_daily_df(250)
        hourly_indexed = hourly.set_index("timestamp").sort_index()
        daily_indexed = daily.set_index("timestamp").sort_index()
        return strat, hourly_indexed, daily_indexed, cfg

    def test_macro_off_means_target_zero(self):
        """When macro gate is OFF, final target must be 0."""
        # Use very high thresholds to keep gate OFF
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v4_macro_enter_threshold=2.0,  # impossible to meet
            v4_macro_full_threshold=2.0,
        )
        ts = hourly.index[200]
        bundle = strat.compute_target_position(
            timestamp=ts,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=200,
        )
        assert bundle.final_target == 0.0
        assert bundle.macro_risk_on is False

    def test_micro_never_increases_above_core(self):
        """Micro mult should only scale down, never above core_target."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v4_micro_mult_trend=1.0,
            v4_micro_mult_range=0.5,
            v4_micro_mult_neutral=0.3,
            v4_micro_mult_high_vol=0.0,
            v4_macro_enter_threshold=0.0,  # always on
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
        )
        # Force macro on by stepping gate
        for day_offset in range(5):
            ts = hourly.index[day_offset * 24]
            bundle = strat.compute_target_position(
                timestamp=ts,
                hourly_df=hourly,
                daily_df=daily,
                current_exposure=0.0,
                hourly_idx=day_offset * 24,
            )

        # Now check at a later point
        ts = hourly.index[200]
        bundle = strat.compute_target_position(
            timestamp=ts,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=200,
        )
        core_target = float(bundle.metadata.get("core_target", 0.0))
        final_target = bundle.final_target
        # final_target <= core_target (micro mult <= 1.0)
        assert final_target <= core_target + 1e-9

    def test_intraday_increase_suppression(self):
        """Target should not increase intraday."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v4_macro_enter_threshold=0.0,
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
            v4_micro_mult_trend=1.0,
            v4_micro_mult_range=1.0,
            v4_micro_mult_neutral=1.0,
            v4_micro_mult_high_vol=1.0,
        )
        # First call on day boundary sets frozen target
        ts0 = hourly.index[48]  # some day
        bundle0 = strat.compute_target_position(
            timestamp=ts0,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=48,
        )
        first_target = bundle0.final_target

        # Manually set a lower current_target to simulate a decrease
        strat._current_target = first_target * 0.5

        # Same day, a few hours later — should not increase back to first_target
        ts1 = hourly.index[51]
        bundle1 = strat.compute_target_position(
            timestamp=ts1,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=51,
        )
        # Should be suppressed (not higher than current_target)
        assert bundle1.final_target <= first_target * 0.5 + 1e-9

    def test_daily_refresh_allows_increase(self):
        """At daily boundary, target can increase."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v4_macro_enter_threshold=0.0,
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
            v4_micro_mult_trend=1.0,
            v4_micro_mult_range=1.0,
            v4_micro_mult_neutral=1.0,
            v4_micro_mult_high_vol=1.0,
        )
        # First day
        ts0 = hourly.index[48]
        bundle0 = strat.compute_target_position(
            timestamp=ts0,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=48,
        )
        first_target = bundle0.final_target

        # Set low current target
        strat._current_target = first_target * 0.1

        # Next day — should refresh and allow increase
        next_day_idx = 48 + 24
        if next_day_idx < len(hourly):
            ts1 = hourly.index[next_day_idx]
            bundle1 = strat.compute_target_position(
                timestamp=ts1,
                hourly_df=hourly,
                daily_df=daily,
                current_exposure=0.0,
                hourly_idx=next_day_idx,
            )
            # On daily refresh, the target is recomputed fresh
            assert bundle1.metadata.get("macro_refresh") == 1

    def test_bundle_metadata_completeness(self):
        """RegimeDecisionBundle should contain all required metadata keys."""
        strat, hourly, daily, cfg = self._make_strategy_and_data()
        ts = hourly.index[100]
        bundle = strat.compute_target_position(
            timestamp=ts,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.2,
            hourly_idx=100,
        )
        required_keys = {
            "macro_score", "macro_state", "macro_mult", "micro_mult",
            "base_fraction", "core_target", "desired_target", "final_target",
            "macro_refresh", "micro_cap", "intraday_increase_suppressed",
            "current_position_fraction",
        }
        assert required_keys.issubset(set(bundle.metadata.keys())), \
            f"Missing keys: {required_keys - set(bundle.metadata.keys())}"
        assert isinstance(bundle, RegimeDecisionBundle)
        assert bundle.strategy_name == "v4_core"

    def test_reset_clears_state(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data()
        ts = hourly.index[100]
        strat.compute_target_position(
            timestamp=ts,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=100,
        )
        strat.reset()
        assert strat._current_target == 0.0
        assert strat._frozen_base_fraction == 0.0
        assert strat._last_refresh_day is None

    def test_empty_hourly_returns_zero(self):
        cfg = _default_cfg()
        strat = V4CoreStrategy(cfg)
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        daily = _make_daily_df(100).set_index("timestamp")
        bundle = strat.compute_target_position(
            timestamp=pd.Timestamp("2023-06-01", tz="UTC"),
            hourly_df=empty,
            daily_df=daily,
            current_exposure=0.0,
        )
        assert bundle.final_target == 0.0


# ---------------------------------------------------------------------------
# 3. Benchmark strategy
# ---------------------------------------------------------------------------

class TestMacroGateBenchmark:
    def _make_strategy_and_data(self, **cfg_overrides):
        cfg = _default_cfg(**cfg_overrides)
        strat = MacroGateBenchmarkStrategy(cfg)
        hourly = _make_hourly_df(500)
        daily = _make_daily_df(250)
        hourly_indexed = hourly.set_index("timestamp").sort_index()
        daily_indexed = daily.set_index("timestamp").sort_index()
        return strat, hourly_indexed, daily_indexed, cfg

    def test_no_micro_regime_effect(self):
        """Benchmark should always use micro_mult=1.0."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v4_macro_enter_threshold=0.0,
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
        )
        for idx in [50, 100, 200, 300]:
            if idx < len(hourly):
                ts = hourly.index[idx]
                bundle = strat.compute_target_position(
                    timestamp=ts,
                    hourly_df=hourly,
                    daily_df=daily,
                    current_exposure=0.0,
                    hourly_idx=idx,
                )
                assert float(bundle.metadata.get("micro_mult", 0)) == 1.0, \
                    f"micro_mult should be 1.0, got {bundle.metadata.get('micro_mult')}"

    def test_benchmark_strategy_name(self):
        strat, hourly, daily, cfg = self._make_strategy_and_data()
        ts = hourly.index[100]
        bundle = strat.compute_target_position(
            timestamp=ts,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=100,
        )
        assert bundle.strategy_name == "macro_gate_benchmark"

    def test_intraday_suppression_still_active(self):
        """Benchmark should also suppress intraday increases."""
        strat, hourly, daily, cfg = self._make_strategy_and_data(
            v4_macro_enter_threshold=0.0,
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
        )
        ts0 = hourly.index[48]
        strat.compute_target_position(
            timestamp=ts0,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=48,
        )
        orig = strat._current_target
        strat._current_target = orig * 0.3

        ts1 = hourly.index[51]
        bundle = strat.compute_target_position(
            timestamp=ts1,
            hourly_df=hourly,
            daily_df=daily,
            current_exposure=0.0,
            hourly_idx=51,
        )
        assert bundle.final_target <= orig * 0.3 + 1e-9

    def test_benchmark_vs_v4_same_macro(self):
        """Both strategies should produce same macro state for same data."""
        cfg = _default_cfg(
            v4_macro_enter_threshold=0.0,
            v4_macro_full_threshold=0.0,
            v4_macro_confirm_days=1,
            v4_macro_min_off_days=1,
        )
        v4 = V4CoreStrategy(cfg)
        bench = MacroGateBenchmarkStrategy(cfg)
        hourly = _make_hourly_df(300).set_index("timestamp").sort_index()
        daily = _make_daily_df(200).set_index("timestamp").sort_index()

        ts = hourly.index[100]
        b_v4 = v4.compute_target_position(ts, hourly, daily, 0.0, 100)
        b_bench = bench.compute_target_position(ts, hourly, daily, 0.0, 100)

        assert b_v4.metadata["macro_state"] == b_bench.metadata["macro_state"]
        assert b_v4.metadata["macro_score"] == b_bench.metadata["macro_score"]


# ---------------------------------------------------------------------------
# 4. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_benchmark_strategy_accepted(self):
        from bot.config import BacktestConfig
        bc = BacktestConfig(strategy="macro_gate_benchmark")
        assert bc.strategy == "macro_gate_benchmark"

    def test_v4_core_strategy_rejected(self):
        from bot.config import BacktestConfig
        with pytest.raises(Exception):
            BacktestConfig(strategy="regime_switching_v4_core")

    def test_invalid_strategy_rejected(self):
        from bot.config import BacktestConfig
        with pytest.raises(Exception):
            BacktestConfig(strategy="nonexistent_strategy")

    def test_v4_config_fields_exist(self):
        cfg = RegimeConfig()
        assert hasattr(cfg, "v4_macro_enter_threshold")
        assert hasattr(cfg, "v4_macro_exit_threshold")
        assert hasattr(cfg, "v4_macro_half_threshold")
        assert hasattr(cfg, "v4_macro_full_threshold")
        assert hasattr(cfg, "v4_macro_confirm_days")
        assert hasattr(cfg, "v4_macro_min_on_days")
        assert hasattr(cfg, "v4_macro_min_off_days")
        assert hasattr(cfg, "v4_macro_half_multiplier")
        assert hasattr(cfg, "v4_macro_full_multiplier")
        assert hasattr(cfg, "v4_micro_mult_trend")
        assert hasattr(cfg, "v4_micro_mult_range")
        assert hasattr(cfg, "v4_micro_mult_neutral")
        assert hasattr(cfg, "v4_micro_mult_high_vol")
        assert hasattr(cfg, "v4_core_risk_refresh")


# ---------------------------------------------------------------------------
# 5. Attribution alignment with synthetic data
# ---------------------------------------------------------------------------

class TestV4Attribution:
    def test_macro_bucket_attribution_with_v4_decisions(self):
        """Verify macro_bucket_attribution works with v4-style decisions."""
        from bot.backtest.macro_attribution import compute_macro_bucket_attribution

        n = 200
        dates = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")

        # Build synthetic equity curve with macro_state column
        equity = 10000.0 + np.cumsum(np.random.RandomState(42).normal(0, 10, n))
        eq = pd.DataFrame({
            "equity": equity,
            "exposure": np.clip(np.random.RandomState(42).uniform(0, 0.5, n), 0, 1),
            "micro_regime": ["TREND"] * 50 + ["RANGE"] * 50 + ["NEUTRAL"] * 50 + ["HIGH_VOL"] * 50,
            "macro_risk_on": [True] * 100 + [False] * 50 + [True] * 50,
            "macro_state": ["ON_FULL"] * 50 + ["ON_HALF"] * 50 + ["OFF"] * 50 + ["ON_FULL"] * 50,
            "macro_multiplier": [1.0] * 50 + [0.5] * 50 + [0.0] * 50 + [1.0] * 50,
        }, index=dates)

        decisions = pd.DataFrame({
            "decision_applies_at": dates,
            "micro_regime": eq["micro_regime"].values,
            "macro_state": eq["macro_state"].values,
            "macro_multiplier": eq["macro_multiplier"].values,
        }, index=dates)

        trades = pd.DataFrame({
            "ts": [dates[25], dates[75], dates[150]],
            "side": ["BUY", "SELL", "BUY"],
            "notional": [500.0, 300.0, 400.0],
            "fee": [1.0, 0.75, 1.0],
        })

        report, table = compute_macro_bucket_attribution(
            eq, decisions, trades, initial_equity=10000.0
        )

        assert "buckets" in report
        # Should have entries for all three buckets
        for bucket in ("OFF", "ON_HALF", "ON_FULL"):
            assert bucket in report["buckets"], f"Missing bucket {bucket}"
            info = report["buckets"][bucket]
            assert info["time_bars"] >= 0
            assert 0.0 <= info["time_share"] <= 1.0

    def test_regime_reports_with_v4_data(self):
        """Verify performance_by_regime works with v4-like equity data."""
        from bot.backtest.regime_reports import performance_by_regime, time_in_regime

        n = 200
        dates = pd.date_range("2023-06-01", periods=n, freq="h", tz="UTC")
        equity = 10000.0 + np.cumsum(np.random.RandomState(42).normal(0, 10, n))
        eq = pd.DataFrame({
            "equity": equity,
            "micro_regime": ["TREND"] * 80 + ["RANGE"] * 60 + ["NEUTRAL"] * 40 + ["HIGH_VOL"] * 20,
        }, index=dates)

        by_regime = performance_by_regime(eq, freq_per_year=8760)
        in_regime = time_in_regime(eq)

        # Should have entries for regimes that appear
        assert len(by_regime) > 0
        assert len(in_regime) > 0
        # Time shares should sum to approximately 1.0
        total_share = sum(in_regime.values())
        assert abs(total_share - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Macro bucket attribution fix regression test
# ---------------------------------------------------------------------------

class TestMacroBucketAttributionFromEquityCurve:
    """Verify that macro_bucket_attribution uses equity_curve's own macro_state
    column rather than relying on a merge_asof from decisions_df."""

    def test_buckets_reflect_equity_curve_macro_state(self):
        """When equity_curve has macro_state column with mixed OFF/ON_HALF/ON_FULL,
        the attribution must reflect those — not collapse to all OFF."""
        from bot.backtest.macro_attribution import compute_macro_bucket_attribution

        n = 100
        ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        # First 40 bars OFF, next 30 ON_HALF, last 30 ON_FULL
        states = ["OFF"] * 40 + ["ON_HALF"] * 30 + ["ON_FULL"] * 30
        mults = [0.0] * 40 + [0.5] * 30 + [1.0] * 30
        exposures = [0.0] * 40 + [0.25] * 30 + [0.5] * 30
        equity_base = 10000.0
        equities = [equity_base + i * 1.0 for i in range(n)]

        eq = pd.DataFrame({
            "equity": equities,
            "exposure": exposures,
            "macro_state": states,
            "macro_multiplier": mults,
        }, index=ts)
        eq.index.name = "timestamp"

        # decisions_df can be None — the fix should use equity_curve directly
        report, table = compute_macro_bucket_attribution(eq, None, None, initial_equity=equity_base)

        buckets = report["buckets"]
        assert buckets["OFF"]["time_bars"] == 40
        assert buckets["ON_HALF"]["time_bars"] == 30
        assert buckets["ON_FULL"]["time_bars"] == 30
        # Avg exposure should reflect actual exposures
        assert buckets["OFF"]["avg_exposure"] == pytest.approx(0.0, abs=0.01)
        assert buckets["ON_HALF"]["avg_exposure"] == pytest.approx(0.25, abs=0.01)
        assert buckets["ON_FULL"]["avg_exposure"] == pytest.approx(0.5, abs=0.01)

    def test_buckets_prefer_decisions_when_provided(self):
        """When decisions_df is provided, macro bucket attribution should use it."""
        from bot.backtest.macro_attribution import compute_macro_bucket_attribution

        n = 60
        ts = pd.date_range("2024-06-01", periods=n, freq="h", tz="UTC")
        states = ["OFF"] * 20 + ["ON_FULL"] * 40
        mults = [0.0] * 20 + [1.0] * 40

        eq = pd.DataFrame({
            "equity": [10000 + i for i in range(n)],
            "exposure": [0.0] * 20 + [0.5] * 40,
            "macro_state": states,
            "macro_multiplier": mults,
        }, index=ts)
        eq.index.name = "timestamp"

        # decisions_df deliberately conflicts (all OFF) and should take precedence.
        dec = pd.DataFrame({
            "decision_applies_at": ts,
            "macro_state": ["OFF"] * n,
            "macro_multiplier": [0.0] * n,
        }, index=ts)
        dec.index.name = "timestamp"

        report, table = compute_macro_bucket_attribution(eq, dec, None, initial_equity=10000)

        buckets = report["buckets"]
        assert buckets["OFF"]["time_bars"] == 60
        assert buckets["ON_HALF"]["time_bars"] == 0
        assert buckets["ON_FULL"]["time_bars"] == 0
        assert "macro_bucket_from_decisions" in report.get("warnings", [])
