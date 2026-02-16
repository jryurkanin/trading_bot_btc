"""Macro Gate Benchmark — same as V4 core but without micro regime scaling.

This strategy measures the pure value of the macro gate by removing
micro-regime modulation.  micro_mult is always 1.0.  Intraday increase
suppression is still applied.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import RegimeConfig
from ..features import indicators
from ..features.macro_score import MacroState
from ..features.regime import RegimeState
from .macro_gate import V4MacroGate
from .regime_switching_orchestrator import RegimeDecisionBundle


class MacroGateBenchmarkStrategy:
    """Benchmark strategy: macro gate only, no micro regime scaling."""

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg
        self._gate = V4MacroGate(cfg)

        # Daily refresh state
        self._last_refresh_day: pd.Timestamp | None = None
        self._frozen_base_fraction: float = 0.0
        self._current_target: float = 0.0

        # Cache closed-daily slices once per UTC day to avoid repeated dataframe copies.
        self._daily_ts_cache: dict[int, pd.DataFrame] = {}
        self._daily_last_ts_cache: dict[int, pd.Timestamp | None] = {}
        self._daily_index_cache_sig: tuple[int, int] | None = None
        self._daily_index_values: np.ndarray | None = None

    def reset(self) -> None:
        self._gate.reset()
        self._last_refresh_day = None
        self._frozen_base_fraction = 0.0
        self._current_target = 0.0
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()
        self._daily_index_cache_sig = None
        self._daily_index_values = None

    # ------------------------------------------------------------------
    def runtime_state(self) -> dict:
        state = self._gate._cached_state
        return {
            "gate": {
                "cached_state": state.value if isinstance(state, MacroState) else str(state),
                "cached_multiplier": float(self._gate._cached_multiplier),
                "cached_score": float(self._gate._cached_score),
                "cached_components": dict(self._gate._cached_components),
                "last_daily_ts": str(self._gate._last_daily_ts) if self._gate._last_daily_ts is not None else None,
            },
            "last_refresh_day": str(self._last_refresh_day) if self._last_refresh_day is not None else None,
            "frozen_base_fraction": float(self._frozen_base_fraction),
            "current_target": float(self._current_target),
        }

    def load_runtime_state(self, payload: dict | None) -> None:
        if not isinstance(payload, dict):
            return

        if isinstance(payload.get("current_target"), (int, float)):
            self._current_target = float(payload.get("current_target"))
        if isinstance(payload.get("frozen_base_fraction"), (int, float)):
            self._frozen_base_fraction = float(payload.get("frozen_base_fraction"))

        last_refresh = payload.get("last_refresh_day")
        if last_refresh:
            try:
                self._last_refresh_day = pd.Timestamp(last_refresh)
            except Exception:
                self._last_refresh_day = None

        gate_state = payload.get("gate")
        if isinstance(gate_state, dict):
            raw_state = gate_state.get("cached_state")
            if isinstance(raw_state, str):
                try:
                    self._gate._cached_state = MacroState(raw_state)
                except Exception:
                    pass
            elif isinstance(raw_state, MacroState):
                self._gate._cached_state = raw_state

            mult = gate_state.get("cached_multiplier")
            if isinstance(mult, (int, float)):
                self._gate._cached_multiplier = float(mult)

            score = gate_state.get("cached_score")
            if isinstance(score, (int, float)):
                self._gate._cached_score = float(score)

            comp = gate_state.get("cached_components")
            if isinstance(comp, dict):
                self._gate._cached_components = comp

            lt = gate_state.get("last_daily_ts")
            if lt:
                try:
                    self._gate._last_daily_ts = pd.Timestamp(lt)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Helpers — same daily-bar logic as V4CoreStrategy
    # ------------------------------------------------------------------

    @staticmethod
    def _to_timestamp_col(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"], utc=True)
        return pd.to_datetime(df.index, utc=True)

    @staticmethod
    def _day_cache_key(decision_ts: pd.Timestamp) -> int:
        ts = pd.Timestamp(decision_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.floor("D").value)

    def _ensure_daily_index_cache(self, daily_df: pd.DataFrame) -> None:
        if daily_df is None or daily_df.empty:
            self._daily_index_cache_sig = None
            self._daily_index_values = None
            return

        ts = self._to_timestamp_col(daily_df)
        if len(ts):
            ts_last = ts.iloc[-1] if isinstance(ts, pd.Series) else ts[-1]
            last_val = int(pd.Timestamp(ts_last).value)
        else:
            last_val = 0
        sig = (int(len(daily_df)), last_val)
        if self._daily_index_cache_sig == sig and self._daily_index_values is not None:
            return

        self._daily_index_values = ts.to_numpy(dtype="datetime64[ns]")
        self._daily_index_cache_sig = sig
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()

    def _closed_daily_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
        if daily_df is None or daily_df.empty:
            return pd.DataFrame(columns=daily_df.columns if daily_df is not None else [])

        self._ensure_daily_index_cache(daily_df)
        key = self._day_cache_key(decision_ts)
        if key in self._daily_ts_cache:
            return self._daily_ts_cache[key]

        cutoff = pd.Timestamp(decision_ts)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        closed_cutoff = cutoff.floor("D")

        idx_vals = self._daily_index_values
        if idx_vals is None:
            d = pd.DataFrame(columns=daily_df.columns)
        else:
            pos = int(np.searchsorted(idx_vals, closed_cutoff.to_numpy(), side="left"))
            d = daily_df.iloc[:pos]

        self._daily_ts_cache[key] = d
        return d

    def _latest_daily_ts_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.Timestamp | None:
        key = self._day_cache_key(decision_ts)
        if key in self._daily_last_ts_cache:
            return self._daily_last_ts_cache[key]

        closed = self._closed_daily_cached(daily_df, decision_ts)
        if closed.empty:
            self._daily_last_ts_cache[key] = None
            return None

        ts = self._to_timestamp_col(closed)
        ts_last = ts.iloc[-1] if isinstance(ts, pd.Series) else ts[-1]
        last = pd.to_datetime(ts_last, utc=True)
        self._daily_last_ts_cache[key] = last
        return last

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_target_position(
        self,
        timestamp: pd.Timestamp,
        hourly_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        current_exposure: float,
        hourly_idx: int | None = None,
        micro_precomputed: dict[str, Any] | None = None,
    ) -> RegimeDecisionBundle:
        if hourly_df.empty:
            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason="no_hourly_data",
                micro_regime=RegimeState.NEUTRAL,
                micro_reason="no_data",
                strategy_name="macro_gate_benchmark",
                base_target=0.0,
                regime_multiplier=0.0,
                regime_target=0.0,
                final_target=0.0,
                metadata={"timestamp": str(timestamp)},
            )

        if hourly_idx is None:
            hourly_idx = len(hourly_df) - 1
        hourly_idx = max(0, min(int(hourly_idx), len(hourly_df) - 1))

        # Normalize timestamp to UTC
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts_day = ts.floor("D")

        # Determine if this is a daily refresh boundary
        at_daily_refresh = self._last_refresh_day is None or ts_day != self._last_refresh_day

        # --- Daily bar data ---
        daily_closed = self._closed_daily_cached(daily_df, ts)
        daily_bar_ts = self._latest_daily_ts_cached(daily_df, ts)

        # --- Macro gate ---
        macro_state, macro_mult, macro_score, macro_components = self._gate.update(
            daily_closed, daily_bar_ts
        )

        # --- Realized vol & base fraction ---
        rv_pre = micro_precomputed.get("realized_vol") if micro_precomputed else None
        if isinstance(rv_pre, pd.Series) and len(rv_pre) > hourly_idx:
            rv_last = rv_pre.iloc[hourly_idx]
        else:
            rv_last = indicators.realized_vol(
                hourly_df["close"].pct_change(),
                self.cfg.realized_vol_window,
                backend=str((micro_precomputed or {}).get("acceleration_backend", "cpu")),
            ).iloc[hourly_idx]

        realized_vol = float(rv_last) if rv_last is not None and pd.notna(rv_last) else 0.0

        if at_daily_refresh:
            if realized_vol > 0:
                self._frozen_base_fraction = min(
                    self.cfg.target_ann_vol / realized_vol,
                    self.cfg.max_position_fraction,
                )
            else:
                self._frozen_base_fraction = 0.0
            self._last_refresh_day = ts_day

        base_fraction = self._frozen_base_fraction

        # --- No micro regime scaling (benchmark) ---
        micro_mult = 1.0

        # --- Target computation ---
        core_target = max(0.0, min(self.cfg.max_position_fraction, base_fraction * macro_mult))
        desired_target = max(0.0, min(self.cfg.max_position_fraction, core_target * micro_mult))

        # Intraday increase suppression
        intraday_suppressed = False
        if not at_daily_refresh and desired_target > self._current_target:
            desired_target = self._current_target
            intraday_suppressed = True

        final_target = max(0.0, min(self.cfg.max_position_fraction, desired_target))
        self._current_target = final_target

        # --- Build metadata ---
        macro_on = macro_mult > 0
        macro_reason = f"v4_bench_{macro_state.value.lower()}"

        metadata: Dict[str, float | str | int] = {
            "realized_vol": realized_vol,
            "base_fraction": base_fraction,
            "macro_score": macro_score,
            "macro_state": macro_state.value,
            "macro_multiplier": macro_mult,
            "macro_mult": macro_mult,
            "macro_reason": macro_reason,
            "micro_regime": "NEUTRAL",
            "micro_mult": micro_mult,
            "core_target": core_target,
            "desired_target": desired_target,
            "final_target": final_target,
            "macro_refresh": int(at_daily_refresh),
            "micro_cap": 0,
            "intraday_increase_suppressed": int(intraday_suppressed),
            "current_position_fraction": current_exposure,
            "base_target": base_fraction,
            "macro_mode": "macro_gate_benchmark",
            "macro_components": str(macro_components),
            # Compat fields for engine recording
            "trend_boost_active": 0,
            "boost_multiplier_applied": 1.0,
        }

        return RegimeDecisionBundle(
            macro_risk_on=macro_on,
            macro_reason=macro_reason,
            micro_regime=RegimeState.NEUTRAL,
            micro_reason="benchmark_no_micro",
            strategy_name="macro_gate_benchmark",
            base_target=base_fraction,
            regime_multiplier=1.0,
            regime_target=core_target,
            final_target=final_target,
            metadata=metadata,
        )
