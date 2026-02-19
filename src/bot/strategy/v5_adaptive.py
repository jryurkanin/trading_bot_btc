"""V5 Adaptive Strategy — adaptive macro gate + asymmetric micro regimes.

Two key improvements over V4 Core and the Benchmark:

1. **Asymmetric micro regimes**: The micro layer can *increase* exposure in
   strong trends (mult > 1.0), not just reduce it.  This addresses the core
   weakness of V4 where subtractive-only micro scaling always underperformed
   the benchmark.

2. **Adaptive macro gate**: Enter/exit/half/full thresholds adjust based on
   current volatility relative to recent history.  High vol widens the
   hysteresis band (avoids whipsaw); low vol tightens it (captures trends
   earlier).

The strategy retains all of the benchmark's strengths:
- Vol-targeted base fraction frozen at daily refresh
- Intraday increase suppression
- Same macro score computation

But adds the two improvements above via ``AdaptiveMacroGate`` and
asymmetric ``_micro_mult()``.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import RegimeConfig
from ..features import indicators
from ..features.macro_score import MacroState
from ..features.regime import RegimeState, RuleBasedRegimeSwitcher, compute_adx, compute_chop
from ..system_log import get_system_logger
from .macro_gate import AdaptiveMacroGate
from .regime_switching_orchestrator import RegimeDecisionBundle

logger = get_system_logger("strategy.v5_adaptive")


class V5AdaptiveStrategy:
    """Adaptive macro gate + asymmetric micro regime scaling."""

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg
        self._gate = AdaptiveMacroGate(cfg)
        self._rule_switcher = RuleBasedRegimeSwitcher(
            adx_trend=cfg.adx_trend_threshold,
            adx_range=cfg.adx_range_threshold,
            chop_trend=cfg.chop_trend_threshold,
            chop_range=cfg.chop_range_threshold,
            confirmation_bars=cfg.regime_confirmation_bars,
            min_duration_hours=cfg.min_regime_duration_hours,
        )

        # Daily refresh state
        self._last_refresh_day: pd.Timestamp | None = None
        self._frozen_base_fraction: float = 0.0
        self._current_target: float = 0.0
        self._last_logged_micro_regime: RegimeState | None = None
        self._last_logged_macro_state: str | None = None

    def reset(self) -> None:
        self._gate.reset()
        self._rule_switcher.reset()
        self._last_refresh_day = None
        self._frozen_base_fraction = 0.0
        self._current_target = 0.0
        self._last_logged_micro_regime = None
        self._last_logged_macro_state = None
        logger.debug("v5_state_reset")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_timestamp_col(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"], utc=True)
        return pd.to_datetime(df.index, utc=True)

    def _closed_daily(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
        if daily_df is None or daily_df.empty:
            return pd.DataFrame(columns=daily_df.columns if daily_df is not None else [])

        d = daily_df.copy()
        d["__ts"] = self._to_timestamp_col(d)
        cutoff = pd.Timestamp(decision_ts)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        closed_cutoff = cutoff.floor("D")
        d = d[d["__ts"] < closed_cutoff].sort_values("__ts")
        return d.drop(columns=["__ts"])

    @staticmethod
    def _latest_daily_ts(daily_df: pd.DataFrame) -> pd.Timestamp | None:
        if daily_df is None or daily_df.empty:
            return None
        ts = V5AdaptiveStrategy._to_timestamp_col(daily_df)
        if isinstance(ts, pd.Series):
            if ts.empty:
                return None
            ts_last = ts.iloc[-1]
        else:
            if len(ts) == 0:
                return None
            ts_last = ts[-1]
        return pd.to_datetime(ts_last, utc=True)

    # ------------------------------------------------------------------
    # Micro regime detection
    # ------------------------------------------------------------------

    def _micro_regime(
        self,
        hourly: pd.DataFrame,
        idx: int,
        precomputed: dict[str, Any] | None = None,
    ) -> RegimeState:
        if idx < 2 or hourly.empty:
            return RegimeState.NEUTRAL

        adx = precomputed.get("adx") if precomputed else None
        chop = precomputed.get("chop") if precomputed else None
        rv = precomputed.get("realized_vol") if precomputed else None

        if isinstance(adx, pd.Series) and len(adx) > idx:
            adx_val = adx.iloc[idx]
        else:
            adx_val = compute_adx(
                hourly["high"], hourly["low"], hourly["close"], self.cfg.adx_window
            ).iloc[idx]

        if isinstance(chop, pd.Series) and len(chop) > idx:
            chop_val = chop.iloc[idx]
        else:
            chop_val = compute_chop(
                hourly["high"], hourly["low"], hourly["close"], self.cfg.chop_window
            ).iloc[idx]

        if isinstance(rv, pd.Series) and len(rv) > idx:
            rv_series = rv
        else:
            rv_series = indicators.realized_vol(
                hourly["close"].pct_change(), self.cfg.realized_vol_window
            )

        lookback = max(24, self.cfg.vol_lookback_days * 24)
        min_periods = max(30, lookback // 4)
        vol_thr_pre = precomputed.get("vol_thresholds") if precomputed else None
        if isinstance(vol_thr_pre, pd.Series) and len(vol_thr_pre) > idx:
            vol_thr = (
                float(vol_thr_pre.iloc[idx])
                if pd.notna(vol_thr_pre.iloc[idx])
                else float("inf")
            )
        else:
            vol_thr_s = rv_series.rolling(lookback, min_periods=min_periods).quantile(
                self.cfg.vol_high_threshold_quantile
            )
            val = vol_thr_s.iloc[idx] if idx < len(vol_thr_s) else None
            vol_thr = float(val) if val is not None and pd.notna(val) else float("inf")

        rv_val = (
            float(rv_series.iloc[idx])
            if idx < len(rv_series) and pd.notna(rv_series.iloc[idx])
            else 0.0
        )

        return self._rule_switcher.step(float(adx_val), float(chop_val), bool(rv_val > vol_thr))

    # ------------------------------------------------------------------
    # Asymmetric micro multiplier — can boost above 1.0 in TREND
    # ------------------------------------------------------------------

    def _micro_mult(self, regime: RegimeState) -> float:
        """Map micro regime to a multiplier.

        Unlike V4 Core which clamps all multipliers to [0, 1], V5 allows
        TREND to exceed 1.0 (up to ``v5_micro_max_mult``), enabling the
        micro layer to *increase* exposure in strong trends.
        """
        mapping = {
            RegimeState.TREND: self.cfg.v5_micro_trend_mult,
            RegimeState.RANGE: self.cfg.v5_micro_range_mult,
            RegimeState.NEUTRAL: self.cfg.v5_micro_neutral_mult,
            RegimeState.HIGH_VOL: self.cfg.v5_micro_high_vol_mult,
        }
        raw = float(mapping.get(regime, 1.0))
        return max(0.0, min(float(self.cfg.v5_micro_max_mult), raw))

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
            logger.warning("v5_no_hourly_data timestamp=%s", timestamp)
            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason="no_hourly_data",
                micro_regime=RegimeState.NEUTRAL,
                micro_reason="no_data",
                strategy_name="v5_adaptive",
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

        at_daily_refresh = self._last_refresh_day is None or ts_day != self._last_refresh_day

        # --- Daily bar data ---
        daily_closed = self._closed_daily(daily_df, ts)
        daily_bar_ts = self._latest_daily_ts(daily_closed)

        # --- Adaptive macro gate ---
        macro_state, macro_mult, macro_score, macro_components = self._gate.update(
            daily_closed, daily_bar_ts
        )
        macro_score_raw = float(macro_components.get("macro_score_raw", macro_score))
        macro_score_after_fred = float(macro_components.get("macro_score_after_fred", macro_score))
        fred_risk_off_score = float(macro_components.get("fred_risk_off_score", 0.0))
        fred_penalty_multiplier = float(macro_components.get("fred_penalty_multiplier", 1.0))

        # --- Realized vol & base fraction ---
        rv_pre = micro_precomputed.get("realized_vol") if micro_precomputed else None
        if isinstance(rv_pre, pd.Series) and len(rv_pre) > hourly_idx:
            rv_last = rv_pre.iloc[hourly_idx]
        else:
            rv_last = indicators.realized_vol(
                hourly_df["close"].pct_change(), self.cfg.realized_vol_window
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

        # --- Micro regime (asymmetric) ---
        micro_regime = self._micro_regime(hourly_df, hourly_idx, precomputed=micro_precomputed)
        micro_mult = self._micro_mult(micro_regime)

        # --- Target computation ---
        core_target = max(0.0, min(self.cfg.max_position_fraction, base_fraction * macro_mult))
        # Micro mult can now push above core_target (when > 1.0), but stays
        # within max_position_fraction.
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
        macro_reason = f"v5_adaptive_{macro_state.value.lower()}"

        gate_thresholds = self._gate.thresholds

        metadata: Dict[str, float | str | int] = {
            "realized_vol": realized_vol,
            "base_fraction": base_fraction,
            "macro_score": macro_score_after_fred,
            "macro_score_raw": macro_score_raw,
            "macro_score_after_fred": macro_score_after_fred,
            "fred_risk_off_score": fred_risk_off_score,
            "fred_penalty_multiplier": fred_penalty_multiplier,
            "fred_comp_vix_z": float(macro_components.get("fred_VIXCLS_z_level", np.nan)),
            "fred_comp_hy_oas_z": float(macro_components.get("fred_BAMLH0A0HYM2_z_level", np.nan)),
            "fred_comp_stlfsi_z": float(macro_components.get("fred_STLFSI4_z_level", np.nan)),
            "fred_comp_nfci_z": float(macro_components.get("fred_NFCI_z_level", np.nan)),
            "macro_state": macro_state.value,
            "macro_multiplier": macro_mult,
            "macro_mult": macro_mult,
            "macro_reason": macro_reason,
            "micro_regime": micro_regime.value,
            "micro_mult": micro_mult,
            "core_target": core_target,
            "desired_target": desired_target,
            "final_target": final_target,
            "macro_refresh": int(at_daily_refresh),
            "micro_cap": int(micro_mult < 1.0),
            "micro_boost": int(micro_mult > 1.0),
            "intraday_increase_suppressed": int(intraday_suppressed),
            "current_position_fraction": current_exposure,
            "base_target": base_fraction,
            "macro_mode": "v5_adaptive",
            "macro_components": str(macro_components),
            # Adaptive gate diagnostics
            "adaptive_enter_threshold": gate_thresholds.get("enter", 0.0),
            "adaptive_exit_threshold": gate_thresholds.get("exit", 0.0),
            "adaptive_half_threshold": gate_thresholds.get("half", 0.0),
            "adaptive_full_threshold": gate_thresholds.get("full", 0.0),
            "adaptive_vol_z": gate_thresholds.get("vol_z", 0.0),
            "adaptive_daily_vol": gate_thresholds.get("daily_vol", 0.0),
            # Compat fields for engine recording
            "trend_boost_active": 0,
            "boost_multiplier_applied": 1.0,
        }

        state_changed = (
            at_daily_refresh
            or micro_regime != self._last_logged_micro_regime
            or macro_state.value != self._last_logged_macro_state
        )
        if state_changed:
            logger.info(
                "v5_decision_event ts=%s refresh=%s macro_state=%s macro_mult=%.4f micro_regime=%s micro_mult=%.4f base=%.4f core=%.4f final=%.4f intraday_suppressed=%s",
                ts,
                at_daily_refresh,
                macro_state.value,
                macro_mult,
                micro_regime.value,
                micro_mult,
                base_fraction,
                core_target,
                final_target,
                intraday_suppressed,
            )

        self._last_logged_micro_regime = micro_regime
        self._last_logged_macro_state = macro_state.value

        return RegimeDecisionBundle(
            macro_risk_on=macro_on,
            macro_reason=macro_reason,
            micro_regime=micro_regime,
            micro_reason="v5_asymmetric_micro",
            strategy_name="v5_adaptive",
            base_target=base_fraction,
            regime_multiplier=micro_mult,
            regime_target=core_target,
            final_target=final_target,
            metadata=metadata,
        )
