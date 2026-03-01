"""V4 Core Strategy — macro-gated vol-targeted position with micro regime scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..config import RegimeConfig
from ..features import indicators
from ..features.macro_score import MacroState
from ..features.regime import RegimeState, RuleBasedRegimeSwitcher, compute_adx, compute_chop
from ..system_log import get_system_logger
from .macro_gate import V4MacroGate
from .regime_switching_orchestrator import RegimeDecisionBundle

logger = get_system_logger("strategy.v4_core")


class V4CoreStrategy:
    """Regime-switching v4 core: macro gate + micro regime scaling.

    Key design choices
    ------------------
    * Macro gate (OFF / ON_HALF / ON_FULL) is the *primary* allocation driver.
    * Base fraction is vol-targeted and frozen at daily refresh.
    * Micro regimes only *reduce* risk (multipliers ≤ 1.0).
    * Intra-day increases are suppressed — only decreases allowed until
      the next daily refresh boundary (calendar-day change).
    """

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg
        self._gate = V4MacroGate(cfg)
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

        # Cache closed-daily slices once per UTC day to avoid repeated dataframe copies.
        self._daily_ts_cache: dict[int, pd.DataFrame] = {}
        self._daily_last_ts_cache: dict[int, pd.Timestamp | None] = {}
        self._daily_index_cache_sig: tuple[int, int] | None = None
        self._daily_index_values: np.ndarray | None = None
        self._acceleration_backend: str = str(getattr(cfg, "acceleration_backend", "cpu") or "cpu")

    def reset(self) -> None:
        self._gate.reset()
        self._rule_switcher.reset()
        self._last_refresh_day = None
        self._frozen_base_fraction = 0.0
        self._current_target = 0.0
        self._last_logged_micro_regime = None
        self._last_logged_macro_state = None
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()
        self._daily_index_cache_sig = None
        self._daily_index_values = None
        logger.debug("v4_state_reset")

    # ------------------------------------------------------------------
    # State persistence
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
                "state_machine": self._gate._gate.snapshot().__dict__,
            },
            "last_refresh_day": str(self._last_refresh_day) if self._last_refresh_day is not None else None,
            "frozen_base_fraction": float(self._frozen_base_fraction),
            "current_target": float(self._current_target),
            "rule_switcher": {
                "confirmed_regime": self._rule_switcher._confirmed_regime.value,
                "candidate_regime": self._rule_switcher._candidate_regime.value,
                "candidate_count": self._rule_switcher._candidate_count,
                "regime_age": self._rule_switcher._regime_age,
            },
        }

    def load_runtime_state(self, payload: dict | None) -> None:
        if not isinstance(payload, dict):
            return

        if isinstance(payload.get("current_target"), (int, float)):
            self._current_target = float(payload["current_target"])
        if isinstance(payload.get("frozen_base_fraction"), (int, float)):
            self._frozen_base_fraction = float(payload["frozen_base_fraction"])

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
            sm = gate_state.get("state_machine")
            if isinstance(sm, dict):
                self._gate._gate.restore(sm)

        rs = payload.get("rule_switcher")
        if isinstance(rs, dict):
            cr = rs.get("confirmed_regime")
            if isinstance(cr, str):
                try:
                    self._rule_switcher._confirmed_regime = RegimeState(cr)
                except Exception:
                    pass
            ca = rs.get("candidate_regime")
            if isinstance(ca, str):
                try:
                    self._rule_switcher._candidate_regime = RegimeState(ca)
                except Exception:
                    pass
            cc = rs.get("candidate_count")
            if isinstance(cc, int):
                self._rule_switcher._candidate_count = cc
            ra = rs.get("regime_age")
            if isinstance(ra, int):
                self._rule_switcher._regime_age = ra

    # ------------------------------------------------------------------
    # Helpers — mirror orchestrator's daily-bar logic
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
    # Micro regime
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
        backend = str((precomputed or {}).get("acceleration_backend", self._acceleration_backend or "cpu"))

        if isinstance(adx, pd.Series) and len(adx) > idx:
            adx_val = adx.iloc[idx]
        else:
            adx_val = compute_adx(
                hourly["high"], hourly["low"], hourly["close"], self.cfg.adx_window, backend=backend
            ).iloc[idx]

        if isinstance(chop, pd.Series) and len(chop) > idx:
            chop_val = chop.iloc[idx]
        else:
            chop_val = compute_chop(
                hourly["high"], hourly["low"], hourly["close"], self.cfg.chop_window, backend=backend
            ).iloc[idx]

        if isinstance(rv, pd.Series) and len(rv) > idx:
            rv_series = rv
        else:
            rv_series = indicators.realized_vol(
                hourly["close"].pct_change(), self.cfg.realized_vol_window, backend=backend
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
    # Micro multiplier mapping
    # ------------------------------------------------------------------

    def _micro_mult(self, regime: RegimeState) -> float:
        mapping = {
            RegimeState.TREND: self.cfg.v4_micro_mult_trend,
            RegimeState.RANGE: self.cfg.v4_micro_mult_range,
            RegimeState.NEUTRAL: self.cfg.v4_micro_mult_neutral,
            RegimeState.HIGH_VOL: self.cfg.v4_micro_mult_high_vol,
        }
        return float(min(1.0, max(0.0, mapping.get(regime, 1.0))))

    # ------------------------------------------------------------------
    # Precompute features (one-time per backtest run)
    # ------------------------------------------------------------------

    @staticmethod
    def get_precomputed_features(
        hourly_df: pd.DataFrame,
        cfg: RegimeConfig,
        *,
        backend: str = "cpu",
    ) -> dict[str, Any]:
        if hourly_df is None or hourly_df.empty:
            return {}

        high = hourly_df["high"].astype(float)
        low = hourly_df["low"].astype(float)
        close = hourly_df["close"].astype(float)

        adx = compute_adx(high, low, close, window=cfg.adx_window, backend=backend)
        chop = compute_chop(high, low, close, window=cfg.chop_window, backend=backend)
        realized_vol = indicators.realized_vol(close.pct_change(), int(cfg.realized_vol_window), backend=backend)

        lookback = max(24, int(cfg.vol_lookback_days) * 24)
        min_periods = max(30, lookback // 4)
        vol_thresholds = realized_vol.rolling(lookback, min_periods=min_periods).quantile(cfg.vol_high_threshold_quantile)

        return {
            "adx": adx,
            "chop": chop,
            "realized_vol": realized_vol,
            "vol_thresholds": vol_thresholds,
            "acceleration_backend": backend,
        }

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
            logger.warning("v4_no_hourly_data timestamp=%s", timestamp)
            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason="no_hourly_data",
                micro_regime=RegimeState.NEUTRAL,
                micro_reason="no_data",
                strategy_name="v4_core",
                base_target=0.0,
                regime_multiplier=0.0,
                regime_target=0.0,
                final_target=0.0,
                metadata={"timestamp": str(timestamp)},
            )

        if hourly_idx is None:
            hourly_idx = len(hourly_df) - 1
        hourly_idx = max(0, min(int(hourly_idx), len(hourly_df) - 1))

        if micro_precomputed is not None:
            backend_hint = micro_precomputed.get("acceleration_backend")
            if isinstance(backend_hint, str) and backend_hint:
                self._acceleration_backend = backend_hint

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
                hourly_df["close"].pct_change(), self.cfg.realized_vol_window, backend=self._acceleration_backend
            ).iloc[hourly_idx]

        realized_vol = float(rv_last) if rv_last is not None and pd.notna(rv_last) else 0.0

        if at_daily_refresh:
            # Recompute and freeze base fraction
            if realized_vol > 0:
                self._frozen_base_fraction = min(
                    self.cfg.target_ann_vol / realized_vol,
                    self.cfg.max_position_fraction,
                )
            else:
                self._frozen_base_fraction = 0.0
            self._last_refresh_day = ts_day

        base_fraction = self._frozen_base_fraction

        # --- Micro regime ---
        micro_regime = self._micro_regime(hourly_df, hourly_idx, precomputed=micro_precomputed)
        micro_mult = self._micro_mult(micro_regime)

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
        macro_reason = f"v4_gate_{macro_state.value.lower()}"

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
            "intraday_increase_suppressed": int(intraday_suppressed),
            "current_position_fraction": current_exposure,
            "base_target": base_fraction,
            "macro_mode": "v4_core",
            "macro_components": str(macro_components),
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
                "v4_decision_event ts=%s refresh=%s macro_state=%s macro_mult=%.4f micro_regime=%s micro_mult=%.4f base=%.4f core=%.4f final=%.4f intraday_suppressed=%s",
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
            micro_reason="v4_micro_scaling",
            strategy_name="v4_core",
            base_target=base_fraction,
            regime_multiplier=micro_mult,
            regime_target=core_target,
            final_target=final_target,
            metadata=metadata,
        )
