from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

from ..config import RegimeConfig
from ..features import indicators
from ..features.macro_score import MacroGateStateMachine, MacroState, macro_result
from ..features.regime import HMMRegimeSwitcher, RegimeState, RuleBasedRegimeSwitcher, compute_adx, compute_adx_di, compute_chop
from ..system_log import get_system_logger
from .sub_strategies.mean_reversion_bb import MeanReversionBBStrategy, RangeStrategyConfig
from .sub_strategies.trend_following_breakout import TrendFollowingBreakoutStrategy, TrendStrategyConfig
from .base import StrategyDecision

logger = get_system_logger("strategy.regime_switching_orchestrator")


@dataclass
class RegimeDecisionBundle:
    macro_risk_on: bool
    macro_reason: str
    micro_regime: RegimeState
    micro_reason: str
    strategy_name: str
    base_target: float
    regime_multiplier: float
    regime_target: float
    final_target: float
    metadata: Dict[str, float | str | int]


class RegimeSwitchingOrchestrator:
    """Main strategy orchestrator."""

    REGIME_MULTIPLIERS = {
        RegimeState.TREND: 1.0,
        RegimeState.RANGE: 0.75,
        RegimeState.NEUTRAL: 0.5,
        RegimeState.HIGH_VOL: 0.0,
    }

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self.rule_switcher = RuleBasedRegimeSwitcher(
            adx_trend=cfg.adx_trend_threshold,
            adx_range=cfg.adx_range_threshold,
            chop_trend=cfg.chop_trend_threshold,
            chop_range=cfg.chop_range_threshold,
            confirmation_bars=cfg.regime_confirmation_bars,
            min_duration_hours=cfg.min_regime_duration_hours,
        )
        self.hmm_switcher = HMMRegimeSwitcher(
            enabled=cfg.hmm_regime_enabled,
            n_states=cfg.hmm_n_states,
            window=cfg.hmm_window_hours,
        )
        self.range_strategy = MeanReversionBBStrategy(
            RangeStrategyConfig(
                bb_window=cfg.bb_window,
                bb_stdev=cfg.bb_stdev,
                tranche_size=cfg.range_tranche_size,
                max_exposure=cfg.range_max_exposure,
                min_time_between_trades_hours=cfg.range_min_time_between_trades_hours,
                max_trades_per_day=cfg.range_max_trades_per_day,
            )
        )
        self.trend_strategy = TrendFollowingBreakoutStrategy(
            TrendStrategyConfig(
                mode=cfg.trend_mode,
                donchian_window=cfg.donchian_window,
                ema_fast=cfg.ema_fast,
                ema_slow=cfg.ema_slow,
                atr_window=cfg.atr_window,
                atr_mult=cfg.atr_mult,
                trend_exposure_cap=cfg.trend_exposure_cap,
                vol_target_multiplier=cfg.vol_target_multiplier,
            )
        )

        # Stateful macro gate (v3)
        self._macro_gate = MacroGateStateMachine(
            enter_threshold=cfg.macro_enter_threshold,
            exit_threshold=cfg.macro_exit_threshold,
            full_threshold=cfg.macro_full_threshold,
            half_threshold=cfg.macro_half_threshold,
            confirm_days=cfg.macro_confirm_days,
            min_on_days=cfg.macro_min_on_days,
            min_off_days=cfg.macro_min_off_days,
        )

        # trend booster state (daily cadence)
        self._boost_active = False
        self._boost_state_age_days = 0
        self._boost_on_streak = 0
        self._boost_off_streak = 0
        self._boost_last_daily_ts: pd.Timestamp | None = None

        # Daily computation cache — avoids recomputing macro/daily signals
        # on every hourly bar when the daily bar hasn't changed.
        self._daily_cache: dict[str, Any] = {}

        self._last_logged_day: pd.Timestamp | None = None
        self._last_logged_macro_state: str | None = None
        self._last_logged_micro_regime: RegimeState | None = None
        self._last_logged_strategy_name: str | None = None

    def reset(self):
        self.rule_switcher.reset()
        self.range_strategy.reset()
        self.trend_strategy.reset()
        self._macro_gate.reset()
        self._boost_active = False
        self._boost_state_age_days = 0
        self._boost_on_streak = 0
        self._boost_off_streak = 0
        self._boost_last_daily_ts = None
        self._daily_cache = {}
        self._last_logged_day = None
        self._last_logged_macro_state = None
        self._last_logged_micro_regime = None
        self._last_logged_strategy_name = None
        logger.debug("orchestrator_reset")

    def runtime_state(self) -> dict[str, Any]:
        return {
            "macro_gate": self._macro_gate.snapshot().__dict__,
            "boost": {
                "active": bool(self._boost_active),
                "state_age_days": int(self._boost_state_age_days),
                "on_streak": int(self._boost_on_streak),
                "off_streak": int(self._boost_off_streak),
                "last_daily_ts": self._boost_last_daily_ts.isoformat() if self._boost_last_daily_ts is not None else None,
            },
        }

    def load_runtime_state(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return

        self._macro_gate.restore(payload.get("macro_gate") if isinstance(payload.get("macro_gate"), dict) else None)

        boost = payload.get("boost") if isinstance(payload.get("boost"), dict) else {}
        self._boost_active = bool(boost.get("active", self._boost_active))
        self._boost_state_age_days = int(boost.get("state_age_days", self._boost_state_age_days) or 0)
        self._boost_on_streak = int(boost.get("on_streak", self._boost_on_streak) or 0)
        self._boost_off_streak = int(boost.get("off_streak", self._boost_off_streak) or 0)
        ts_raw = boost.get("last_daily_ts")
        self._boost_last_daily_ts = pd.Timestamp(ts_raw) if ts_raw else self._boost_last_daily_ts

    @staticmethod
    def _to_timestamp_col(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            return ts
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

        # Daily bars are treated as period-start timestamps; only use fully closed bars.
        closed_cutoff = cutoff.floor("D")
        d = d[d["__ts"] < closed_cutoff].sort_values("__ts")
        return d.drop(columns=["__ts"]) if not d.empty else d.drop(columns=["__ts"])

    def _latest_daily_ts(self, daily_df: pd.DataFrame) -> pd.Timestamp | None:
        if daily_df is None or daily_df.empty:
            return None
        ts = self._to_timestamp_col(daily_df)
        if isinstance(ts, pd.Series):
            if ts.empty:
                return None
            ts_last = ts.iloc[-1]
        else:
            if len(ts) == 0:
                return None
            ts_last = ts[-1]
        return pd.to_datetime(ts_last, utc=True)

    def _macro_risk_binary(self, daily_df: pd.DataFrame) -> Tuple[bool, str]:
        if len(daily_df) < max(self.cfg.daily_trend_window, self.cfg.daily_momentum_window + 1):
            return False, "insufficient_daily_history"

        close = daily_df["close"]
        sma = indicators.sma(close, self.cfg.daily_trend_window).iloc[-1]

        base = close.shift(self.cfg.daily_momentum_window).iloc[-1]
        ret = (close.iloc[-1] / base) - 1.0 if pd.notna(base) and float(base) != 0 else 0.0
        mom_quant = indicators.percentile(close.pct_change(), self.cfg.daily_momentum_window, self.cfg.daily_momentum_quantile).iloc[-1]
        mom_quant = float(mom_quant) if pd.notna(mom_quant) else 0.0

        if pd.isna(sma) or close.iloc[-1] <= sma:
            return False, "daily_trend_below_sma"
        if not (ret > 0 or ret > mom_quant):
            return False, "daily_momentum_not_ok"
        return True, "risk_on"

    def _realized_vol(self, hourly: pd.DataFrame) -> pd.Series:
        return indicators.realized_vol(hourly["close"].pct_change(), self.cfg.realized_vol_window)

    def _core_momentum_ratio(self, daily_df: pd.DataFrame) -> float:
        if len(daily_df) < max(self.cfg.daily_trend_window, self.cfg.daily_momentum_window + 1):
            return 0.0
        close = daily_df["close"]
        sma = indicators.sma(close, self.cfg.daily_trend_window).iloc[-1]
        if pd.isna(sma):
            return 0.0

        base = close.shift(self.cfg.daily_momentum_window).iloc[-1]
        if pd.isna(base) or float(base) == 0.0:
            return 0.0
        momentum = float(close.iloc[-1] / base - 1.0)

        if float(close.iloc[-1]) <= float(sma):
            return 0.0
        if momentum <= 0:
            return 0.25
        if momentum < 0.02:
            return 0.5
        if momentum < 0.06:
            return 0.75
        return 1.0

    def _micro_regime(self, hourly: pd.DataFrame, idx: int, precomputed: dict[str, Any] | None = None) -> RegimeState:
        if idx < 2 or hourly.empty:
            return RegimeState.NEUTRAL

        adx = precomputed.get("adx") if precomputed else None
        chop = precomputed.get("chop") if precomputed else None
        rv = precomputed.get("realized_vol") if precomputed else None

        if isinstance(adx, pd.Series) and len(adx) > idx:
            adx_val = adx.iloc[idx]
        else:
            adx_series = compute_adx(hourly["high"], hourly["low"], hourly["close"], self.cfg.adx_window)
            adx_val = adx_series.iloc[idx]

        if isinstance(chop, pd.Series) and len(chop) > idx:
            chop_val = chop.iloc[idx]
        else:
            chop_series = compute_chop(hourly["high"], hourly["low"], hourly["close"], self.cfg.chop_window)
            chop_val = chop_series.iloc[idx]

        if isinstance(rv, pd.Series) and len(rv) > idx:
            rv_series = rv
        else:
            rv_series = self._realized_vol(hourly)

        if self.cfg.hmm_regime_enabled and len(hourly) >= self.cfg.hmm_window_hours:
            start = max(0, idx - self.cfg.hmm_window_hours + 1)

            feats_pre = precomputed.get("hmm_features") if precomputed else None
            if isinstance(feats_pre, pd.DataFrame) and len(feats_pre) > idx:
                feats = feats_pre.iloc[start : idx + 1]
            else:
                feats = pd.concat(
                    [
                        hourly[["close", "high", "low", "volume"]].pct_change().fillna(0).iloc[start : idx + 1],
                        rv_series.iloc[start : idx + 1],
                    ],
                    axis=1,
                ).fillna(0)

            if len(feats) >= 20:
                self.hmm_switcher.fit(feats.tail(self.cfg.hmm_window_hours).values)
                return self.hmm_switcher.predict_one(feats.iloc[-1].values)

        lookback = max(24, self.cfg.vol_lookback_days * 24)
        min_periods = max(30, lookback // 4)
        vol_thr_series_pre = precomputed.get("vol_thresholds") if precomputed else None
        if isinstance(vol_thr_series_pre, pd.Series) and len(vol_thr_series_pre) > idx:
            vol_thr = float(vol_thr_series_pre.iloc[idx]) if pd.notna(vol_thr_series_pre.iloc[idx]) else float("inf")
        else:
            vol_thr_series = rv_series.rolling(lookback, min_periods=min_periods).quantile(self.cfg.vol_high_threshold_quantile)
            vol_thr_val = vol_thr_series.iloc[idx] if idx < len(vol_thr_series) else None
            vol_thr = float(vol_thr_val) if vol_thr_val is not None and pd.notna(vol_thr_val) else float("inf")
        rv_val = float(rv_series.iloc[idx]) if idx < len(rv_series) and pd.notna(rv_series.iloc[idx]) else 0.0

        return self.rule_switcher.step(float(adx_val), float(chop_val), bool(rv_val > vol_thr))

    def _daily_directional_inputs(self, daily_df: pd.DataFrame) -> dict[str, float]:
        if daily_df is None or daily_df.empty:
            return {
                "daily_adx": 0.0,
                "plus_di": 0.0,
                "minus_di": 0.0,
                "sma200": 0.0,
                "sma50": 0.0,
                "sma50_slope": 0.0,
                "daily_close": 0.0,
            }

        close = daily_df["close"].astype(float)
        high = daily_df["high"].astype(float)
        low = daily_df["low"].astype(float)

        adx_s, plus_di_s, minus_di_s = compute_adx_di(high, low, close, window=self.cfg.adx_window)
        adx = float(adx_s.iloc[-1]) if len(adx_s) and pd.notna(adx_s.iloc[-1]) else 0.0
        plus_di = float(plus_di_s.iloc[-1]) if len(plus_di_s) and pd.notna(plus_di_s.iloc[-1]) else 0.0
        minus_di = float(minus_di_s.iloc[-1]) if len(minus_di_s) and pd.notna(minus_di_s.iloc[-1]) else 0.0

        sma200_s = indicators.sma(close, 200)
        sma50_s = indicators.sma(close, 50)
        sma200 = float(sma200_s.iloc[-1]) if len(sma200_s) and pd.notna(sma200_s.iloc[-1]) else 0.0
        sma50 = float(sma50_s.iloc[-1]) if len(sma50_s) and pd.notna(sma50_s.iloc[-1]) else 0.0

        lookback = max(1, int(self.cfg.trend_boost_sma50_slope_lookback_days))
        if len(sma50_s) > lookback and pd.notna(sma50_s.iloc[-1]) and pd.notna(sma50_s.iloc[-1 - lookback]):
            sma50_slope = float(sma50_s.iloc[-1] - sma50_s.iloc[-1 - lookback])
        else:
            sma50_slope = 0.0

        daily_close = float(close.iloc[-1]) if len(close) else 0.0

        return {
            "daily_adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "sma200": sma200,
            "sma50": sma50,
            "sma50_slope": sma50_slope,
            "daily_close": daily_close,
        }

    def _daily_boost_gate(self, daily_df: pd.DataFrame, micro_regime: RegimeState) -> bool:
        if self.cfg.trend_boost_regime_gate == "micro_trend":
            return micro_regime == RegimeState.TREND

        if daily_df.empty:
            return False
        close = daily_df["close"].astype(float)
        sma = indicators.sma(close, self.cfg.daily_trend_window).iloc[-1]
        if pd.isna(sma):
            return False
        return float(close.iloc[-1]) > float(sma)

    def _update_booster_state(self, daily_bar_ts: pd.Timestamp | None, condition: bool) -> bool:
        if daily_bar_ts is None:
            return bool(self._boost_active)

        bar_ts = pd.Timestamp(daily_bar_ts)
        if self._boost_last_daily_ts is not None and bar_ts == pd.Timestamp(self._boost_last_daily_ts):
            return bool(self._boost_active)

        self._boost_last_daily_ts = bar_ts
        self._boost_state_age_days += 1

        if condition:
            self._boost_on_streak += 1
            self._boost_off_streak = 0
        else:
            self._boost_off_streak += 1
            self._boost_on_streak = 0

        confirm_days = max(1, int(self.cfg.trend_boost_confirm_days))
        min_on_days = max(1, int(self.cfg.trend_boost_min_on_days))
        min_off_days = max(1, int(self.cfg.trend_boost_min_off_days))

        if self._boost_active:
            if (not condition) and self._boost_state_age_days >= min_on_days and self._boost_off_streak >= confirm_days:
                self._boost_active = False
                self._boost_state_age_days = 0
                self._boost_on_streak = 0
        else:
            if condition and self._boost_state_age_days >= min_off_days and self._boost_on_streak >= confirm_days:
                self._boost_active = True
                self._boost_state_age_days = 0
                self._boost_off_streak = 0

        return bool(self._boost_active)

    def _macro_state_and_multiplier(
        self,
        daily_closed: pd.DataFrame,
        macro_score: float,
        macro_linear_multiplier: float,
        binary_on: bool,
        daily_bar_ts: pd.Timestamp | None,
    ) -> tuple[MacroState, float, bool, str]:
        if self.cfg.macro_mode == "stateful_gate":
            state = self._macro_gate.step(macro_score, daily_bar_ts)
            mult = MacroGateStateMachine.multiplier(
                state,
                half_multiplier=self.cfg.macro_half_multiplier,
                full_multiplier=self.cfg.macro_full_multiplier,
            )
            return state, float(mult), bool(mult > 0), f"stateful_{state.value.lower()}"

        if self.cfg.macro_mode == "score":
            mult = float(max(0.0, min(1.0, macro_linear_multiplier)))
            if mult <= 0:
                state = MacroState.OFF
            elif mult < 0.999999:
                state = MacroState.ON_HALF
            else:
                state = MacroState.ON_FULL
            return state, mult, bool(mult > 0), "score_on" if mult > 0 else "score_below_threshold"

        # binary mode
        state = MacroState.ON_FULL if binary_on else MacroState.OFF
        mult = 1.0 if binary_on else 0.0
        return state, mult, bool(binary_on), "risk_on" if binary_on else "macro_off"

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
            logger.warning("orchestrator_no_hourly_data timestamp=%s", timestamp)
            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason="no_hourly_data",
                micro_regime=RegimeState.NEUTRAL,
                micro_reason="no_data",
                strategy_name="none",
                base_target=0.0,
                regime_multiplier=0.0,
                regime_target=0.0,
                final_target=0.0,
                metadata={"timestamp": str(timestamp)},
            )

        if hourly_idx is None:
            hourly_idx = len(hourly_df) - 1
        hourly_idx = max(0, min(int(hourly_idx), len(hourly_df) - 1))

        # Fast path: reuse cached daily-derived results when the signal
        # timestamp falls on the same calendar day.  All daily outputs
        # (_closed_daily, macro scoring, directional inputs, core_momentum)
        # are stored in a single _daily_cache dict keyed by "_ts_day".
        _ts_floor = pd.Timestamp(timestamp)
        if _ts_floor.tzinfo is None:
            _ts_floor = _ts_floor.tz_localize("UTC")
        else:
            _ts_floor = _ts_floor.tz_convert("UTC")
        _ts_day = _ts_floor.floor("D")
        _dc = self._daily_cache

        if _dc.get("_ts_day") == _ts_day and "macro_score" in _dc:
            # Full cache hit — reuse everything.
            daily_closed = _dc["daily_closed"]
            daily_bar_ts = _dc["daily_bar_ts"]
            macro_score = _dc["macro_score"]
            macro_score_raw = _dc.get("macro_score_raw", macro_score)
            fred_risk_off_score = _dc.get("fred_risk_off_score", 0.0)
            macro_score_after_fred = _dc.get("macro_score_after_fred", macro_score)
            macro_components = _dc["macro_components"]
            macro_components_used = _dc["macro_components_used"]
            binary_on = _dc["binary_on"]
            binary_reason = _dc["binary_reason"]
            macro_state = _dc["macro_state"]
            macro_multiplier = _dc["macro_multiplier"]
            macro_on = _dc["macro_on"]
            macro_reason = _dc["macro_reason"]
        else:
            # Day changed — recompute everything and store.
            daily_closed = self._closed_daily(daily_df, timestamp)
            daily_bar_ts = self._latest_daily_ts(daily_closed)

            macro = macro_result(daily_closed, self.cfg)
            macro_score = float(macro.score)
            macro_score_raw = float(macro.raw_score)
            fred_risk_off_score = float(macro.fred_risk_off_score)
            macro_score_after_fred = float(macro.score)
            macro_components = macro.components
            macro_components_used = macro.enabled_components

            binary_on, binary_reason = self._macro_risk_binary(daily_closed)
            macro_state, macro_multiplier, macro_on, macro_reason = self._macro_state_and_multiplier(
                daily_closed,
                macro_score=macro_score,
                macro_linear_multiplier=float(macro.multiplier),
                binary_on=binary_on,
                daily_bar_ts=daily_bar_ts,
            )
            if self.cfg.macro_mode == "binary":
                macro_reason = binary_reason

            self._daily_cache = {
                "_ts_day": _ts_day,
                "daily_bar_ts": daily_bar_ts,
                "daily_closed": daily_closed,
                "macro_score": macro_score,
                "macro_score_raw": macro_score_raw,
                "fred_risk_off_score": fred_risk_off_score,
                "macro_score_after_fred": macro_score_after_fred,
                "macro_components": macro_components,
                "macro_components_used": macro_components_used,
                "binary_on": binary_on,
                "binary_reason": binary_reason,
                "macro_state": macro_state,
                "macro_multiplier": macro_multiplier,
                "macro_on": macro_on,
                "macro_reason": macro_reason,
            }

        micro_regime = self._micro_regime(hourly_df, hourly_idx, precomputed=micro_precomputed)

        rv_pre = micro_precomputed.get("realized_vol") if micro_precomputed else None
        if isinstance(rv_pre, pd.Series) and len(rv_pre) > hourly_idx:
            rv_last = rv_pre.iloc[hourly_idx]
        else:
            rv_last = self._realized_vol(hourly_df).iloc[hourly_idx]

        realized_vol = float(rv_last) if rv_last is not None and pd.notna(rv_last) else 0.0
        if realized_vol <= 0 or pd.isna(realized_vol):
            realized_vol = 0.0
            base_target = 0.0
        else:
            base_target = min(self.cfg.target_ann_vol / realized_vol, self.cfg.max_position_fraction)

        metadata: Dict[str, float | str | int] = {
            "realized_vol": realized_vol,
            "base_target": base_target,
            "macro_mode": self.cfg.macro_mode,
            "macro_score": macro_score_after_fred,
            "macro_score_raw": macro_score_raw,
            "macro_score_after_fred": macro_score_after_fred,
            "fred_risk_off_score": fred_risk_off_score,
            "fred_penalty_multiplier": float(macro_components.get("fred_penalty_multiplier", 1.0)),
            "fred_comp_vix_z": float(macro_components.get("fred_VIXCLS_z_level", np.nan)) if isinstance(macro_components, dict) else np.nan,
            "fred_comp_hy_oas_z": float(macro_components.get("fred_BAMLH0A0HYM2_z_level", np.nan)) if isinstance(macro_components, dict) else np.nan,
            "fred_comp_stlfsi_z": float(macro_components.get("fred_STLFSI4_z_level", np.nan)) if isinstance(macro_components, dict) else np.nan,
            "fred_comp_nfci_z": float(macro_components.get("fred_NFCI_z_level", np.nan)) if isinstance(macro_components, dict) else np.nan,
            "macro_state": macro_state.value,
            "macro_multiplier": macro_multiplier,
            "macro_reason": macro_reason,
            "macro_components": str(macro_components),
            "macro_components_used": str(macro_components_used),
        }

        if not macro_on or macro_multiplier <= 0:
            metadata.update({
                "trend_boost_active": 0,
                "boost_multiplier_applied": 1.0,
            })
            strategy_name = "flat"
            state_changed = (
                self._last_logged_day != _ts_day
                or self._last_logged_macro_state != macro_state.value
                or self._last_logged_micro_regime != micro_regime
                or self._last_logged_strategy_name != strategy_name
            )
            if state_changed:
                logger.info(
                    "orchestrator_decision_event ts=%s day=%s macro_state=%s micro_regime=%s strategy=%s base_target=%.4f final_target=0.0000 reason=%s",
                    _ts_floor,
                    _ts_day,
                    macro_state.value,
                    micro_regime.value,
                    strategy_name,
                    base_target,
                    macro_reason,
                )
            self._last_logged_day = _ts_day
            self._last_logged_macro_state = macro_state.value
            self._last_logged_micro_regime = micro_regime
            self._last_logged_strategy_name = strategy_name

            return RegimeDecisionBundle(
                macro_risk_on=False,
                macro_reason=macro_reason,
                micro_regime=micro_regime,
                micro_reason="macro_off",
                strategy_name=strategy_name,
                base_target=base_target,
                regime_multiplier=0.0,
                regime_target=0.0,
                final_target=0.0,
                metadata=metadata,
            )

        strategy_ratio = 0.0
        strategy_name = ""
        overlay_adj = 0.0

        regime_mult = self.REGIME_MULTIPLIERS.get(micro_regime, 0.5)
        if micro_regime == RegimeState.HIGH_VOL and self.cfg.high_vol_cap > 0:
            regime_mult = self.cfg.high_vol_cap

        # macro multiplier scales base target before regime/sub-strategy logic.
        macro_target = base_target * macro_multiplier

        if self.cfg.legacy_substrategy_switching:
            if micro_regime == RegimeState.TREND:
                strategy_name = self.trend_strategy.name
                strategy_ratio = self.trend_strategy.compute_target(hourly_df, current_exposure, timestamp,
                                                                     idx=hourly_idx, precomputed=micro_precomputed)
                metadata.update(self.trend_strategy.signal_reason(hourly_df.iloc[:hourly_idx + 1], current_exposure, timestamp))
            elif micro_regime == RegimeState.RANGE:
                strategy_name = self.range_strategy.name
                prev_row = hourly_df.iloc[hourly_idx - 1] if hourly_idx >= 1 else None
                strategy_ratio = self.range_strategy.compute_target(
                    hourly_df.iloc[hourly_idx], prev_row, current_exposure, timestamp, hourly_df["close"],
                    idx=hourly_idx, precomputed=micro_precomputed,
                )
                metadata.update(
                    self.range_strategy.signal_reason(hourly_df.iloc[hourly_idx], prev_row, current_exposure, timestamp, hourly_df["close"],
                                                      idx=hourly_idx, precomputed=micro_precomputed)
                )
            elif micro_regime == RegimeState.HIGH_VOL:
                strategy_name = "high_vol_guard"
                strategy_ratio = 1.0 if self.cfg.high_vol_cap > 0 else 0.0
            else:
                strategy_name = "neutral"
                strategy_ratio = min(1.0, current_exposure / macro_target) if macro_target > 0 else 0.0

            regime_target = macro_target * regime_mult
            final_target = regime_target * strategy_ratio
        else:
            regime_target = macro_target * regime_mult
            if micro_regime == RegimeState.TREND and self.cfg.trend_playbook == "core_momentum_daily":
                strategy_name = "core_momentum_daily"
                if "core_momentum_ratio" in self._daily_cache:
                    core_ratio = self._daily_cache["core_momentum_ratio"]
                else:
                    core_ratio = self._core_momentum_ratio(daily_closed)
                    self._daily_cache["core_momentum_ratio"] = core_ratio
                regime_target = regime_target * core_ratio
                strategy_ratio = 1.0
                metadata["core_ratio"] = core_ratio

                if self.cfg.enable_hourly_overlay:
                    trend_signal = self.trend_strategy.compute_target(hourly_df, current_exposure, timestamp,
                                                                      idx=hourly_idx, precomputed=micro_precomputed)
                    overlay_adj = (trend_signal - 0.5) * 2.0 * self.cfg.overlay_max_adjustment
                    metadata["overlay_signal"] = trend_signal
                    metadata["overlay_adj"] = overlay_adj

                final_target = regime_target * (1.0 + overlay_adj)
            elif micro_regime == RegimeState.RANGE:
                strategy_name = self.range_strategy.name
                prev_row = hourly_df.iloc[hourly_idx - 1] if hourly_idx >= 1 else None
                strategy_ratio = self.range_strategy.compute_target(
                    hourly_df.iloc[hourly_idx], prev_row, current_exposure, timestamp, hourly_df["close"],
                    idx=hourly_idx, precomputed=micro_precomputed,
                )
                metadata.update(
                    self.range_strategy.signal_reason(hourly_df.iloc[hourly_idx], prev_row, current_exposure, timestamp, hourly_df["close"],
                                                      idx=hourly_idx, precomputed=micro_precomputed)
                )
                final_target = regime_target * strategy_ratio
            elif micro_regime == RegimeState.HIGH_VOL:
                strategy_name = "high_vol_guard"
                strategy_ratio = 1.0 if self.cfg.high_vol_cap > 0 else 0.0
                final_target = regime_target * strategy_ratio
            else:
                strategy_name = "neutral_hold"
                strategy_ratio = min(1.0, current_exposure / macro_target) if macro_target > 0 else 0.0
                final_target = regime_target * strategy_ratio

        # Directional daily trend inputs (cached with daily data).
        if "directional_inputs" in self._daily_cache:
            d = self._daily_cache["directional_inputs"]
        else:
            d = self._daily_directional_inputs(daily_closed)
            self._daily_cache["directional_inputs"] = d
        daily_adx = float(d["daily_adx"])
        plus_di = float(d["plus_di"])
        minus_di = float(d["minus_di"])
        sma200 = float(d["sma200"])
        sma50 = float(d["sma50"])
        sma50_slope = float(d["sma50_slope"])
        daily_close = float(d["daily_close"])

        if self.cfg.macro_mode == "stateful_gate":
            directional_checks = [
                macro_state == MacroState.ON_FULL,
                daily_adx >= float(self.cfg.trend_boost_adx_threshold),
                plus_di > minus_di,
                sma50_slope > 0,
            ]
            if self.cfg.trend_boost_require_above_sma200:
                directional_checks.append(daily_close > sma200)
            if self.cfg.trend_boost_require_micro_trend:
                directional_checks.append(micro_regime == RegimeState.TREND)
            boost_condition = all(directional_checks)
        else:
            boost_gate = self._daily_boost_gate(daily_closed, micro_regime)
            boost_condition = (
                macro_multiplier > 0
                and macro_score >= float(self.cfg.trend_boost_macro_score_threshold)
                and boost_gate
                and daily_adx >= float(self.cfg.trend_boost_adx_threshold)
            )

        booster_active = False
        boost_multiplier_applied = 1.0
        if self.cfg.trend_boost_enabled:
            booster_active = self._update_booster_state(daily_bar_ts, bool(boost_condition))
            if booster_active:
                boost_multiplier_applied = float(self.cfg.trend_boost_multiplier)
                final_target = final_target * boost_multiplier_applied

        final_target = max(0.0, min(self.cfg.max_position_fraction, float(final_target)))

        # Derived multiplier from micro + strategy stack
        pre_micro_target = max(1e-12, macro_target)
        micro_multiplier = float((regime_target * strategy_ratio) / pre_micro_target) if pre_micro_target > 0 else 0.0

        metadata.update(
            {
                "regime_multiplier": regime_mult,
                "strategy_ratio": strategy_ratio,
                "micro_multiplier": micro_multiplier,
                "regime_target": regime_target,
                "target_pre_boost": max(0.0, min(self.cfg.max_position_fraction, float(regime_target * strategy_ratio))),
                "daily_adx": daily_adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "sma200": sma200,
                "sma50": sma50,
                "sma50_slope": sma50_slope,
                "trend_boost_enabled": int(bool(self.cfg.trend_boost_enabled)),
                "trend_boost_condition": int(bool(boost_condition)),
                "trend_boost_active": int(bool(booster_active)),
                "boost_multiplier_applied": boost_multiplier_applied,
                "final_target": final_target,
            }
        )

        state_changed = (
            self._last_logged_day != _ts_day
            or self._last_logged_macro_state != macro_state.value
            or self._last_logged_micro_regime != micro_regime
            or self._last_logged_strategy_name != strategy_name
        )
        if state_changed:
            logger.info(
                "orchestrator_decision_event ts=%s day=%s macro_state=%s micro_regime=%s strategy=%s base_target=%.4f regime_target=%.4f final_target=%.4f trend_boost_active=%s",
                _ts_floor,
                _ts_day,
                macro_state.value,
                micro_regime.value,
                strategy_name,
                base_target,
                regime_target,
                final_target,
                bool(booster_active),
            )

        self._last_logged_day = _ts_day
        self._last_logged_macro_state = macro_state.value
        self._last_logged_micro_regime = micro_regime
        self._last_logged_strategy_name = strategy_name

        return RegimeDecisionBundle(
            macro_risk_on=True,
            macro_reason=macro_reason,
            micro_regime=micro_regime,
            micro_reason="computed",
            strategy_name=strategy_name,
            base_target=base_target,
            regime_multiplier=regime_mult,
            regime_target=regime_target,
            final_target=final_target,
            metadata=metadata,
        )

    def to_decision(self, bundle: RegimeDecisionBundle, timestamp: pd.Timestamp, current_exposure: float) -> StrategyDecision:
        return StrategyDecision(
            timestamp=timestamp,
            target_fraction=float(bundle.final_target),
            regime=bundle.micro_regime.value,
            sub_regime=bundle.strategy_name,
            metadata={
                **bundle.metadata,
                "macro_risk_on": bundle.macro_risk_on,
                "macro_reason": bundle.macro_reason,
                "micro_reason": bundle.micro_reason,
                "base_target": bundle.base_target,
                "regime_multiplier": bundle.regime_multiplier,
                "regime_target": bundle.regime_target,
                "prev_exposure": current_exposure,
            },
        )
