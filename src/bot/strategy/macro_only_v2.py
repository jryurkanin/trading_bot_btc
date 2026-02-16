from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import numpy as np

from ..config import RegimeConfig
from ..features.macro_score import MacroState
from ..features.regime import RegimeState
from ..features.macro_signals import macro_signal_strength, MacroStrength
from ..features.vol_sizing import realized_ann_vol_from_daily, sized_weight
from ..features.indicators import realized_vol
from .macro_gate_state import MacroGateV2
from .drawdown_breaker import DrawdownBreaker
from .regime_switching_orchestrator import RegimeDecisionBundle


class MacroOnlyV2Strategy:
    """Pure macro-only strategy with two-level state gate, vol sizing and breaker."""

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg
        self._gate = MacroGateV2(
            confirm_days=cfg.macro2_confirm_days,
            min_on_days=cfg.macro2_min_on_days,
            min_off_days=cfg.macro2_min_off_days,
        )
        self._breaker = DrawdownBreaker(
            enabled=cfg.macro2_dd_enabled,
            threshold=cfg.macro2_dd_threshold,
            cooldown_days=cfg.macro2_dd_cooldown_days,
            reentry_confirm_days=cfg.macro2_dd_reentry_confirm_days,
            safe_weight=cfg.macro2_dd_safe_weight,
        )

        self._last_refresh_day: pd.Timestamp | None = None
        self._frozen_base_fraction: float = 0.0
        self._current_target: float = 0.0
        self._last_hourly_idx: int = -1
        self._equity_proxy: float = 1.0
        self._equity_proxy_initialized: bool = False
        self._last_close: float = 0.0
        self._daily_ts_cache: dict[int, pd.DataFrame] = {}
        self._daily_signal_cache: dict[int, MacroState] = {}
        self._daily_realized_cache: dict[int, float] = {}
        self._daily_last_ts_cache: dict[int, pd.Timestamp | None] = {}
        self._acceleration_backend: str = "cpu"

    def reset(self) -> None:
        self._gate.reset()
        self._breaker.reset()
        self._last_refresh_day = None
        self._frozen_base_fraction = 0.0
        self._current_target = 0.0
        self._last_hourly_idx = -1
        self._equity_proxy = 1.0
        self._equity_proxy_initialized = False
        self._last_close = 0.0
        self._daily_ts_cache.clear()
        self._daily_signal_cache.clear()
        self._daily_realized_cache.clear()
        self._daily_last_ts_cache.clear()
        self._acceleration_backend = "cpu"

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
    def _day_cache_key(decision_ts: pd.Timestamp) -> int:
        ts = pd.Timestamp(decision_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.floor("D").value)

    def _closed_daily_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
        if daily_df is None or daily_df.empty:
            return pd.DataFrame(columns=daily_df.columns if daily_df is not None else [])

        key = self._day_cache_key(decision_ts)
        if key in self._daily_ts_cache:
            return self._daily_ts_cache[key]

        cutoff = pd.Timestamp(decision_ts)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        closed_cutoff = cutoff.floor("D")
        ts = self._to_timestamp_col(daily_df)
        pos = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), closed_cutoff.to_numpy(), side="left"))
        d = daily_df.iloc[:pos]
        self._daily_ts_cache[key] = d
        return d

    def _latest_daily_ts_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.Timestamp | None:
        key = self._day_cache_key(decision_ts)
        if key in self._daily_last_ts_cache:
            return self._daily_last_ts_cache[key]

        if daily_df is None or daily_df.empty:
            self._daily_last_ts_cache[key] = None
            return None

        d = self._closed_daily_cached(daily_df, decision_ts)
        if d.empty:
            self._daily_last_ts_cache[key] = None
            return None

        ts = self._to_timestamp_col(d)
        ts_last = ts.iloc[-1] if isinstance(ts, pd.Series) else ts[-1]
        last = pd.to_datetime(ts_last, utc=True)
        self._daily_last_ts_cache[key] = last
        return last

    def _daily_realized_cache_fn(self, daily_closed: pd.DataFrame, decision_ts: pd.Timestamp) -> float:
        key = self._day_cache_key(decision_ts)
        if key in self._daily_realized_cache:
            return self._daily_realized_cache[key]

        value = self._daily_realized_vol(daily_closed)
        self._daily_realized_cache[key] = value
        return value

    @staticmethod
    def _latest_daily_ts(daily_df: pd.DataFrame) -> pd.Timestamp | None:
        if daily_df is None or daily_df.empty:
            return None
        ts = MacroOnlyV2Strategy._to_timestamp_col(daily_df)
        if isinstance(ts, pd.Series):
            if ts.empty:
                return None
            ts_last = ts.iloc[-1]
        else:
            if len(ts) == 0:
                return None
            ts_last = ts[-1]
        return pd.to_datetime(ts_last, utc=True)

    @staticmethod
    def _macro_state_to_strength(state: MacroState) -> MacroStrength:
        if state == MacroState.ON_FULL:
            return MacroStrength.ON_FULL
        if state == MacroState.ON_HALF:
            return MacroStrength.ON_HALF
        return MacroStrength.OFF

    @staticmethod
    def _signal_weight_for_state(state: MacroState, cfg: RegimeConfig) -> float:
        if state == MacroState.ON_FULL:
            return float(cfg.macro2_weight_full)
        if state == MacroState.ON_HALF:
            return float(cfg.macro2_weight_half)
        return float(cfg.macro2_weight_off)

    def _update_equity_proxy(self, current_exposure: float, close: float) -> float:
        close_f = float(close)
        if not self._equity_proxy_initialized:
            self._equity_proxy = 1.0
            self._last_close = close_f
            self._equity_proxy_initialized = True
            return self._equity_proxy

        if self._last_close > 0 and close_f > 0:
            ret = (close_f / self._last_close) - 1.0
            self._equity_proxy *= (1.0 + max(-0.999999, min(1.0, float(current_exposure)) ) * ret)
        self._last_close = close_f
        return self._equity_proxy

    def _daily_realized_vol(self, daily_df: pd.DataFrame) -> float:
        # Prefer close-based daily realized volatility for no-lookahead use of closed bars.
        if daily_df is None or daily_df.empty:
            return 0.0

        if self.cfg.macro2_vol_lookback_days is not None and self.cfg.macro2_vol_lookback_days > 1:
            return realized_ann_vol_from_daily(daily_df, self.cfg.macro2_vol_lookback_days)

        # fallback: compute from raw close series without look-ahead
        if "close" not in daily_df.columns:
            return 0.0

        close = pd.to_numeric(daily_df["close"], errors="coerce").astype(float)
        if len(close) < 2:
            return 0.0

        daily_returns = close.pct_change()
        if daily_returns.empty or daily_returns.isna().all():
            return 0.0

        realized = realized_vol(
            daily_returns.fillna(0.0),
            window=max(2, len(daily_returns)),
            backend=self._acceleration_backend,
        )
        value = float(realized.iloc[-1]) if len(realized) else 0.0
        return max(0.0, value)

    def runtime_state(self) -> dict[str, Any]:
        gate_snapshot = self._gate.snapshot()
        breaker_snapshot = self._breaker.snapshot()
        return {
            "gate": gate_snapshot.__dict__,
            "breaker": breaker_snapshot.__dict__,
            "last_refresh_day": str(self._last_refresh_day) if self._last_refresh_day is not None else None,
            "frozen_base_fraction": float(self._frozen_base_fraction),
            "current_target": float(self._current_target),
            "last_hourly_idx": int(self._last_hourly_idx),
            "equity_proxy": float(self._equity_proxy),
            "equity_proxy_initialized": bool(self._equity_proxy_initialized),
            "last_close": float(self._last_close),
        }

    def load_runtime_state(self, payload: dict | None) -> None:
        if not isinstance(payload, dict):
            return

        if payload.get("gate"):
            self._gate.restore(payload.get("gate"))
        if payload.get("breaker"):
            self._breaker.restore(payload.get("breaker"))

        last_refresh = payload.get("last_refresh_day")
        if last_refresh:
            try:
                self._last_refresh_day = pd.Timestamp(last_refresh)
            except Exception:
                self._last_refresh_day = None

        frozen = payload.get("frozen_base_fraction")
        if isinstance(frozen, (int, float)):
            self._frozen_base_fraction = float(frozen)

        current = payload.get("current_target")
        if isinstance(current, (int, float)):
            self._current_target = float(current)

        idx = payload.get("last_hourly_idx")
        if isinstance(idx, int):
            self._last_hourly_idx = int(idx)

        proxy = payload.get("equity_proxy")
        if isinstance(proxy, (int, float)):
            self._equity_proxy = float(proxy)

        proxy_init = payload.get("equity_proxy_initialized")
        if isinstance(proxy_init, bool):
            self._equity_proxy_initialized = proxy_init

        last_close = payload.get("last_close")
        if isinstance(last_close, (int, float)):
            self._last_close = float(last_close)

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
                strategy_name="macro_only_v2",
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

        # Normalize timestamp to UTC and detect daily refresh boundaries.
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        ts_day = ts.floor("D")

        at_daily_refresh = self._last_refresh_day is None or ts_day != self._last_refresh_day

        # --- Daily-only macro inputs (look-back to closed daily candles only). ---
        daily_closed = self._closed_daily_cached(daily_df, ts)
        daily_bar_ts = self._latest_daily_ts_cached(daily_df, ts)

        # --- Macro strength + stateful gate ---
        daily_signal = self._daily_signal_cache.get(self._day_cache_key(ts))
        if daily_signal is None:
            daily_signal = macro_signal_strength(daily_closed, self.cfg)
            self._daily_signal_cache[self._day_cache_key(ts)] = daily_signal
        macro_state = self._gate.step(daily_signal, daily_bar_ts)

        # --- Vol-target sizing ---
        realized_ann_vol = self._daily_realized_cache.get(self._day_cache_key(ts))
        if realized_ann_vol is None:
            realized_ann_vol = self._daily_realized_cache_fn(daily_closed, ts)
        base_target = sized_weight(
            state=macro_state,
            realized_vol=float(realized_ann_vol),
            mode=str(getattr(self.cfg, "macro2_vol_mode", "inverse_vol")),
            cfg=self.cfg,
        )

        # update daily base only on refresh
        if at_daily_refresh:
            self._frozen_base_fraction = base_target
            self._last_refresh_day = ts_day

        base_fraction = self._frozen_base_fraction

        close = float(hourly_df["close"].iloc[hourly_idx])
        self._update_equity_proxy(current_exposure=current_exposure, close=close)

        target_pre_breaker = float(base_fraction)
        target_after_breaker = target_pre_breaker
        if self.cfg.macro2_dd_enabled:
            target_after_breaker = self._breaker.step(
                equity=self._equity_proxy,
                daily_ts=ts_day if at_daily_refresh else None,
                macro_state=macro_state,
                raw_target=target_pre_breaker,
            )

        # Intraday hold: do not increase target within a day; keep last observed value.
        intraday_suppressed = False
        desired_target = target_after_breaker
        if not at_daily_refresh and desired_target > self._current_target:
            desired_target = self._current_target
            intraday_suppressed = True

        final_target = max(0.0, min(self.cfg.max_position_fraction, desired_target))
        self._current_target = final_target
        self._last_hourly_idx = hourly_idx

        regime_mult = 1.0
        macro_reason = f"macro_only_v2_{macro_state.value.lower()}"
        macro_mult = float({
            MacroState.OFF: self.cfg.macro2_weight_off,
            MacroState.ON_HALF: self.cfg.macro2_weight_half,
            MacroState.ON_FULL: self.cfg.macro2_weight_full,
        }.get(macro_state, 0.0))

        metadata: Dict[str, float | str | int] = {
            "signal_mode": str(getattr(self.cfg, "macro2_signal_mode", "")),
            "macro_signal": daily_signal.value,
            "macro_state": macro_state.value,
            "macro_state_weight": macro_mult,
            "macro_reason": macro_reason,
            "macro2_signal_rank": int({
                MacroStrength.OFF: 0,
                MacroStrength.ON_HALF: 1,
                MacroStrength.ON_FULL: 2,
            }.get(daily_signal, 0)),
            "macro2_weight_off": float(self.cfg.macro2_weight_off),
            "macro2_weight_half": float(self.cfg.macro2_weight_half),
            "macro2_weight_full": float(self.cfg.macro2_weight_full),
            "macro2_target_ann_vol_half": float(self.cfg.macro2_target_ann_vol_half),
            "macro2_target_ann_vol_full": float(self.cfg.macro2_target_ann_vol_full),
            "macro2_target_ann_vol": float(realized_ann_vol),
            "macro2_vol_mode": str(self.cfg.macro2_vol_mode),
            "macro2_vol_lookback_days": int(self.cfg.macro2_vol_lookback_days),
            "macro2_vol_floor": float(self.cfg.macro2_vol_floor),
            "realized_ann_vol": float(realized_ann_vol),
            "base_fraction": float(base_fraction),
            "target_pre_breaker": float(target_pre_breaker),
            "target_after_breaker": float(target_after_breaker),
            "macro_refresh": int(at_daily_refresh),
            "macro2_dd_enabled": int(int(self.cfg.macro2_dd_enabled)),
            "macro2_dd_active": int(self._breaker.active),
            "macro2_dd_threshold": float(self.cfg.macro2_dd_threshold),
            "macro2_dd_cooldown_days": int(self.cfg.macro2_dd_cooldown_days),
            "macro2_dd_reentry_confirm_days": int(self.cfg.macro2_dd_reentry_confirm_days),
            "macro2_dd_safe_weight": float(self.cfg.macro2_dd_safe_weight),
            "breaker_drawdown": float(self._breaker._drawdown),
            "equity_proxy": float(self._equity_proxy),
            "daily_bar_ts": str(daily_bar_ts) if daily_bar_ts is not None else None,
            "current_position_fraction": float(current_exposure),
            "intraday_increase_suppressed": int(intraday_suppressed),
            "macro_target": final_target,
            "base_target": float(base_fraction),
            "acceleration_backend": str(self._acceleration_backend),
            # compat fields used in reporting
            "trend_boost_active": 0,
            "boost_multiplier_applied": 1.0,
        }

        return RegimeDecisionBundle(
            macro_risk_on=bool(final_target > 0.0),
            macro_reason=macro_reason,
            micro_regime=RegimeState.NEUTRAL,
            micro_reason="macro_only_v2",
            strategy_name="macro_only_v2",
            base_target=base_fraction,
            regime_multiplier=regime_mult,
            regime_target=base_fraction,
            final_target=final_target,
            metadata=metadata,
        )
