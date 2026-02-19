from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import RiskConfig
from ..system_log import get_system_logger

logger = get_system_logger("execution.risk")


@dataclass
class RiskState:
    equity_peak: float = 1.0
    current_equity: float = 1.0
    last_data_ts: pd.Timestamp | None = None
    is_stale: bool = False

    day_anchor: pd.Timestamp | None = None
    day_start_equity: float = 1.0
    consecutive_losses: int = 0
    last_equity: float | None = None

    @property
    def drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return max(0.0, 1 - (self.current_equity / self.equity_peak))


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self._last_kill_reason: str | None = None
        self._last_stale_triggered: bool = False
        self._last_drawdown_capped: bool = False
        logger.debug(
            "risk_manager_init max_exposure=%s max_drawdown_cut_pct=%s max_additional_exposure_on_drawdown=%s stale_bar_max_multiplier=%s safe_mode=%s cutoff_no_new_entries=%s",
            float(cfg.max_exposure),
            float(cfg.max_drawdown_cut_pct),
            float(cfg.max_additional_exposure_on_drawdown),
            float(cfg.stale_bar_max_multiplier),
            bool(cfg.safe_mode),
            bool(cfg.cutoff_no_new_entries),
        )

    def stale_data_breaker(self, latest_bar_ts: pd.Timestamp, now: pd.Timestamp, timeframe_minutes: int = 60) -> bool:
        latest = pd.Timestamp(latest_bar_ts)
        cur = pd.Timestamp(now)
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")
        if cur.tzinfo is None:
            cur = cur.tz_localize("UTC")
        else:
            cur = cur.tz_convert("UTC")
        age = (cur - latest).total_seconds() / 60.0
        threshold = self.cfg.stale_bar_max_multiplier * timeframe_minutes
        stale = age > threshold
        if stale and not self._last_stale_triggered:
            logger.warning(
                "risk_stale_data_triggered latest_bar_ts=%s now=%s age_minutes=%.2f threshold_minutes=%.2f",
                latest,
                cur,
                age,
                threshold,
            )
        elif not stale and self._last_stale_triggered:
            logger.info(
                "risk_stale_data_cleared latest_bar_ts=%s now=%s age_minutes=%.2f threshold_minutes=%.2f",
                latest,
                cur,
                age,
                threshold,
            )
        self._last_stale_triggered = stale
        return stale

    def check_drawdown(self, state: RiskState) -> float:
        dd = state.drawdown
        capped = dd > self.cfg.max_drawdown_cut_pct
        if capped and not self._last_drawdown_capped:
            logger.warning(
                "risk_drawdown_cap_triggered drawdown=%.6f max_drawdown_cut_pct=%.6f factor=%.6f",
                dd,
                float(self.cfg.max_drawdown_cut_pct),
                float(self.cfg.max_additional_exposure_on_drawdown),
            )
        elif not capped and self._last_drawdown_capped:
            logger.info(
                "risk_drawdown_cap_cleared drawdown=%.6f max_drawdown_cut_pct=%.6f",
                dd,
                float(self.cfg.max_drawdown_cut_pct),
            )
        self._last_drawdown_capped = capped

        if not capped:
            return 1.0
        return self.cfg.max_additional_exposure_on_drawdown

    def update_runtime_state(self, state: RiskState, equity: float, now: pd.Timestamp) -> None:
        now_ts = pd.Timestamp(now)
        if now_ts.tzinfo is None:
            now_ts = now_ts.tz_localize("UTC")
        else:
            now_ts = now_ts.tz_convert("UTC")

        state.current_equity = float(equity)
        if equity > state.equity_peak:
            state.equity_peak = float(equity)

        if state.day_anchor is None or state.day_anchor.date() != now_ts.date():
            state.day_anchor = now_ts
            state.day_start_equity = float(equity)
            state.consecutive_losses = 0

        if state.last_equity is not None:
            if equity < state.last_equity:
                state.consecutive_losses += 1
            elif equity > state.last_equity:
                state.consecutive_losses = 0
        state.last_equity = float(equity)

    def daily_pnl_pct(self, state: RiskState) -> float:
        if state.day_start_equity <= 0:
            return 0.0
        return (state.current_equity / state.day_start_equity) - 1.0

    def kill_switch_reason(self, state: RiskState) -> str | None:
        reason: str | None = None

        if self.cfg.manual_kill_switch:
            reason = "manual_kill_switch"
        elif self.cfg.daily_loss_limit_pct is not None:
            if self.daily_pnl_pct(state) <= -abs(float(self.cfg.daily_loss_limit_pct)):
                reason = "daily_loss_limit"
        if reason is None and self.cfg.max_consecutive_losses is not None:
            if state.consecutive_losses >= int(self.cfg.max_consecutive_losses):
                reason = "max_consecutive_losses"

        if reason != self._last_kill_reason:
            if reason is not None:
                logger.warning(
                    "risk_kill_switch_triggered reason=%s daily_pnl_pct=%.6f consecutive_losses=%d",
                    reason,
                    self.daily_pnl_pct(state),
                    int(state.consecutive_losses),
                )
            elif self._last_kill_reason is not None:
                logger.info("risk_kill_switch_cleared previous_reason=%s", self._last_kill_reason)
            self._last_kill_reason = reason

        return reason

    def apply_caps(
        self,
        target_fraction: float,
        state: RiskState,
        latest_bar_ts: pd.Timestamp,
        now: pd.Timestamp,
        timeframe_minutes: int = 60,
        current_fraction: float = 0.0,
    ) -> float:
        original_target = float(target_fraction)
        if target_fraction < 0:
            logger.debug("risk_target_negative_clamped original_target=%.6f", original_target)
            return 0.0

        if self.stale_data_breaker(latest_bar_ts, now, timeframe_minutes=timeframe_minutes):
            if original_target > 0:
                logger.warning(
                    "risk_target_zeroed_stale_data original_target=%.6f latest_bar_ts=%s now=%s timeframe_minutes=%s",
                    original_target,
                    latest_bar_ts,
                    now,
                    timeframe_minutes,
                )
            return 0.0

        reason = self.kill_switch_reason(state)
        if reason:
            if self.cfg.safe_mode:
                if original_target > 0:
                    logger.warning(
                        "risk_target_zeroed_kill_switch reason=%s safe_mode=%s original_target=%.6f",
                        reason,
                        bool(self.cfg.safe_mode),
                        original_target,
                    )
                return 0.0
            if self.cfg.cutoff_no_new_entries:
                capped = min(target_fraction, max(0.0, current_fraction))
                if abs(capped - target_fraction) > 1e-12:
                    logger.info(
                        "risk_target_cutoff_no_new_entries reason=%s original_target=%.6f capped_target=%.6f current_fraction=%.6f",
                        reason,
                        target_fraction,
                        capped,
                        current_fraction,
                    )
                target_fraction = capped

        # apply drawdown breaker
        factor = self.check_drawdown(state)
        adjusted = target_fraction * factor
        if abs(adjusted - target_fraction) > 1e-12:
            logger.debug(
                "risk_target_drawdown_scaled input_target=%.6f factor=%.6f output_target=%.6f drawdown=%.6f",
                target_fraction,
                factor,
                adjusted,
                state.drawdown,
            )
        target_fraction = adjusted

        final_target = min(max(0.0, target_fraction), self.cfg.max_exposure)
        if abs(final_target - target_fraction) > 1e-12:
            logger.info(
                "risk_target_max_exposure_clamp input_target=%.6f max_exposure=%.6f output_target=%.6f",
                target_fraction,
                float(self.cfg.max_exposure),
                final_target,
            )
        return final_target

    def update_equity_peak(self, state: RiskState, equity: float) -> float:
        state.current_equity = equity
        if equity > state.equity_peak:
            state.equity_peak = equity
        return state.drawdown


class PositionSizer:
    def __init__(self, target_ann_vol: float, max_position_fraction: float = 1.0):
        self.target_ann_vol = target_ann_vol
        self.max_position_fraction = max_position_fraction

    def volatility_target_fraction(self, realized_ann_vol: float) -> float:
        if realized_ann_vol <= 0:
            return 0.0
        return max(0.0, min(self.max_position_fraction, self.target_ann_vol / realized_ann_vol))
