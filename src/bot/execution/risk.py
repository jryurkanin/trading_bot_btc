from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import RiskConfig


@dataclass
class RiskState:
    equity_peak: float = 1.0
    current_equity: float = 1.0
    last_data_ts: pd.Timestamp | None = None
    is_stale: bool = False

    @property
    def drawdown(self) -> float:
        if self.equity_peak <= 0:
            return 0.0
        return max(0.0, 1 - (self.current_equity / self.equity_peak))


class RiskManager:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

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
        return age > (self.cfg.stale_bar_max_multiplier * timeframe_minutes)

    def check_drawdown(self, state: RiskState) -> float:
        dd = state.drawdown
        if dd <= self.cfg.max_drawdown_cut_pct:
            return 1.0
        return self.cfg.max_additional_exposure_on_drawdown

    def apply_caps(self, target_fraction: float, state: RiskState, latest_bar_ts: pd.Timestamp, now: pd.Timestamp, timeframe_minutes: int = 60) -> float:
        if target_fraction < 0:
            return 0.0
        if self.stale_data_breaker(latest_bar_ts, now, timeframe_minutes=timeframe_minutes):
            return 0.0

        # apply drawdown breaker
        factor = self.check_drawdown(state)
        target_fraction *= factor

        return min(max(0.0, target_fraction), self.cfg.max_exposure)

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
