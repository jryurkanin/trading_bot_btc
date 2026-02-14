from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from ...features.indicators import bollinger_bands


@dataclass
class RangeState:
    last_signal_ts: Optional[pd.Timestamp] = None
    trades_today: int = 0
    current_day: Optional[pd.Timestamp] = None


@dataclass
class RangeStrategyConfig:
    bb_window: int = 20
    bb_stdev: float = 2.0
    tranche_size: float = 0.25
    max_exposure: float = 0.75
    min_time_between_trades_hours: float = 2.0
    max_trades_per_day: int = 4


class MeanReversionBBStrategy:
    name = "mean_reversion_bb"

    def __init__(self, cfg: Optional[RangeStrategyConfig] = None):
        self.cfg = cfg or RangeStrategyConfig()
        self.state = RangeState()

    def reset(self):
        self.state = RangeState()

    def allowed_trade(self, now: pd.Timestamp) -> bool:
        if self.state.last_signal_ts is not None:
            if now - self.state.last_signal_ts < pd.Timedelta(hours=self.cfg.min_time_between_trades_hours):
                return False

        day = pd.Timestamp(now.date())
        if self.state.current_day is None or day.date() != self.state.current_day.date():
            self.state.current_day = day
            self.state.trades_today = 0

        if self.state.trades_today >= self.cfg.max_trades_per_day:
            return False
        return True

    def on_trade(self, now: pd.Timestamp):
        self.state.last_signal_ts = now
        self.state.trades_today += 1

    def compute_target(self, hourly_row: pd.Series, prev_row: Optional[pd.Series], current_exposure: float, now: pd.Timestamp, close_series: pd.Series) -> float:
        # compute bands
        mid, up, low = bollinger_bands(close_series, window=self.cfg.bb_window, stdev=self.cfg.bb_stdev)
        mid = float(mid.iloc[-1])
        up = float(up.iloc[-1])
        low = float(low.iloc[-1])
        close = float(hourly_row["close"])

        target = current_exposure

        crossed_below = prev_row is not None and float(prev_row["close"]) > float(low) and close < low
        crossed_above_mid = prev_row is not None and float(prev_row["close"]) < mid and close > mid
        crossed_above_up = close > up

        if crossed_below and self.allowed_trade(now):
            target = min(current_exposure + self.cfg.tranche_size, self.cfg.max_exposure)
            self.on_trade(now)
        elif crossed_above_up:
            target = 0.0
            self.on_trade(now)
        elif crossed_above_mid and current_exposure > 0:
            target = max(0.0, current_exposure - self.cfg.tranche_size)
            self.on_trade(now)

        return target

    def signal_reason(
        self,
        hourly_row: pd.Series,
        prev_row: Optional[pd.Series],
        current_exposure: float,
        now: pd.Timestamp,
        close_series: pd.Series,
    ) -> Dict[str, float]:
        mid, up, lo = bollinger_bands(close_series, window=self.cfg.bb_window, stdev=self.cfg.bb_stdev)
        return {
            "close": float(hourly_row["close"]),
            "bb_mid": float(mid.iloc[-1]),
            "bb_up": float(up.iloc[-1]),
            "bb_low": float(lo.iloc[-1]),
            "current_exposure": float(current_exposure),
        }
