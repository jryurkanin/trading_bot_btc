from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


PolicyName = Literal["signal_change_only", "band", "always"]


@dataclass
class RebalanceState:
    last_trade_bucket: Optional[float] = None
    last_trade_ts: Optional[pd.Timestamp] = None
    trades_today: int = 0
    current_day: Optional[pd.Timestamp] = None


class RebalancePolicy:
    def __init__(
        self,
        policy: PolicyName = "signal_change_only",
        min_trade_notional_usd: float = 50.0,
        min_exposure_delta: float = 0.05,
        target_quantization_step: float = 0.25,
        min_time_between_trades_hours: float = 0.0,
        max_trades_per_day: int = 999,
    ) -> None:
        self.policy = policy
        self.min_trade_notional_usd = float(min_trade_notional_usd)
        self.min_exposure_delta = float(min_exposure_delta)
        self.target_quantization_step = float(target_quantization_step)
        self.min_time_between_trades_hours = float(min_time_between_trades_hours)
        self.max_trades_per_day = int(max_trades_per_day)
        self.state = RebalanceState()

    def quantize_target(self, target: float) -> float:
        t = min(1.0, max(0.0, float(target)))
        step = self.target_quantization_step
        if step <= 0:
            return t
        q = round(t / step) * step
        return min(1.0, max(0.0, q))

    def _roll_day(self, now: pd.Timestamp) -> None:
        day = pd.Timestamp(now).normalize()
        if self.state.current_day is None or day != self.state.current_day:
            self.state.current_day = day
            self.state.trades_today = 0

    def _within_cooldown(self, now: pd.Timestamp) -> bool:
        if self.state.last_trade_ts is None or self.min_time_between_trades_hours <= 0:
            return False
        elapsed = pd.Timestamp(now) - pd.Timestamp(self.state.last_trade_ts)
        return elapsed < pd.Timedelta(hours=self.min_time_between_trades_hours)

    def should_rebalance(self, target: float, current: float, equity_usd: float, now: pd.Timestamp) -> tuple[bool, float, str]:
        now_ts = pd.Timestamp(now)
        self._roll_day(now_ts)

        target_q = self.quantize_target(target)
        delta = target_q - float(current)
        abs_delta = abs(delta)
        notional = abs_delta * max(0.0, float(equity_usd))

        if self.state.trades_today >= self.max_trades_per_day:
            return False, target_q, "max_trades_per_day"
        if self._within_cooldown(now_ts):
            return False, target_q, "cooldown"
        if notional < self.min_trade_notional_usd:
            return False, target_q, "below_min_notional"

        if self.policy == "always":
            return abs_delta > 1e-12, target_q, "always"

        if self.policy == "band":
            if abs_delta < self.min_exposure_delta:
                return False, target_q, "inside_band"
            return True, target_q, "band_break"

        # signal_change_only
        if self.state.last_trade_bucket is not None and target_q == self.state.last_trade_bucket:
            return False, target_q, "no_signal_bucket_change"
        if abs_delta < self.min_exposure_delta:
            return False, target_q, "delta_too_small"
        return True, target_q, "signal_bucket_change"

    def on_trade(self, now: pd.Timestamp, traded_target_bucket: float) -> None:
        now_ts = pd.Timestamp(now)
        self._roll_day(now_ts)
        self.state.last_trade_bucket = float(traded_target_bucket)
        self.state.last_trade_ts = now_ts
        self.state.trades_today += 1
