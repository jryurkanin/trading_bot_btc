from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ...features.indicators import donchian_channel, ema, atr


@dataclass
class TrendState:
    entry_price: Optional[float] = None


@dataclass
class TrendStrategyConfig:
    mode: str = "donchian"  # donchian | ema_cross
    donchian_window: int = 55
    ema_fast: int = 20
    ema_slow: int = 50
    atr_window: int = 14
    atr_mult: float = 3.0
    trend_exposure_cap: float = 1.0
    vol_target_multiplier: float = 1.0


class TrendFollowingBreakoutStrategy:
    name = "trend_following_breakout"

    def __init__(self, cfg: Optional[TrendStrategyConfig] = None):
        self.cfg = cfg or TrendStrategyConfig()
        self.state = TrendState()

    def reset(self):
        self.state = TrendState()

    def compute_target(self, hourly: pd.DataFrame, current_exposure: float, now: pd.Timestamp,
                       idx: int | None = None, precomputed: dict | None = None) -> float:
        if hourly.empty or len(hourly) < 2:
            return current_exposure

        # If caller provides index + precomputed series, use direct lookups
        # instead of recomputing rolling indicators on a growing slice.
        if idx is not None and precomputed:
            return self._compute_target_indexed(hourly, current_exposure, idx, precomputed)

        close = hourly["close"]
        high = hourly["high"]
        low = hourly["low"]

        target = current_exposure

        if self.cfg.mode == "ema_cross":
            fast = ema(close, self.cfg.ema_fast)
            slow = ema(close, self.cfg.ema_slow)
            fast_v, slow_v = float(fast.iloc[-1]), float(slow.iloc[-1])
            if len(fast) > 1:
                prev_fast, prev_slow = float(fast.iloc[-2]), float(slow.iloc[-2])
            else:
                prev_fast, prev_slow = fast_v, slow_v
            crossed_up = prev_fast <= prev_slow and fast_v > slow_v
            crossed_down = prev_fast >= prev_slow and fast_v < slow_v
        else:
            ch_low, ch_high = donchian_channel(high, low, self.cfg.donchian_window)
            ch_high_v = float(ch_high.iloc[-2]) if len(ch_high) > 1 else float(ch_high.iloc[-1])
            crossed_up = float(close.iloc[-2]) <= ch_high_v and float(close.iloc[-1]) > ch_high_v
            crossed_down = False

        atr_v = float(atr(high, low, close, self.cfg.atr_window).iloc[-1])
        close_v = float(close.iloc[-1])

        if self.state.entry_price is not None:
            stop = self.state.entry_price - self.cfg.atr_mult * atr_v
            if close_v < stop:
                target = 0.0
                self.state.entry_price = None
                return target

        if self.cfg.mode == "ema_cross" and crossed_down:
            target = 0.0
            self.state.entry_price = None
            return target

        if self.cfg.mode == "ema_cross" and not crossed_up and current_exposure > 0:
            # continue
            return current_exposure

        if crossed_up:
            target = self.cfg.trend_exposure_cap * self.cfg.vol_target_multiplier
            self.state.entry_price = close_v

        if current_exposure <= 0 and target == current_exposure:
            # no signal, no change
            return current_exposure
        if target > current_exposure:
            self.state.entry_price = close_v
        return target

    def _compute_target_indexed(self, hourly: pd.DataFrame, current_exposure: float,
                                idx: int, precomputed: dict) -> float:
        """Fast path: use precomputed indicator series with index-based lookups."""
        close = hourly["close"]
        target = current_exposure
        close_v = float(close.iloc[idx])
        close_prev = float(close.iloc[idx - 1]) if idx >= 1 else close_v

        if self.cfg.mode == "ema_cross":
            fast_s = precomputed.get("ema_fast")
            slow_s = precomputed.get("ema_slow")
            fast_v = float(fast_s.iloc[idx])
            slow_v = float(slow_s.iloc[idx])
            prev_fast = float(fast_s.iloc[idx - 1]) if idx >= 1 else fast_v
            prev_slow = float(slow_s.iloc[idx - 1]) if idx >= 1 else slow_v
            crossed_up = prev_fast <= prev_slow and fast_v > slow_v
            crossed_down = prev_fast >= prev_slow and fast_v < slow_v
        else:
            ch_high = precomputed.get("donchian_high")
            ch_high_v = float(ch_high.iloc[idx - 1]) if idx >= 1 else float(ch_high.iloc[idx])
            crossed_up = close_prev <= ch_high_v and close_v > ch_high_v
            crossed_down = False

        atr_s = precomputed.get("atr")
        atr_v = float(atr_s.iloc[idx])

        if self.state.entry_price is not None:
            stop = self.state.entry_price - self.cfg.atr_mult * atr_v
            if close_v < stop:
                self.state.entry_price = None
                return 0.0

        if self.cfg.mode == "ema_cross" and crossed_down:
            self.state.entry_price = None
            return 0.0

        if self.cfg.mode == "ema_cross" and not crossed_up and current_exposure > 0:
            return current_exposure

        if crossed_up:
            target = self.cfg.trend_exposure_cap * self.cfg.vol_target_multiplier
            self.state.entry_price = close_v

        if current_exposure <= 0 and target == current_exposure:
            return current_exposure
        if target > current_exposure:
            self.state.entry_price = close_v
        return target

    def signal_reason(self, hourly: pd.DataFrame, current_exposure: float, now: pd.Timestamp) -> dict:
        if hourly.empty:
            return {"close": 0.0}
        close_v = float(hourly["close"].iloc[-1])
        low_v = float(hourly["low"].iloc[-1])
        high_v = float(hourly["high"].iloc[-1])
        return {
            "close": close_v,
            "high": high_v,
            "low": low_v,
            "mode": self.cfg.mode,
            "entry_price": self.state.entry_price,
            "current_exposure": current_exposure,
        }
