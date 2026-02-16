from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..features.macro_score import MacroState


def _on_state(state: MacroState | str) -> bool:
    if isinstance(state, MacroState):
        return state in {MacroState.ON_HALF, MacroState.ON_FULL}
    return str(state) in {MacroState.ON_HALF.value, MacroState.ON_FULL.value}


@dataclass(frozen=True)
class DrawdownBreakerSnapshot:
    active: bool
    drawdown: float
    peak_equity: float
    equity: float
    last_daily_ts: str | None
    cooldown_days: int
    reentry_confirm_days: int
    enabled: bool


class DrawdownBreaker:
    """Stateful drawdown breaker with cooldown + re-entry confirmation.

    The breaker watches a synthetic/strategy-side equity series:
    - when drawdown <= -threshold => breaker activates
    - during activation output target is clamped to dd_safe_weight
    - re-entry requires:
        * cooldown_days elapsed
        * macro regime in ON_* state for reentry_confirm_days consecutive days
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold: float = 0.25,
        cooldown_days: int = 10,
        reentry_confirm_days: int = 2,
        safe_weight: float = 0.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold = float(threshold)
        self.cooldown_days = max(1, int(cooldown_days))
        self.reentry_confirm_days = max(1, int(reentry_confirm_days))
        self.safe_weight = max(0.0, min(1.0, float(safe_weight)))

        self.active: bool = False
        self._equity: float = 1.0
        self._peak_equity: float = 1.0
        self.last_daily_ts: Any = None
        self._cooldown_count: int = 0
        self._reentry_streak: int = 0
        self._drawdown: float = 0.0

    def reset(self) -> None:
        self.active = False
        self._equity = 1.0
        self._peak_equity = 1.0
        self.last_daily_ts = None
        self._cooldown_count = 0
        self._reentry_streak = 0
        self._drawdown = 0.0

    def snapshot(self) -> DrawdownBreakerSnapshot:
        return DrawdownBreakerSnapshot(
            active=bool(self.active),
            drawdown=float(self._drawdown),
            peak_equity=float(self._peak_equity),
            equity=float(self._equity),
            last_daily_ts=self.last_daily_ts.isoformat() if self.last_daily_ts is not None else None,
            cooldown_days=int(self._cooldown_count),
            reentry_confirm_days=int(self._reentry_streak),
            enabled=bool(self.enabled),
        )

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return

        self.active = bool(payload.get("active", self.active))
        self._equity = float(payload.get("equity", self._equity))
        self._peak_equity = float(payload.get("peak_equity", self._peak_equity))
        self._drawdown = float(payload.get("drawdown", self._drawdown))
        self._cooldown_count = int(payload.get("cooldown_days", self._cooldown_count) or 0)
        self._reentry_streak = int(payload.get("reentry_confirm_days", self._reentry_streak) or 0)
        self.enabled = bool(payload.get("enabled", self.enabled))
        ts_raw = payload.get("last_daily_ts")
        self.last_daily_ts = pd.Timestamp(ts_raw) if ts_raw else None

    def update_equity(self, equity: float, daily_ts: object | None) -> None:
        if not self.enabled:
            return

        self._equity = max(1e-12, float(equity))
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        if self._peak_equity <= 0:
            self._drawdown = 0.0
        else:
            self._drawdown = (self._equity - self._peak_equity) / self._peak_equity

    def step(
        self,
        equity: float,
        daily_ts: object | None,
        macro_state: MacroState,
        raw_target: float,
    ) -> float:
        if not self.enabled:
            return raw_target

        if daily_ts is None:
            return raw_target if not self.active else min(raw_target, self.safe_weight)

        ts = pd.Timestamp(daily_ts)
        if self.last_daily_ts is not None and ts == pd.Timestamp(self.last_daily_ts):
            return raw_target if not self.active else min(raw_target, self.safe_weight)

        # new bar -> update state
        self.last_daily_ts = ts
        self.update_equity(equity, ts)

        if not self.active:
            if self._drawdown <= -abs(self.threshold):
                self.active = True
                self._cooldown_count = 0
                self._reentry_streak = 0
            return min(raw_target, self.safe_weight) if self.active else raw_target

        # Active breaker path
        if _on_state(macro_state):
            self._reentry_streak += 1
        else:
            self._reentry_streak = 0

        self._cooldown_count += 1

        if self._cooldown_count >= self.cooldown_days and self._reentry_streak >= self.reentry_confirm_days:
            self.active = False
            self._cooldown_count = 0
            self._reentry_streak = 0
            return raw_target

        return min(raw_target, self.safe_weight)
