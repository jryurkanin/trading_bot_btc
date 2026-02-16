from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..features.macro_signals import MacroStrength
from ..features.macro_score import MacroState


@dataclass(frozen=True)
class MacroGateV2Snapshot:
    state: str
    state_age_days: int
    last_daily_ts: str | None
    on_half_streak: int
    on_full_streak: int
    off_streak: int
    deescalate_streak: int


class MacroGateV2:
    """Stateful macro gate wrapper for MacroStrength signals.

    The gate is fed a *discrete* signal each day (OFF/HALF/FULL) and applies
    stateful hysteresis through confirm/min-age constraints.
    """

    def __init__(
        self,
        *,
        confirm_days: int = 2,
        min_on_days: int = 2,
        min_off_days: int = 1,
    ) -> None:
        self.confirm_days = max(1, int(confirm_days))
        self.min_on_days = max(1, int(min_on_days))
        self.min_off_days = max(1, int(min_off_days))

        self.state: MacroState = MacroState.OFF
        self.state_age_days: int = 0
        self.last_daily_ts: pd.Timestamp | None = None

        self._on_half_streak: int = 0
        self._on_full_streak: int = 0
        self._off_streak: int = 0
        self._deescalate_streak: int = 0

    def reset(self) -> None:
        self.state = MacroState.OFF
        self.state_age_days = 0
        self.last_daily_ts = None
        self._on_half_streak = 0
        self._on_full_streak = 0
        self._off_streak = 0
        self._deescalate_streak = 0

    def snapshot(self) -> MacroGateV2Snapshot:
        return MacroGateV2Snapshot(
            state=self.state.value,
            state_age_days=int(self.state_age_days),
            last_daily_ts=self.last_daily_ts.isoformat() if self.last_daily_ts is not None else None,
            on_half_streak=int(self._on_half_streak),
            on_full_streak=int(self._on_full_streak),
            off_streak=int(self._off_streak),
            deescalate_streak=int(self._deescalate_streak),
        )

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return

        raw_state = payload.get("state", self.state.value)
        try:
            self.state = MacroState(str(raw_state))
        except Exception:
            self.state = MacroState.OFF

        self.state_age_days = int(payload.get("state_age_days", 0) or 0)
        ts_raw = payload.get("last_daily_ts")
        self.last_daily_ts = pd.Timestamp(ts_raw) if ts_raw else None

        self._on_half_streak = int(payload.get("on_half_streak", 0) or 0)
        self._on_full_streak = int(payload.get("on_full_streak", 0) or 0)
        self._off_streak = int(payload.get("off_streak", 0) or 0)
        self._deescalate_streak = int(payload.get("deescalate_streak", 0) or 0)

    def _same_bar(self, daily_ts: pd.Timestamp | None) -> bool:
        if daily_ts is None:
            return True
        if self.last_daily_ts is None:
            return False
        return pd.Timestamp(daily_ts) == pd.Timestamp(self.last_daily_ts)

    def _set_state(self, next_state: MacroState) -> None:
        if self.state == next_state:
            return
        self.state = next_state
        self.state_age_days = 0

    @staticmethod
    def _strength_rank(value: MacroStrength) -> int:
        if value == MacroStrength.OFF:
            return 0
        if value == MacroStrength.ON_HALF:
            return 1
        return 2

    def _update_streaks(self, signal: MacroStrength) -> None:
        self._on_full_streak = self._on_full_streak + 1 if signal == MacroStrength.ON_FULL else 0
        self._on_half_streak = self._on_half_streak + 1 if signal in {MacroStrength.ON_HALF, MacroStrength.ON_FULL} else 0
        self._off_streak = self._off_streak + 1 if signal == MacroStrength.OFF else 0

        if signal == MacroStrength.ON_FULL:
            self._deescalate_streak = 0
        elif signal in {MacroStrength.ON_HALF, MacroStrength.OFF}:
            self._deescalate_streak += 1
        else:
            self._deescalate_streak = 0

    def _state_allows_reentry(self) -> bool:
        return self.state_age_days >= self.min_off_days

    def _state_allows_exit(self) -> bool:
        return self.state_age_days >= self.min_on_days

    def step(self, signal: MacroStrength, daily_ts: pd.Timestamp | None) -> MacroState:
        if daily_ts is None:
            return self.state

        if self._same_bar(daily_ts):
            return self.state

        self.last_daily_ts = pd.Timestamp(daily_ts)
        self.state_age_days += 1

        self._update_streaks(signal)

        rank = self._strength_rank(signal)

        if self.state == MacroState.OFF:
            if self._state_allows_reentry():
                if self._on_full_streak >= self.confirm_days:
                    self._set_state(MacroState.ON_FULL)
                elif self._on_half_streak >= self.confirm_days:
                    self._set_state(MacroState.ON_HALF)
            return self.state

        if self.state == MacroState.ON_HALF:
            if self._on_full_streak >= self.confirm_days:
                self._set_state(MacroState.ON_FULL)
            elif self._state_allows_exit() and self._off_streak >= self.confirm_days:
                self._set_state(MacroState.OFF)
            return self.state

        # ON_FULL
        if self._state_allows_exit() and self._off_streak >= self.confirm_days:
            self._set_state(MacroState.OFF)
        elif self._deescalate_streak >= self.confirm_days and rank <= 1:
            self._set_state(MacroState.ON_HALF)

        return self.state
