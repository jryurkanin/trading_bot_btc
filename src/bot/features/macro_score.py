from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from . import indicators


DEFAULT_COMPONENTS = [
    "close_gt_sma50",
    "close_gt_sma200",
    "ret_28d_pos",
    "ret_90d_pos",
]


class MacroState(str, Enum):
    OFF = "OFF"
    ON_HALF = "ON_HALF"
    ON_FULL = "ON_FULL"


@dataclass
class MacroScoreResult:
    score: float
    components: dict[str, float]
    multiplier: float
    enabled_components: list[str]


@dataclass
class MacroGateSnapshot:
    state: str
    state_age_days: int
    last_daily_ts: str | None
    enter_half_streak: int
    enter_full_streak: int
    exit_streak: int
    deescalate_streak: int


class MacroGateStateMachine:
    """Daily macro gate with hysteresis and min on/off durations."""

    def __init__(
        self,
        *,
        enter_threshold: float = 0.75,
        exit_threshold: float = 0.25,
        full_threshold: float = 1.0,
        half_threshold: float = 0.75,
        confirm_days: int = 2,
        min_on_days: int = 2,
        min_off_days: int = 1,
    ) -> None:
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.full_threshold = float(full_threshold)
        self.half_threshold = float(half_threshold)
        self.confirm_days = max(1, int(confirm_days))
        self.min_on_days = max(1, int(min_on_days))
        self.min_off_days = max(1, int(min_off_days))

        self.state: MacroState = MacroState.OFF
        self.state_age_days: int = 0
        self.last_daily_ts: pd.Timestamp | None = None

        self._enter_half_streak: int = 0
        self._enter_full_streak: int = 0
        self._exit_streak: int = 0
        self._deescalate_streak: int = 0

    def reset(self) -> None:
        self.state = MacroState.OFF
        self.state_age_days = 0
        self.last_daily_ts = None
        self._enter_half_streak = 0
        self._enter_full_streak = 0
        self._exit_streak = 0
        self._deescalate_streak = 0

    def snapshot(self) -> MacroGateSnapshot:
        return MacroGateSnapshot(
            state=self.state.value,
            state_age_days=int(self.state_age_days),
            last_daily_ts=self.last_daily_ts.isoformat() if self.last_daily_ts is not None else None,
            enter_half_streak=int(self._enter_half_streak),
            enter_full_streak=int(self._enter_full_streak),
            exit_streak=int(self._exit_streak),
            deescalate_streak=int(self._deescalate_streak),
        )

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        try:
            self.state = MacroState(str(payload.get("state", self.state.value)))
        except Exception:
            self.state = MacroState.OFF
        self.state_age_days = int(payload.get("state_age_days", self.state_age_days) or 0)
        ts_raw = payload.get("last_daily_ts")
        self.last_daily_ts = pd.Timestamp(ts_raw) if ts_raw else None
        self._enter_half_streak = int(payload.get("enter_half_streak", 0) or 0)
        self._enter_full_streak = int(payload.get("enter_full_streak", 0) or 0)
        self._exit_streak = int(payload.get("exit_streak", 0) or 0)
        self._deescalate_streak = int(payload.get("deescalate_streak", 0) or 0)

    def _same_bar(self, daily_ts: pd.Timestamp | None) -> bool:
        if daily_ts is None:
            return True
        if self.last_daily_ts is None:
            return False
        return pd.Timestamp(daily_ts) == pd.Timestamp(self.last_daily_ts)

    def _set_state(self, state: MacroState) -> None:
        self.state = state
        self.state_age_days = 0

    def _update_streaks(self, score: float) -> None:
        self._enter_half_streak = self._enter_half_streak + 1 if score >= self.enter_threshold else 0
        self._enter_full_streak = self._enter_full_streak + 1 if score >= self.full_threshold else 0
        self._exit_streak = self._exit_streak + 1 if score <= self.exit_threshold else 0
        self._deescalate_streak = self._deescalate_streak + 1 if score <= self.half_threshold else 0

    def step(self, score: float, daily_ts: pd.Timestamp | None) -> MacroState:
        """Advance state once per new closed daily bar."""
        score = float(max(0.0, min(1.0, score)))
        if daily_ts is None:
            return self.state
        if self._same_bar(daily_ts):
            return self.state

        self.last_daily_ts = pd.Timestamp(daily_ts)
        self.state_age_days += 1
        self._update_streaks(score)

        if self.state == MacroState.OFF:
            if self.state_age_days >= self.min_off_days:
                if self._enter_full_streak >= self.confirm_days:
                    self._set_state(MacroState.ON_FULL)
                elif self._enter_half_streak >= self.confirm_days:
                    self._set_state(MacroState.ON_HALF)
            return self.state

        if self.state == MacroState.ON_HALF:
            if self._enter_full_streak >= self.confirm_days:
                self._set_state(MacroState.ON_FULL)
                return self.state
            if self.state_age_days >= self.min_on_days and self._exit_streak >= self.confirm_days:
                self._set_state(MacroState.OFF)
            return self.state

        # ON_FULL
        if self.state_age_days >= self.min_on_days and self._exit_streak >= self.confirm_days:
            self._set_state(MacroState.OFF)
        elif self._deescalate_streak >= self.confirm_days:
            self._set_state(MacroState.ON_HALF)
        return self.state

    @staticmethod
    def multiplier(state: MacroState, half_multiplier: float = 0.5, full_multiplier: float = 1.0) -> float:
        if state == MacroState.OFF:
            return 0.0
        if state == MacroState.ON_HALF:
            return float(max(0.0, min(1.0, half_multiplier)))
        return float(max(0.0, min(1.0, full_multiplier)))


def _safe_component(condition: bool | float | int | None) -> float:
    if condition is None:
        return 0.0
    try:
        return 1.0 if bool(condition) else 0.0
    except Exception:
        return 0.0


def _last_close(daily_df: pd.DataFrame) -> float:
    if daily_df.empty:
        return 0.0
    try:
        return float(daily_df["close"].iloc[-1])
    except Exception:
        return 0.0


def _ret_pos(close: pd.Series, lookback_days: int) -> float:
    if close.empty:
        return 0.0
    base = close.shift(lookback_days).iloc[-1]
    if pd.isna(base) or float(base) == 0.0:
        return 0.0
    ret = float(close.iloc[-1] / float(base) - 1.0)
    return _safe_component(ret > 0)


def compute_macro_score(daily_df: pd.DataFrame, enabled_components: list[str] | None = None) -> tuple[float, dict[str, float], list[str]]:
    if daily_df is None or daily_df.empty:
        comps = {k: 0.0 for k in DEFAULT_COMPONENTS}
        return 0.0, comps, []

    close = daily_df["close"].astype(float)
    enabled = list(enabled_components) if enabled_components else list(DEFAULT_COMPONENTS)
    enabled_set = set(enabled)

    last = _last_close(daily_df)
    sma50 = indicators.sma(close, 50).iloc[-1] if len(close) else float("nan")
    sma200 = indicators.sma(close, 200).iloc[-1] if len(close) else float("nan")

    components = {
        "close_gt_sma50": _safe_component(pd.notna(sma50) and last > float(sma50)),
        "close_gt_sma200": _safe_component(pd.notna(sma200) and last > float(sma200)),
        "ret_28d_pos": _ret_pos(close, 28),
        "ret_90d_pos": _ret_pos(close, 90),
    }

    used = [k for k in components.keys() if k in enabled_set]
    if not used:
        used = list(DEFAULT_COMPONENTS)

    score = float(sum(components[k] for k in used) / max(1, len(used)))
    return score, components, used


def macro_multiplier_from_score(
    score: float,
    *,
    transform: str = "linear",
    floor: float = 0.0,
    min_to_trade: float = 0.25,
    piecewise_levels: list[float] | None = None,
) -> float:
    score = float(max(0.0, min(1.0, score)))
    floor = float(max(0.0, min(1.0, floor)))
    min_to_trade = float(max(0.0, min(1.0, min_to_trade)))

    if score < min_to_trade:
        return 0.0

    if transform == "piecewise":
        levels = [float(x) for x in (piecewise_levels or [0.0, 0.33, 0.66, 1.0])]
        levels = sorted(max(0.0, min(1.0, x)) for x in levels)
        if len(levels) == 1:
            return levels[0]

        # normalize score by floor, then map to equal-width bins.
        denom = max(1e-12, 1.0 - floor)
        normalized = max(0.0, min(1.0, (score - floor) / denom))
        bins = len(levels) - 1
        idx = min(int(normalized * bins + 1e-12), bins)
        return float(levels[idx])

    denom = max(1e-12, 1.0 - floor)
    linear = (score - floor) / denom
    return float(max(0.0, min(1.0, linear)))


def macro_result(daily_df: pd.DataFrame, cfg: Any) -> MacroScoreResult:
    score, components, used = compute_macro_score(daily_df, getattr(cfg, "macro_score_components", None))
    mult = macro_multiplier_from_score(
        score,
        transform=str(getattr(cfg, "macro_score_transform", "linear")),
        floor=float(getattr(cfg, "macro_score_floor", 0.0)),
        min_to_trade=float(getattr(cfg, "macro_score_min_to_trade", 0.25)),
        piecewise_levels=list(getattr(cfg, "macro_piecewise_levels", [0.0, 0.33, 0.66, 1.0])),
    )
    return MacroScoreResult(score=score, components=components, multiplier=mult, enabled_components=used)
