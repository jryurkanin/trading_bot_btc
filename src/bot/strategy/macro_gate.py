"""Macro Gate modules — V4 fixed-threshold and V5 adaptive-threshold gates."""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from ..features.macro_score import MacroGateStateMachine, MacroState, macro_result
from ..config import RegimeConfig


class V4MacroGate:
    """Wraps MacroGateStateMachine with v4-specific config keys."""

    def __init__(self, cfg: RegimeConfig):
        self.cfg = cfg
        self._gate = MacroGateStateMachine(
            enter_threshold=cfg.v4_macro_enter_threshold,
            exit_threshold=cfg.v4_macro_exit_threshold,
            full_threshold=cfg.v4_macro_full_threshold,
            half_threshold=cfg.v4_macro_half_threshold,
            confirm_days=cfg.v4_macro_confirm_days,
            min_on_days=cfg.v4_macro_min_on_days,
            min_off_days=cfg.v4_macro_min_off_days,
        )
        self._last_daily_ts: pd.Timestamp | None = None
        self._cached_score: float = 0.0
        self._cached_components: dict[str, float] = {}
        self._cached_state: MacroState = MacroState.OFF
        self._cached_multiplier: float = 0.0

    def reset(self) -> None:
        self._gate.reset()
        self._last_daily_ts = None
        self._cached_score = 0.0
        self._cached_components = {}
        self._cached_state = MacroState.OFF
        self._cached_multiplier = 0.0

    def update(
        self,
        daily_closed: pd.DataFrame,
        daily_bar_ts: pd.Timestamp | None,
    ) -> tuple[MacroState, float, float, dict[str, float]]:
        """Compute macro state for a given daily bar.

        Returns ``(state, multiplier, score, components)``.
        """
        if daily_bar_ts is not None and self._last_daily_ts is not None:
            if pd.Timestamp(daily_bar_ts) == pd.Timestamp(self._last_daily_ts):
                return (
                    self._cached_state,
                    self._cached_multiplier,
                    self._cached_score,
                    self._cached_components,
                )

        macro = macro_result(daily_closed, self.cfg)
        score = float(macro.score)
        components = dict(macro.components)
        state = self._gate.step(score, daily_bar_ts)
        mult = MacroGateStateMachine.multiplier(
            state,
            half_multiplier=self.cfg.v4_macro_half_multiplier,
            full_multiplier=self.cfg.v4_macro_full_multiplier,
        )

        self._last_daily_ts = daily_bar_ts
        self._cached_score = score
        self._cached_components = components
        self._cached_state = state
        self._cached_multiplier = mult

        return state, mult, score, components

    @property
    def state(self) -> MacroState:
        return self._cached_state

    @property
    def multiplier(self) -> float:
        return self._cached_multiplier

    @property
    def score(self) -> float:
        return self._cached_score


# ---------------------------------------------------------------------------
# V5 Adaptive Macro Gate
# ---------------------------------------------------------------------------

class AdaptiveMacroGate:
    """Macro gate with volatility-adapted thresholds.

    Instead of fixed enter/exit/half/full thresholds, the gate adjusts them
    based on where the current daily realized volatility sits relative to its
    recent history:

    * **High vol** → widen the hysteresis band (harder to enter, harder to
      exit = stay in current state longer, avoid whipsaw).
    * **Low vol** → tighten the band (more responsive to score changes,
      capture trends earlier).

    The adjustment is symmetric around the base thresholds, scaled by
    ``sensitivity * vol_z`` where ``vol_z`` is the z-score of current daily
    vol relative to a trailing window.

    The underlying state machine is a standard ``MacroGateStateMachine``
    whose thresholds are updated before each ``step()``.
    """

    def __init__(self, cfg: RegimeConfig) -> None:
        self.cfg = cfg

        # Base thresholds (centers of the adaptive range)
        self._enter_base = float(cfg.v5_adaptive_enter_base)
        self._exit_base = float(cfg.v5_adaptive_exit_base)
        self._half_base = float(cfg.v5_adaptive_half_base)
        self._full_base = float(cfg.v5_adaptive_full_base)
        self._sensitivity = float(cfg.v5_adaptive_sensitivity)

        self._gate = MacroGateStateMachine(
            enter_threshold=self._enter_base,
            exit_threshold=self._exit_base,
            full_threshold=self._full_base,
            half_threshold=self._half_base,
            confirm_days=cfg.v5_adaptive_confirm_days,
            min_on_days=cfg.v5_adaptive_min_on_days,
            min_off_days=cfg.v5_adaptive_min_off_days,
        )

        self._vol_window = max(10, int(cfg.v5_adaptive_vol_window_days))
        self._vol_history: deque[float] = deque(maxlen=self._vol_window)

        # Cache
        self._last_daily_ts: pd.Timestamp | None = None
        self._cached_score: float = 0.0
        self._cached_components: dict[str, float] = {}
        self._cached_state: MacroState = MacroState.OFF
        self._cached_multiplier: float = 0.0
        self._cached_thresholds: dict[str, float] = {}

    def reset(self) -> None:
        self._gate.reset()
        self._vol_history.clear()
        self._last_daily_ts = None
        self._cached_score = 0.0
        self._cached_components = {}
        self._cached_state = MacroState.OFF
        self._cached_multiplier = 0.0
        self._cached_thresholds = {}

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _compute_vol_z(self, current_vol: float) -> float:
        """Z-score of current vol vs trailing window. Returns 0 if insufficient data."""
        if len(self._vol_history) < 5:
            return 0.0
        arr = np.array(self._vol_history)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        if sigma < 1e-12:
            return 0.0
        return (current_vol - mu) / sigma

    def _adapt_thresholds(self, vol_z: float) -> None:
        """Shift gate thresholds based on vol z-score.

        Positive vol_z (high vol) → raise enter/full, lower exit/half (wider band).
        Negative vol_z (low vol)  → lower enter/full, raise exit/half (tighter band).
        """
        shift = self._clamp(vol_z * self._sensitivity, -0.25, 0.25)

        # Upper clamp allows intentionally-high bases (e.g., 2.0) to disable
        # the gate by producing thresholds unreachable by a [0, 1] score.
        enter_hi = max(0.95, self._enter_base)
        full_hi = max(1.0, self._full_base)
        half_hi = max(0.95, self._half_base)

        self._gate.enter_threshold = self._clamp(self._enter_base + shift, 0.10, enter_hi)
        self._gate.exit_threshold = self._clamp(self._exit_base - shift, 0.05, 0.90)
        self._gate.half_threshold = self._clamp(self._half_base + shift, 0.10, half_hi)
        self._gate.full_threshold = self._clamp(self._full_base + shift, 0.30, full_hi)

        # Enforce ordering invariants
        if self._gate.exit_threshold >= self._gate.enter_threshold:
            mid = (self._enter_base + self._exit_base) / 2
            self._gate.enter_threshold = mid + 0.05
            self._gate.exit_threshold = mid - 0.05
        if self._gate.half_threshold > self._gate.full_threshold:
            self._gate.half_threshold = self._gate.full_threshold

    def _daily_realized_vol(self, daily_closed: pd.DataFrame) -> float:
        """Compute trailing daily realized vol from close prices."""
        if daily_closed is None or len(daily_closed) < 2:
            return 0.0
        close = daily_closed["close"].astype(float)
        returns = close.pct_change().dropna()
        if len(returns) < 2:
            return 0.0
        window = min(20, len(returns))
        rv = float(returns.iloc[-window:].std() * np.sqrt(365))
        return rv if np.isfinite(rv) else 0.0

    def update(
        self,
        daily_closed: pd.DataFrame,
        daily_bar_ts: pd.Timestamp | None,
    ) -> tuple[MacroState, float, float, dict[str, float]]:
        """Compute macro state with adaptive thresholds.

        Returns ``(state, multiplier, score, components)``.
        """
        if daily_bar_ts is not None and self._last_daily_ts is not None:
            if pd.Timestamp(daily_bar_ts) == pd.Timestamp(self._last_daily_ts):
                return (
                    self._cached_state,
                    self._cached_multiplier,
                    self._cached_score,
                    self._cached_components,
                )

        # Compute daily vol and adapt thresholds
        daily_vol = self._daily_realized_vol(daily_closed)
        vol_z = self._compute_vol_z(daily_vol)
        self._adapt_thresholds(vol_z)
        self._vol_history.append(daily_vol)

        # Score and step
        macro = macro_result(daily_closed, self.cfg)
        score = float(macro.score)
        components = dict(macro.components)
        state = self._gate.step(score, daily_bar_ts)
        mult = MacroGateStateMachine.multiplier(
            state,
            half_multiplier=self.cfg.v5_adaptive_half_multiplier,
            full_multiplier=self.cfg.v5_adaptive_full_multiplier,
        )

        self._last_daily_ts = daily_bar_ts
        self._cached_score = score
        self._cached_components = components
        self._cached_state = state
        self._cached_multiplier = mult
        self._cached_thresholds = {
            "enter": self._gate.enter_threshold,
            "exit": self._gate.exit_threshold,
            "half": self._gate.half_threshold,
            "full": self._gate.full_threshold,
            "vol_z": vol_z,
            "daily_vol": daily_vol,
        }

        return state, mult, score, components

    @property
    def state(self) -> MacroState:
        return self._cached_state

    @property
    def multiplier(self) -> float:
        return self._cached_multiplier

    @property
    def score(self) -> float:
        return self._cached_score

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._cached_thresholds)
