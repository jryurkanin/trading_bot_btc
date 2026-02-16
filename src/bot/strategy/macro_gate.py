"""V4 Macro Gate — single source of truth for macro state."""
from __future__ import annotations

from typing import Any

import pandas as pd

from ..features.macro_score import MacroGateStateMachine, MacroState, compute_macro_score
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

        score, components, _ = compute_macro_score(daily_closed)
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
