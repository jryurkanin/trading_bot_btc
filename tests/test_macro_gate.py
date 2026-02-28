from __future__ import annotations

import pandas as pd

from bot.features.macro_score import MacroGateStateMachine, MacroState


def test_macro_gate_hysteresis_and_min_durations():
    gate = MacroGateStateMachine(
        enter_threshold=0.75,
        exit_threshold=0.25,
        full_threshold=1.0,
        half_threshold=0.75,
        confirm_days=2,
        min_on_days=2,
        min_off_days=1,
    )

    days = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    scores = [0.80, 0.80, 1.00, 1.00, 0.50, 0.50, 0.00, 0.00]

    states = [gate.step(score=s, daily_ts=ts) for s, ts in zip(scores, days, strict=False)]

    assert states[0] == MacroState.OFF
    assert states[1] == MacroState.ON_HALF
    assert states[2] == MacroState.ON_HALF
    assert states[3] == MacroState.ON_FULL
    assert states[4] == MacroState.ON_FULL
    assert states[5] == MacroState.ON_HALF
    assert states[6] == MacroState.ON_HALF
    assert states[7] == MacroState.OFF


def test_macro_gate_same_day_no_advance_and_multiplier_mapping():
    gate = MacroGateStateMachine(confirm_days=1, min_on_days=1, min_off_days=1)
    ts = pd.Timestamp("2026-01-05", tz="UTC")

    first = gate.step(score=1.0, daily_ts=ts)
    age_after_first = gate.state_age_days

    second = gate.step(score=0.0, daily_ts=ts)
    assert first == MacroState.ON_FULL
    assert second == MacroState.ON_FULL
    assert gate.state_age_days == age_after_first

    assert MacroGateStateMachine.multiplier(MacroState.OFF, 0.5, 1.0) == 0.0
    assert MacroGateStateMachine.multiplier(MacroState.ON_HALF, 0.5, 1.0) == 0.5
    assert MacroGateStateMachine.multiplier(MacroState.ON_FULL, 0.5, 1.0) == 1.0
