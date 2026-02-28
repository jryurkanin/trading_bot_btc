from __future__ import annotations

import pandas as pd

from bot.config import RiskConfig
from bot.execution.risk import RiskManager, RiskState


def test_daily_loss_killswitch_safe_mode_flattens():
    cfg = RiskConfig(daily_loss_limit_pct=0.05, safe_mode=True, manual_kill_switch=False)
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=950.0, day_start_equity=1000.0)

    now = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    state.day_anchor = now
    out = mgr.apply_caps(
        target_fraction=0.8,
        state=state,
        latest_bar_ts=now,
        now=now,
        timeframe_minutes=60,
        current_fraction=0.5,
    )
    assert out == 0.0


def test_killswitch_no_new_entries_when_not_safe_mode():
    cfg = RiskConfig(daily_loss_limit_pct=0.05, safe_mode=False, cutoff_no_new_entries=True)
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=900.0, day_start_equity=1000.0)
    now = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    state.day_anchor = now

    out = mgr.apply_caps(
        target_fraction=0.9,
        state=state,
        latest_bar_ts=now,
        now=now,
        timeframe_minutes=60,
        current_fraction=0.3,
    )
    assert out == 0.3
