"""Tests for DrawdownBreaker re-entry behaviour (Section 7)."""
from __future__ import annotations

import pandas as pd

from bot.config import RiskConfig
from bot.execution.risk import RiskManager, RiskState


def test_drawdown_breaker_allows_reentry_after_recovery():
    """After drawdown exceeds the cut threshold and then recovers, full exposure should be allowed."""
    cfg = RiskConfig(max_drawdown_cut_pct=0.10, max_additional_exposure_on_drawdown=0.25, max_exposure=1.0)
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=850.0)  # 15% dd > 10%

    # While in drawdown, exposure is capped
    factor = mgr.check_drawdown(state)
    assert factor == 0.25

    # Equity recovers past the peak
    state.current_equity = 1010.0
    state.equity_peak = 1010.0

    factor = mgr.check_drawdown(state)
    assert factor == 1.0


def test_drawdown_breaker_caps_during_drawdown():
    cfg = RiskConfig(max_drawdown_cut_pct=0.05, max_additional_exposure_on_drawdown=0.0, max_exposure=1.0)
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=940.0)  # 6% dd > 5%

    factor = mgr.check_drawdown(state)
    assert factor == 0.0


def test_consecutive_loss_resets_on_winning_trade():
    """record_trade should reset consecutive_losses to 0 on a winning trade."""
    cfg = RiskConfig()
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=1000.0, last_trade_equity=1000.0)

    # Three losing trades
    mgr.record_trade(state, 990.0)
    mgr.record_trade(state, 980.0)
    mgr.record_trade(state, 970.0)
    assert state.consecutive_losses == 3

    # One winning trade resets
    mgr.record_trade(state, 985.0)
    assert state.consecutive_losses == 0


def test_day_rollover_resets_consecutive_losses():
    """update_runtime_state should reset consecutive_losses on new day."""
    cfg = RiskConfig()
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=1000.0)
    state.consecutive_losses = 5

    day1 = pd.Timestamp("2026-01-01 23:00:00", tz="UTC")
    mgr.update_runtime_state(state, 1000.0, day1)

    day2 = pd.Timestamp("2026-01-02 01:00:00", tz="UTC")
    mgr.update_runtime_state(state, 1000.0, day2)
    assert state.consecutive_losses == 0
