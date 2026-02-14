from __future__ import annotations

import pandas as pd

from bot.config import RiskConfig
from bot.execution.risk import RiskManager, RiskState


def test_stale_data_breaker_handles_tz_mismatch():
    mgr = RiskManager(RiskConfig(stale_bar_max_multiplier=2))
    latest = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    now_naive = pd.Timestamp("2026-01-01 11:00:00")

    # 60m age, threshold 120m => not stale
    assert mgr.stale_data_breaker(latest, now_naive, timeframe_minutes=60) is False


def test_apply_caps_drawdown_cut():
    cfg = RiskConfig(max_drawdown_cut_pct=0.1, max_additional_exposure_on_drawdown=0.25, max_exposure=1.0)
    mgr = RiskManager(cfg)
    state = RiskState(equity_peak=1000.0, current_equity=800.0)  # 20% drawdown

    latest = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    now = pd.Timestamp("2026-01-01 10:00:00", tz="UTC")
    out = mgr.apply_caps(1.0, state, latest, now, timeframe_minutes=60)
    assert out == 0.25
