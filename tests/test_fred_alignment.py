"""Tests for FRED series alignment - no lookahead (Section 7)."""
from __future__ import annotations

import pandas as pd
import numpy as np

from bot.features.fred_features import align_fred_series_to_target


def test_fred_alignment_no_lookahead():
    """FRED observations lagged by lag_hours should not appear before they are available."""
    target_ts = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"))

    obs = pd.DataFrame({
        "observation_date": pd.to_datetime([
            "2026-01-01 00:00:00",
            "2026-01-01 06:00:00",
        ], utc=True),
        "value": [100.0, 200.0],
    })

    # With 4h lag, second observation (06:00) available at 10:00
    aligned = align_fred_series_to_target(target_ts, obs, lag_hours=4.0)

    # Hours 0-3 should be NaN (first obs available at 04:00)
    assert aligned.iloc[0] != aligned.iloc[0] or pd.isna(aligned.iloc[0])  # NaN

    # After 04:00, first value should be available
    assert aligned.iloc[4] == 100.0

    # Before 10:00, second value should NOT be visible
    assert aligned.iloc[8] == 100.0  # Still seeing first obs at 08:00

    # At 10:00 (hour 10 not in range), second should appear - check hour 9
    assert aligned.iloc[9] == 100.0  # 09:00 < 10:00, still first obs


def test_fred_alignment_empty_observations():
    """Empty observations should return NaN series."""
    target_ts = pd.Series(pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC"))
    obs = pd.DataFrame(columns=["observation_date", "value"])
    aligned = align_fred_series_to_target(target_ts, obs, lag_hours=24.0)
    assert len(aligned) == 5
    assert aligned.isna().all()


def test_fred_alignment_preserves_target_length():
    """Output should have same length as target timestamps."""
    target_ts = pd.Series(pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"))
    obs = pd.DataFrame({
        "observation_date": pd.to_datetime(["2025-12-30"], utc=True),
        "value": [50.0],
    })
    aligned = align_fred_series_to_target(target_ts, obs, lag_hours=0.0)
    assert len(aligned) == 100
