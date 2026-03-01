"""Tests for NaN propagation in feature pipelines (Section 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.features.indicators import realized_vol, ema, donchian_channel, atr


def test_realized_vol_nan_input():
    """realized_vol should return NaN series for all-NaN input, not crash."""
    returns = pd.Series([np.nan] * 50)
    result = realized_vol(returns, window=10)
    assert len(result) == 50
    assert result.isna().all()


def test_realized_vol_with_leading_nans():
    """Leading NaNs should not corrupt later values."""
    returns = pd.Series([np.nan] * 20 + [0.01, -0.005, 0.003, 0.01, -0.01] * 10)
    result = realized_vol(returns, window=10)
    # After warmup, values should be finite
    valid = result.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid).all()


def test_ema_nan_propagation():
    """EMA should handle NaN without inf."""
    series = pd.Series([np.nan] * 5 + [100.0, 101.0, 102.0, 100.0, 99.0] * 5)
    result = ema(series, window=5)
    valid = result.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid).all()


def test_donchian_nan_input():
    """Donchian channels should not crash on NaN input."""
    high = pd.Series([np.nan] * 5 + [105.0, 106.0, 107.0, 108.0, 109.0])
    low = pd.Series([np.nan] * 5 + [95.0, 94.0, 93.0, 92.0, 91.0])
    upper, lower = donchian_channel(high, low, window=3)
    assert len(upper) == 10
    assert len(lower) == 10


def test_atr_nan_input():
    """ATR should handle NaN inputs gracefully."""
    n = 30
    high = pd.Series([np.nan] * 5 + [float(100 + i) for i in range(n - 5)])
    low = pd.Series([np.nan] * 5 + [float(95 + i) for i in range(n - 5)])
    close = pd.Series([np.nan] * 5 + [float(97 + i) for i in range(n - 5)])
    result = atr(high, low, close, window=14)
    valid = result.dropna()
    assert len(valid) > 0
    assert np.isfinite(valid).all()
