from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=1).mean()


def bollinger_bands(series: pd.Series, window: int = 20, stdev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std(ddof=0)
    upper = mid + stdev * std
    lower = mid - stdev * std
    return mid, upper, lower


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).rolling(window=window, min_periods=1).mean()


def realized_vol(returns: pd.Series, window: int = 24) -> pd.Series:
    # annualized to one-day hourly assumption; caller may adapt scaling
    rv = returns.rolling(window=window, min_periods=max(2, window // 2)).std() * np.sqrt(8760)
    return rv


def donchian_channel(high: pd.Series, low: pd.Series, window: int = 55) -> tuple[pd.Series, pd.Series]:
    return low.rolling(window=window, min_periods=1).min(), high.rolling(window=window, min_periods=1).max()


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def percentile(series: pd.Series, window: int, q: float) -> pd.Series:
    return series.rolling(window=window, min_periods=max(1, int(window * 0.5))).quantile(q)


@dataclass(frozen=True)
class IndicatorSnapshot:
    close: pd.Series
    open: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series

    @property
    def returns(self) -> pd.Series:
        return returns(self.close)
