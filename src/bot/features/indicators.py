from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..acceleration.cuda_backend import get_array_module, resolve_acceleration_backend, to_numpy


def _rolling_sum_xp(arr, window: int, xp):
    if window <= 1:
        return arr.copy()
    csum = xp.cumsum(arr)
    out = csum.copy()
    out[window:] = csum[window:] - csum[:-window]
    return out


def _rolling_mean_xp(arr, window: int, min_periods: int, xp):
    valid = xp.isfinite(arr)
    x = xp.where(valid, arr, 0.0)
    count = _rolling_sum_xp(valid.astype(np.float64), window, xp)
    total = _rolling_sum_xp(x, window, xp)
    mean = xp.where(count > 0, total / xp.maximum(count, 1.0), xp.nan)
    mean = xp.where(count >= float(min_periods), mean, xp.nan)
    return mean


def _rolling_std_xp(arr, window: int, min_periods: int, xp, ddof: int = 1):
    valid = xp.isfinite(arr)
    x = xp.where(valid, arr, 0.0)

    count = _rolling_sum_xp(valid.astype(np.float64), window, xp)
    total = _rolling_sum_xp(x, window, xp)
    total_sq = _rolling_sum_xp(x * x, window, xp)

    denom = count - float(ddof)
    numer = total_sq - (total * total) / xp.where(count > 0, count, 1.0)
    var = xp.where(denom > 0, numer / denom, xp.nan)
    var = xp.maximum(var, 0.0)
    std = xp.sqrt(var)
    std = xp.where(count >= float(min_periods), std, xp.nan)
    return std


def sma(series: pd.Series, window: int, *, backend: str = "cpu") -> pd.Series:
    if backend == "cpu":
        return series.rolling(window=window, min_periods=1).mean()

    ctx = resolve_acceleration_backend(backend)
    if ctx.backend != "cuda":
        return series.rolling(window=window, min_periods=1).mean()

    xp = get_array_module(ctx)
    arr = xp.asarray(series.to_numpy(dtype=float))
    out = _rolling_mean_xp(arr, int(window), min_periods=1, xp=xp)
    return pd.Series(to_numpy(out, xp), index=series.index)


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


def realized_vol(returns: pd.Series, window: int = 24, *, backend: str = "cpu") -> pd.Series:
    # annualized to one-day hourly assumption; caller may adapt scaling
    min_periods = max(2, int(window) // 2)

    if backend == "cpu":
        rv = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(8760)
        return rv

    ctx = resolve_acceleration_backend(backend)
    if ctx.backend != "cuda":
        rv = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(8760)
        return rv

    xp = get_array_module(ctx)
    arr = xp.asarray(returns.to_numpy(dtype=float))
    std = _rolling_std_xp(arr, int(window), min_periods=min_periods, xp=xp, ddof=1)
    rv = std * xp.sqrt(8760.0)
    return pd.Series(to_numpy(rv, xp), index=returns.index)


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
