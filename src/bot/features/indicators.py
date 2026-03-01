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
    # First `window` elements are partial sums (cumsum gives sum of [0..i]),
    # elements from index `window` onward get the correct rolling window.
    out[window:] = csum[window:] - csum[:-window]
    # Elements [0..window-1] already hold cumsum values which represent
    # partial sums of 1..window elements respectively — this is correct
    # for callers that apply their own min_periods masking.
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


def _rolling_max_xp(arr, window: int, min_periods: int, xp):
    window = max(1, int(window))
    min_periods = max(1, int(min_periods))
    n = int(arr.shape[0])
    out = xp.full(n, xp.nan, dtype=arr.dtype)
    if n == 0:
        return out

    try:
        sw = xp.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
        full = xp.nanmax(sw, axis=1)
        out[window - 1 :] = full
    except Exception:
        # Fallback path for array modules without sliding_window_view support.
        for i in range(window - 1, n):
            out[i] = xp.nanmax(arr[i - window + 1 : i + 1])

    head = min(window - 1, n)
    for i in range(head):
        if i + 1 >= min_periods:
            out[i] = xp.nanmax(arr[: i + 1])

    if min_periods > 1:
        out[: min(min_periods - 1, n)] = xp.nan
    return out


def _rolling_min_xp(arr, window: int, min_periods: int, xp):
    window = max(1, int(window))
    min_periods = max(1, int(min_periods))
    n = int(arr.shape[0])
    out = xp.full(n, xp.nan, dtype=arr.dtype)
    if n == 0:
        return out

    try:
        sw = xp.lib.stride_tricks.sliding_window_view(arr, window_shape=window)
        full = xp.nanmin(sw, axis=1)
        out[window - 1 :] = full
    except Exception:
        # Fallback path for array modules without sliding_window_view support.
        for i in range(window - 1, n):
            out[i] = xp.nanmin(arr[i - window + 1 : i + 1])

    head = min(window - 1, n)
    for i in range(head):
        if i + 1 >= min_periods:
            out[i] = xp.nanmin(arr[: i + 1])

    if min_periods > 1:
        out[: min(min_periods - 1, n)] = xp.nan
    return out


def _resolve_fast(backend: str):
    """Fast-path: skip probe when CPU is requested."""
    if backend == "cpu":
        return None, np  # None ctx signals CPU
    ctx = resolve_acceleration_backend(backend)
    if ctx.backend != "cuda":
        return None, np
    return ctx, get_array_module(ctx)


def _true_range_xp(high_arr, low_arr, close_arr, xp):
    """Compute true range directly on array-module arrays (numpy or cupy)."""
    prev_close = xp.empty_like(close_arr)
    prev_close[0] = close_arr[0]
    prev_close[1:] = close_arr[:-1]
    tr1 = high_arr - low_arr
    tr2 = xp.abs(high_arr - prev_close)
    tr3 = xp.abs(low_arr - prev_close)
    return xp.maximum(xp.maximum(tr1, tr2), tr3)


def sma(series: pd.Series, window: int, *, backend: str = "cpu") -> pd.Series:
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        return series.rolling(window=window, min_periods=1).mean()

    arr = xp.asarray(series.to_numpy(dtype=float))
    out = _rolling_mean_xp(arr, int(window), min_periods=1, xp=xp)
    return pd.Series(to_numpy(out, xp), index=series.index)


def ema(series: pd.Series, window: int, *, backend: str = "cpu") -> pd.Series:
    # EMA is inherently sequential — pandas C implementation is fastest
    # regardless of backend. A Python for-loop on GPU is slower than CPU.
    return series.ewm(span=window, adjust=False, min_periods=1).mean()


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    stdev: float = 2.0,
    *,
    backend: str = "cpu",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        mid = sma(series, window)
        std = series.rolling(window=window, min_periods=1).std(ddof=0)
        upper = mid + stdev * std
        lower = mid - stdev * std
        return mid, upper, lower

    arr = xp.asarray(series.to_numpy(dtype=float))
    mean = _rolling_mean_xp(arr, int(window), min_periods=1, xp=xp)
    std = _rolling_std_xp(arr, int(window), min_periods=1, xp=xp, ddof=0)
    upper = mean + float(stdev) * std
    lower = mean - float(stdev) * std
    return (
        pd.Series(to_numpy(mean, xp), index=series.index),
        pd.Series(to_numpy(upper, xp), index=series.index),
        pd.Series(to_numpy(lower, xp), index=series.index),
    )


def rsi(series: pd.Series, window: int = 14, *, backend: str = "cpu") -> pd.Series:
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
        loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        out = 100 - (100 / (1 + rs))
        return out.fillna(50.0)

    arr = xp.asarray(series.to_numpy(dtype=float))
    delta = xp.empty_like(arr)
    delta[0] = xp.nan
    delta[1:] = arr[1:] - arr[:-1]

    gain = xp.where(delta > 0, delta, 0.0)
    loss = xp.where(delta < 0, -delta, 0.0)

    avg_gain = _rolling_mean_xp(gain, int(window), min_periods=1, xp=xp)
    avg_loss = _rolling_mean_xp(loss, int(window), min_periods=1, xp=xp)

    rs = avg_gain / xp.where(avg_loss != 0, avg_loss, xp.nan)
    rsi_arr = 100.0 - (100.0 / (1.0 + rs))

    result = pd.Series(to_numpy(rsi_arr, xp), index=series.index)
    return result.fillna(50.0)


def true_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    backend: str = "cpu",
) -> pd.Series:
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        # Vectorized numpy path — avoids 3 intermediate Series + DataFrame concat
        h = high.to_numpy(dtype=float)
        l = low.to_numpy(dtype=float)
        c = close.to_numpy(dtype=float)
        tr = _true_range_xp(h, l, c, np)
        return pd.Series(tr, index=high.index)

    high_arr = xp.asarray(high.to_numpy(dtype=float))
    low_arr = xp.asarray(low.to_numpy(dtype=float))
    close_arr = xp.asarray(close.to_numpy(dtype=float))
    tr = _true_range_xp(high_arr, low_arr, close_arr, xp)
    return pd.Series(to_numpy(tr, xp), index=high.index)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    *,
    backend: str = "cpu",
) -> pd.Series:
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        return true_range(high, low, close).rolling(window=window, min_periods=1).mean()

    high_arr = xp.asarray(high.to_numpy(dtype=float))
    low_arr = xp.asarray(low.to_numpy(dtype=float))
    close_arr = xp.asarray(close.to_numpy(dtype=float))
    tr = _true_range_xp(high_arr, low_arr, close_arr, xp)
    out = _rolling_mean_xp(tr, int(window), min_periods=1, xp=xp)
    return pd.Series(to_numpy(out, xp), index=high.index)


def realized_vol(returns: pd.Series, window: int = 24, *, backend: str = "cpu", periods_per_year: int = 8760) -> pd.Series:
    """Annualized realized volatility.

    Parameters
    ----------
    periods_per_year : int
        Number of return observations per year.  Default 8760 assumes hourly
        data (365 * 24).  Pass 365 for daily data or 52 for weekly.
    """
    min_periods = max(2, int(window) // 2)
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        rv = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(periods_per_year)
        return rv

    arr = xp.asarray(returns.to_numpy(dtype=float))
    std = _rolling_std_xp(arr, int(window), min_periods=min_periods, xp=xp, ddof=1)
    rv = std * xp.sqrt(float(periods_per_year))
    return pd.Series(to_numpy(rv, xp), index=returns.index)


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    window: int = 55,
    *,
    backend: str = "cpu",
) -> tuple[pd.Series, pd.Series]:
    ctx, xp = _resolve_fast(backend)

    if ctx is None:
        return low.rolling(window=window, min_periods=1).min(), high.rolling(window=window, min_periods=1).max()

    high_arr = xp.asarray(high.to_numpy(dtype=float))
    low_arr = xp.asarray(low.to_numpy(dtype=float))

    low_ch = _rolling_min_xp(low_arr, int(window), min_periods=1, xp=xp)
    high_ch = _rolling_max_xp(high_arr, int(window), min_periods=1, xp=xp)
    return (
        pd.Series(to_numpy(low_ch, xp), index=low.index),
        pd.Series(to_numpy(high_ch, xp), index=high.index),
    )


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
