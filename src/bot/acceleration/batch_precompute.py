"""Batch GPU precompute — uploads OHLCV once, computes all indicators, downloads once.

This eliminates redundant OCuLink/PCIe round trips that occur when each indicator
function independently uploads the same data to GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cuda_backend import (
    AccelerationContext,
    get_array_module,
    resolve_acceleration_backend,
    to_numpy,
    upload_ohlcv,
    download_batch,
)
from ..system_log import get_system_logger

logger = get_system_logger("acceleration.batch_precompute")


def _get_xp_helpers():
    """Lazy import to avoid circular dependency (indicators -> acceleration -> batch_precompute -> indicators)."""
    from ..features.indicators import (
        _true_range_xp,
        _rolling_mean_xp,
        _rolling_std_xp,
        _rolling_sum_xp,
        _rolling_max_xp,
        _rolling_min_xp,
    )
    return _true_range_xp, _rolling_mean_xp, _rolling_std_xp, _rolling_sum_xp, _rolling_max_xp, _rolling_min_xp


def _compute_adx_di_xp(high_arr, low_arr, close_arr, tr_arr, window: int, xp):
    """Compute ADX, +DI, -DI on GPU arrays, reusing pre-computed true_range."""
    _, _rolling_mean_xp, _, _, _, _ = _get_xp_helpers()

    prev_high = xp.empty_like(high_arr)
    prev_low = xp.empty_like(low_arr)
    prev_high[0] = high_arr[0]
    prev_low[0] = low_arr[0]
    prev_high[1:] = high_arr[:-1]
    prev_low[1:] = low_arr[:-1]

    up_move = high_arr - prev_high
    down_move = prev_low - low_arr

    plus_dm = xp.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = xp.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _rolling_mean_xp(tr_arr, window, min_periods=window, xp=xp)

    plus_dm_avg = _rolling_mean_xp(plus_dm, window, min_periods=window, xp=xp)
    minus_dm_avg = _rolling_mean_xp(minus_dm, window, min_periods=window, xp=xp)

    plus_di = 100.0 * (plus_dm_avg / xp.where(atr != 0, atr, xp.nan))
    minus_di = 100.0 * (minus_dm_avg / xp.where(atr != 0, atr, xp.nan))

    di_sum = plus_di + minus_di
    dx = 100.0 * xp.abs(plus_di - minus_di) / xp.where(di_sum != 0, di_sum, xp.nan)
    adx = _rolling_mean_xp(dx, window, min_periods=window, xp=xp)

    return adx, plus_di, minus_di


def _compute_chop_xp(tr_arr, high_arr, low_arr, window: int, xp):
    """Compute Choppiness Index on GPU arrays, reusing pre-computed true_range."""
    _, _, _, _rolling_sum_xp, _rolling_max_xp, _rolling_min_xp = _get_xp_helpers()

    tr_sum = _rolling_sum_xp(tr_arr, window, xp)
    valid_count = _rolling_sum_xp(xp.isfinite(tr_arr).astype(np.float64), window, xp)
    tr_sum = xp.where(valid_count >= float(window), tr_sum, xp.nan)

    high_max = _rolling_max_xp(high_arr, window, min_periods=window, xp=xp)
    low_min = _rolling_min_xp(low_arr, window, min_periods=window, xp=xp)

    denominator = high_max - low_min
    chop = 100.0 * xp.log10(
        tr_sum / xp.where(denominator != 0, denominator, xp.nan)
    ) / xp.log10(float(window))

    return chop


def batch_precompute_indicators(
    hourly_df: pd.DataFrame,
    cfg: Any,
    *,
    backend: str = "cpu",
    include_orchestrator_indicators: bool = False,
    include_rsi: bool = False,
) -> dict[str, pd.Series]:
    """Batch-compute all regime indicators with minimal GPU transfers.

    When backend is "cuda", uploads OHLCV arrays once, computes all indicators
    on GPU without intermediate downloads, then downloads all results in one batch.

    Parameters
    ----------
    hourly_df : pd.DataFrame
        Hourly OHLCV data with 'high', 'low', 'close', 'volume' columns.
    cfg : RegimeConfig
        Configuration with adx_window, chop_window, realized_vol_window, etc.
    backend : str
        "cpu" or "cuda".
    include_orchestrator_indicators : bool
        If True, also compute donchian, atr, and bollinger bands for
        legacy orchestrator strategies.
    include_rsi : bool
        If True, also compute RSI from close prices.

    Returns
    -------
    dict[str, pd.Series]
        Computed indicator series keyed by name.
    """
    if hourly_df is None or hourly_df.empty:
        return {}

    # CPU fallback — use sequential indicator calls
    if backend == "cpu":
        return _batch_precompute_cpu(hourly_df, cfg, include_orchestrator_indicators, include_rsi)

    ctx = resolve_acceleration_backend(backend)
    if ctx.backend != "cuda":
        return _batch_precompute_cpu(hourly_df, cfg, include_orchestrator_indicators, include_rsi)

    xp = get_array_module(ctx)
    _true_range_xp, _rolling_mean_xp, _rolling_std_xp, _rolling_sum_xp, _rolling_max_xp, _rolling_min_xp = _get_xp_helpers()

    # --- Single upload of OHLCV arrays ---
    high_np = hourly_df["high"].to_numpy(dtype=float)
    low_np = hourly_df["low"].to_numpy(dtype=float)
    close_np = hourly_df["close"].to_numpy(dtype=float)

    high_arr, low_arr, close_arr, _vol_arr = upload_ohlcv(high_np, low_np, close_np, None, xp)

    index = hourly_df.index

    # --- Compute shared true_range once (reused by ADX, CHOP, ATR) ---
    tr_arr = _true_range_xp(high_arr, low_arr, close_arr, xp)

    # --- ADX + DI ---
    adx_window = int(cfg.adx_window)
    adx_arr, plus_di_arr, minus_di_arr = _compute_adx_di_xp(
        high_arr, low_arr, close_arr, tr_arr, adx_window, xp
    )

    # --- CHOP ---
    chop_window = int(cfg.chop_window)
    chop_arr = _compute_chop_xp(tr_arr, high_arr, low_arr, chop_window, xp)

    # --- Realized vol from close returns ---
    returns_arr = xp.empty_like(close_arr)
    returns_arr[0] = xp.nan
    returns_arr[1:] = (close_arr[1:] - close_arr[:-1]) / close_arr[:-1]

    rv_window = int(cfg.realized_vol_window)
    rv_min_periods = max(2, rv_window // 2)
    rv_std = _rolling_std_xp(returns_arr, rv_window, min_periods=rv_min_periods, xp=xp, ddof=1)
    rv_arr = rv_std * xp.sqrt(8760.0)

    # --- Collect GPU results for batch download ---
    gpu_results = {
        "adx": adx_arr,
        "plus_di": plus_di_arr,
        "minus_di": minus_di_arr,
        "chop": chop_arr,
        "realized_vol": rv_arr,
    }

    # --- Orchestrator indicators (optional) ---
    if include_orchestrator_indicators:
        # Donchian channel
        donchian_window = int(cfg.donchian_window)
        donchian_low_arr = _rolling_min_xp(low_arr, donchian_window, min_periods=1, xp=xp)
        donchian_high_arr = _rolling_max_xp(high_arr, donchian_window, min_periods=1, xp=xp)
        gpu_results["donchian_low"] = donchian_low_arr
        gpu_results["donchian_high"] = donchian_high_arr

        # ATR (reuses shared true_range)
        atr_window = int(cfg.atr_window)
        atr_arr = _rolling_mean_xp(tr_arr, atr_window, min_periods=1, xp=xp)
        gpu_results["atr"] = atr_arr

        # Bollinger bands
        bb_window = int(cfg.bb_window)
        bb_stdev = float(cfg.bb_stdev)
        bb_mean = _rolling_mean_xp(close_arr, bb_window, min_periods=1, xp=xp)
        bb_std = _rolling_std_xp(close_arr, bb_window, min_periods=1, xp=xp, ddof=0)
        bb_upper = bb_mean + bb_stdev * bb_std
        bb_lower = bb_mean - bb_stdev * bb_std
        gpu_results["bb_mid"] = bb_mean
        gpu_results["bb_upper"] = bb_upper
        gpu_results["bb_lower"] = bb_lower

    # --- RSI (optional) ---
    if include_rsi:
        rsi_delta = xp.empty_like(close_arr)
        rsi_delta[0] = xp.nan
        rsi_delta[1:] = close_arr[1:] - close_arr[:-1]
        rsi_gain = xp.where(rsi_delta > 0, rsi_delta, 0.0)
        rsi_loss = xp.where(rsi_delta < 0, -rsi_delta, 0.0)
        rsi_avg_gain = _rolling_mean_xp(rsi_gain, 14, min_periods=1, xp=xp)
        rsi_avg_loss = _rolling_mean_xp(rsi_loss, 14, min_periods=1, xp=xp)
        rs = rsi_avg_gain / xp.where(rsi_avg_loss != 0, rsi_avg_loss, xp.nan)
        gpu_results["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # --- Single batch download ---
    np_results = download_batch(gpu_results, xp)

    # --- Wrap in pd.Series with correct index ---
    out: dict[str, pd.Series] = {}
    for key, arr in np_results.items():
        s = pd.Series(arr, index=index)
        if key in {"adx", "plus_di", "minus_di", "chop"}:
            s = s.fillna(0.0)
        elif key == "rsi":
            s = s.fillna(50.0)
        out[key] = s

    out["acceleration_backend"] = backend  # type: ignore[assignment]

    logger.debug(
        "batch_precompute_done backend=%s n_rows=%d keys=%s orchestrator=%s",
        backend,
        len(hourly_df),
        sorted(out.keys()),
        include_orchestrator_indicators,
    )

    return out


def _batch_precompute_cpu(
    hourly_df: pd.DataFrame,
    cfg: Any,
    include_orchestrator_indicators: bool,
    include_rsi: bool = False,
) -> dict[str, pd.Series]:
    """CPU fallback — sequential indicator computation."""
    from ..features.indicators import (
        true_range,
        realized_vol,
        bollinger_bands,
        donchian_channel,
        atr as compute_atr,
        rsi as compute_rsi,
    )
    from ..features.regime import compute_adx_di, compute_chop

    high = hourly_df["high"].astype(float)
    low = hourly_df["low"].astype(float)
    close = hourly_df["close"].astype(float)

    adx, plus_di, minus_di = compute_adx_di(high, low, close, window=cfg.adx_window, backend="cpu")
    chop = compute_chop(high, low, close, window=cfg.chop_window, backend="cpu")
    rv = realized_vol(close.pct_change(), int(cfg.realized_vol_window), backend="cpu")

    out: dict[str, Any] = {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "chop": chop,
        "realized_vol": rv,
        "acceleration_backend": "cpu",
    }

    if include_orchestrator_indicators:
        donchian_low, donchian_high = donchian_channel(high, low, cfg.donchian_window, backend="cpu")
        out["donchian_low"] = donchian_low
        out["donchian_high"] = donchian_high
        out["atr"] = compute_atr(high, low, close, cfg.atr_window, backend="cpu")

        bb_mid, bb_upper, bb_lower = bollinger_bands(close, cfg.bb_window, cfg.bb_stdev, backend="cpu")
        out["bb_mid"] = bb_mid
        out["bb_upper"] = bb_upper
        out["bb_lower"] = bb_lower

    if include_rsi:
        out["rsi"] = compute_rsi(close, 14, backend="cpu")

    return out
