from __future__ import annotations

import numpy as np
import pandas as pd

from bot.acceleration.cuda_backend import resolve_acceleration_backend
from bot.acceleration.batch_precompute import batch_precompute_indicators
from bot.acceleration.precompute_cache import PrecomputeCache
from bot.config import RegimeConfig
from bot.features.indicators import (
    realized_vol,
    sma,
    rsi,
    bollinger_bands,
    donchian_channel,
    atr,
)
from bot.features.regime import compute_adx, compute_chop


# ---------------------------------------------------------------------------
# Helper to build a realistic OHLCV DataFrame
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 512) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 40000.0 + np.cumsum(rng.randn(n) * 100)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    opn = close + rng.randn(n) * 30
    volume = rng.uniform(100, 10000, n)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Original tests
# ---------------------------------------------------------------------------

def test_resolve_acceleration_backend_returns_supported_backend():
    ctx = resolve_acceleration_backend("auto")
    assert ctx.backend in {"cpu", "cuda"}


def test_realized_vol_cuda_path_is_shape_compatible_and_numerically_close_to_cpu():
    idx = pd.date_range("2024-01-01", periods=128, freq="h", tz="UTC")
    returns = pd.Series(np.linspace(-0.02, 0.02, 128), index=idx)

    cpu = realized_vol(returns, window=24, backend="cpu")
    maybe_cuda = realized_vol(returns, window=24, backend="cuda")

    assert len(cpu) == len(maybe_cuda)
    assert cpu.index.equals(maybe_cuda.index)
    assert np.allclose(
        cpu.fillna(0.0).to_numpy(),
        maybe_cuda.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )


def test_sma_cuda_path_is_shape_compatible_and_numerically_close_to_cpu():
    idx = pd.date_range("2024-01-01", periods=256, freq="h", tz="UTC")
    close = pd.Series(np.sin(np.linspace(0, 12, 256)) + 100.0, index=idx)

    cpu = sma(close, 20, backend="cpu")
    maybe_cuda = sma(close, 20, backend="cuda")

    assert len(cpu) == len(maybe_cuda)
    assert cpu.index.equals(maybe_cuda.index)
    assert np.allclose(
        cpu.fillna(0.0).to_numpy(),
        maybe_cuda.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Phase 1 tests: batch precompute
# ---------------------------------------------------------------------------

def test_batch_precompute_matches_sequential():
    """Critical correctness test: batch vs sequential must produce identical results."""
    df = _make_ohlcv(512)
    cfg = RegimeConfig()

    batch = batch_precompute_indicators(df, cfg, backend="cpu")
    # Also test CUDA path (falls back to CPU if unavailable)
    batch_cuda = batch_precompute_indicators(df, cfg, backend="cuda")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    seq_adx = compute_adx(high, low, close, window=cfg.adx_window, backend="cpu")
    seq_chop = compute_chop(high, low, close, window=cfg.chop_window, backend="cpu")
    seq_rv = realized_vol(close.pct_change(), int(cfg.realized_vol_window), backend="cpu")

    for label, result in [("cpu_batch", batch), ("cuda_batch", batch_cuda)]:
        assert np.allclose(
            result["adx"].fillna(0.0).to_numpy(),
            seq_adx.fillna(0.0).to_numpy(),
            atol=1e-8,
            rtol=1e-6,
        ), f"{label} ADX mismatch"

        assert np.allclose(
            result["chop"].fillna(0.0).to_numpy(),
            seq_chop.fillna(0.0).to_numpy(),
            atol=1e-8,
            rtol=1e-6,
        ), f"{label} CHOP mismatch"

        assert np.allclose(
            result["realized_vol"].fillna(0.0).to_numpy(),
            seq_rv.fillna(0.0).to_numpy(),
            atol=1e-8,
            rtol=1e-6,
        ), f"{label} realized_vol mismatch"


def test_batch_precompute_includes_orchestrator_indicators():
    """Verify donchian, atr, bollinger are returned when requested."""
    df = _make_ohlcv(256)
    cfg = RegimeConfig()

    result = batch_precompute_indicators(
        df, cfg, backend="cpu", include_orchestrator_indicators=True
    )
    assert "donchian_high" in result
    assert "donchian_low" in result
    assert "atr" in result
    assert "bb_mid" in result
    assert "bb_upper" in result
    assert "bb_lower" in result

    # Verify shapes match
    for key in ["donchian_high", "donchian_low", "atr", "bb_mid", "bb_upper", "bb_lower"]:
        assert len(result[key]) == len(df), f"{key} length mismatch"


def test_batch_precompute_shared_true_range():
    """Verify ADX and CHOP produce same results when true_range is shared vs independently computed."""
    df = _make_ohlcv(512)
    cfg = RegimeConfig()

    # Batch computes true_range once and shares it
    batch = batch_precompute_indicators(df, cfg, backend="cpu")

    # Sequential computes true_range independently in compute_adx and compute_chop
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    ind_adx = compute_adx(high, low, close, window=cfg.adx_window, backend="cpu")
    ind_chop = compute_chop(high, low, close, window=cfg.chop_window, backend="cpu")

    assert np.allclose(
        batch["adx"].fillna(0.0).to_numpy(),
        ind_adx.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )
    assert np.allclose(
        batch["chop"].fillna(0.0).to_numpy(),
        ind_chop.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Phase 3 tests: RSI CUDA
# ---------------------------------------------------------------------------

def test_rsi_cuda_matches_cpu():
    """RSI numerical equivalence between CPU and CUDA paths."""
    idx = pd.date_range("2024-01-01", periods=256, freq="h", tz="UTC")
    close = pd.Series(np.sin(np.linspace(0, 12, 256)) * 500 + 40000, index=idx)

    cpu_rsi = rsi(close, window=14, backend="cpu")
    cuda_rsi = rsi(close, window=14, backend="cuda")

    assert len(cpu_rsi) == len(cuda_rsi)
    assert cpu_rsi.index.equals(cuda_rsi.index)
    assert np.allclose(
        cpu_rsi.to_numpy(),
        cuda_rsi.to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Phase 2 tests: precompute cache
# ---------------------------------------------------------------------------

def test_precompute_cache_dedup():
    """Verify compute function called only once for same key."""
    df = _make_ohlcv(128)
    cfg = RegimeConfig()
    cache = PrecomputeCache()

    call_count = 0

    def compute_fn():
        nonlocal call_count
        call_count += 1
        return batch_precompute_indicators(df, cfg, backend="cpu")

    key = PrecomputeCache.make_key(df, cfg)
    r1 = cache.get_or_compute(key, compute_fn)
    r2 = cache.get_or_compute(key, compute_fn)

    assert call_count == 1
    assert cache.hits == 1
    assert cache.misses == 1
    # Both results should be the same object (from cache)
    assert r1 is r2


def test_precompute_cache_different_keys():
    """Verify cache miss on different indicator params."""
    df = _make_ohlcv(128)
    cache = PrecomputeCache()

    cfg1 = RegimeConfig(adx_window=14)
    cfg2 = RegimeConfig(adx_window=21)

    call_count = 0

    def compute_fn():
        nonlocal call_count
        call_count += 1
        return {"dummy": "result"}

    key1 = PrecomputeCache.make_key(df, cfg1)
    key2 = PrecomputeCache.make_key(df, cfg2)

    cache.get_or_compute(key1, compute_fn)
    cache.get_or_compute(key2, compute_fn)

    assert call_count == 2
    assert cache.misses == 2
    assert cache.hits == 0
    assert key1 != key2


# ---------------------------------------------------------------------------
# Individual indicator CUDA tests
# ---------------------------------------------------------------------------

def test_bollinger_bands_cuda_matches_cpu():
    df = _make_ohlcv(256)
    close = df["close"].astype(float)

    cpu_mid, cpu_upper, cpu_lower = bollinger_bands(close, 20, 2.0, backend="cpu")
    cuda_mid, cuda_upper, cuda_lower = bollinger_bands(close, 20, 2.0, backend="cuda")

    for label, cpu_s, cuda_s in [
        ("mid", cpu_mid, cuda_mid),
        ("upper", cpu_upper, cuda_upper),
        ("lower", cpu_lower, cuda_lower),
    ]:
        assert len(cpu_s) == len(cuda_s), f"BB {label} length mismatch"
        assert np.allclose(
            cpu_s.fillna(0.0).to_numpy(),
            cuda_s.fillna(0.0).to_numpy(),
            atol=1e-8,
            rtol=1e-6,
        ), f"BB {label} value mismatch"


def test_donchian_channel_cuda_matches_cpu():
    df = _make_ohlcv(256)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    cpu_low, cpu_high = donchian_channel(high, low, 55, backend="cpu")
    cuda_low, cuda_high = donchian_channel(high, low, 55, backend="cuda")

    for label, cpu_s, cuda_s in [
        ("low", cpu_low, cuda_low),
        ("high", cpu_high, cuda_high),
    ]:
        assert len(cpu_s) == len(cuda_s), f"Donchian {label} length mismatch"
        assert np.allclose(
            cpu_s.fillna(0.0).to_numpy(),
            cuda_s.fillna(0.0).to_numpy(),
            atol=1e-8,
            rtol=1e-6,
        ), f"Donchian {label} value mismatch"


def test_atr_cuda_matches_cpu():
    df = _make_ohlcv(256)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    cpu_atr = atr(high, low, close, 14, backend="cpu")
    cuda_atr = atr(high, low, close, 14, backend="cuda")

    assert len(cpu_atr) == len(cuda_atr)
    assert np.allclose(
        cpu_atr.fillna(0.0).to_numpy(),
        cuda_atr.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )


def test_compute_adx_cuda_matches_cpu():
    df = _make_ohlcv(256)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    cpu_adx = compute_adx(high, low, close, 14, backend="cpu")
    cuda_adx = compute_adx(high, low, close, 14, backend="cuda")

    assert len(cpu_adx) == len(cuda_adx)
    assert np.allclose(
        cpu_adx.fillna(0.0).to_numpy(),
        cuda_adx.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )


def test_compute_chop_cuda_matches_cpu():
    df = _make_ohlcv(256)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    cpu_chop = compute_chop(high, low, close, 14, backend="cpu")
    cuda_chop = compute_chop(high, low, close, 14, backend="cuda")

    assert len(cpu_chop) == len(cuda_chop)
    assert np.allclose(
        cpu_chop.fillna(0.0).to_numpy(),
        cuda_chop.fillna(0.0).to_numpy(),
        atol=1e-8,
        rtol=1e-6,
    )
