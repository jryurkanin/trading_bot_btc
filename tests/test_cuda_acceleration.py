from __future__ import annotations

import numpy as np
import pandas as pd

from bot.acceleration.cuda_backend import resolve_acceleration_backend
from bot.features.indicators import realized_vol, sma


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
