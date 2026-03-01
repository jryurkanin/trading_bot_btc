from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

import numpy as np


AccelerationBackend = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class AccelerationContext:
    requested: str
    backend: AccelerationBackend
    cuda_available: bool
    device_name: str | None = None
    reason: str | None = None


def _probe_cuda() -> tuple[bool, str | None, str | None]:
    """Return (available, device_name, reason)."""
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        return False, None, f"cupy_import_error:{exc.__class__.__name__}"

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            return False, None, "no_cuda_devices"
        dev = cp.cuda.Device(0)
        name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")

        # Real compute preflight (catches missing NVRTC/toolchain issues that
        # getDeviceCount alone does not detect).
        try:
            x = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float32)
            y = (x * 2.0).sum()
            _ = float(y.get())
        except Exception as exc:
            msg = str(exc).replace("\n", " ")[:180]
            return False, str(name), f"cuda_compute_error:{exc.__class__.__name__}:{msg}"

        return True, str(name), None
    except Exception as exc:
        return False, None, f"cuda_probe_error:{exc.__class__.__name__}"


_cached_ctx: AccelerationContext | None = None


def resolve_acceleration_backend(requested: str | None = "auto") -> AccelerationContext:
    global _cached_ctx
    req = str(requested or "auto").strip().lower()
    if req not in {"auto", "cpu", "cuda"}:
        req = "auto"

    if req == "cpu":
        return AccelerationContext(
            requested=req,
            backend="cpu",
            cuda_available=False,
            reason="forced_cpu",
        )

    if _cached_ctx is not None:
        return _cached_ctx

    available, device_name, reason = _probe_cuda()
    if available:
        _cached_ctx = AccelerationContext(
            requested=req,
            backend="cuda",
            cuda_available=True,
            device_name=device_name,
        )
    else:
        _cached_ctx = AccelerationContext(
            requested=req,
            backend="cpu",
            cuda_available=False,
            reason=reason if reason else "cuda_unavailable",
        )
    return _cached_ctx


_cached_xp: Any = None


def get_array_module(ctx: AccelerationContext) -> Any:
    """Return numpy or cupy based on resolved backend.

    The caller should first obtain *ctx* from ``resolve_acceleration_backend``.
    """
    global _cached_xp
    if ctx.backend == "cuda":
        if _cached_xp is not None:
            return _cached_xp
        try:
            import cupy as cp  # type: ignore

            _cached_xp = cp
            return cp
        except Exception:
            return np
    return np


def to_numpy(arr: Any, xp: Any) -> np.ndarray:
    if xp.__name__ == "cupy":
        return xp.asnumpy(arr)
    return np.asarray(arr)


def upload_ohlcv(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray | None,
    xp: Any,
) -> tuple[Any, Any, Any, Any | None]:
    """Upload OHLCV numpy arrays to GPU in one batch.

    Returns (high_gpu, low_gpu, close_gpu, volume_gpu).
    volume_gpu is None if volume input is None.
    """
    h = xp.asarray(high)
    l = xp.asarray(low)
    c = xp.asarray(close)
    v = xp.asarray(volume) if volume is not None else None
    return h, l, c, v


def download_batch(results: dict[str, Any], xp: Any) -> dict[str, np.ndarray]:
    """Download a dict of GPU arrays to numpy in one batch.

    Stacks all arrays into a single contiguous block, transfers once,
    then slices back. Falls back to per-array transfer if shapes differ.
    """
    if not results:
        return {}

    if xp.__name__ != "cupy":
        return {k: np.asarray(v) for k, v in results.items()}

    keys = list(results.keys())
    arrays = [results[k] for k in keys]

    # Check if all arrays have the same shape for stacked transfer
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) == 1:
        stacked = xp.stack(arrays, axis=0)
        stacked_np = xp.asnumpy(stacked)
        return {k: stacked_np[i] for i, k in enumerate(keys)}

    # Fallback: per-array download if shapes differ
    return {k: xp.asnumpy(v) for k, v in zip(keys, arrays)}


def estimate_transfer_overhead_ms(ctx: AccelerationContext) -> float | None:
    """Benchmark a small GPU round trip to characterize interconnect latency.

    Returns estimated overhead in milliseconds, or None if CUDA unavailable.
    Typical results:
    - PCIe x16: 0.1-0.5ms
    - OCuLink/Thunderbolt: 1-3ms
    """
    if ctx.backend != "cuda":
        return None

    xp = get_array_module(ctx)
    if xp.__name__ != "cupy":
        return None

    try:
        import time

        # Warm up
        test_data = np.random.randn(32768).astype(np.float32)  # ~128KB
        gpu = xp.asarray(test_data)
        _ = xp.asnumpy(gpu)

        # Benchmark 5 round trips
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            gpu = xp.asarray(test_data)
            _ = xp.asnumpy(gpu)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return float(np.median(times))
    except Exception:
        return None
