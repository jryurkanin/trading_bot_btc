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


def resolve_acceleration_backend(requested: str | None = "auto") -> AccelerationContext:
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

    available, device_name, reason = _probe_cuda()
    if available:
        return AccelerationContext(
            requested=req,
            backend="cuda",
            cuda_available=True,
            device_name=device_name,
        )

    return AccelerationContext(
        requested=req,
        backend="cpu",
        cuda_available=False,
        reason=reason if reason else "cuda_unavailable",
    )


def get_array_module(ctx: AccelerationContext) -> Any:
    """Return numpy or cupy based on resolved backend.

    The caller should first obtain *ctx* from ``resolve_acceleration_backend``.
    """
    if ctx.backend == "cuda":
        try:
            import cupy as cp  # type: ignore

            return cp
        except Exception:
            return np
    return np


def to_numpy(arr: Any, xp: Any) -> np.ndarray:
    if xp.__name__ == "cupy":
        return xp.asnumpy(arr)
    return np.asarray(arr)
