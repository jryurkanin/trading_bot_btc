"""Precompute cache — avoids redundant indicator recomputation across frontier sweeps.

When macro-level parameters vary but indicator parameters (adx_window, chop_window,
etc.) stay fixed, all sweep iterations share the same precomputed indicator series.
This cache stores results keyed by (data fingerprint, indicator params) so compute
happens only once per unique combination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..system_log import get_system_logger

logger = get_system_logger("acceleration.precompute_cache")


@dataclass(frozen=True)
class PrecomputeCacheKey:
    n_rows: int
    data_fingerprint: int  # hash of first/last close + shape
    adx_window: int
    chop_window: int
    realized_vol_window: int
    bb_window: int
    donchian_window: int
    atr_window: int


class PrecomputeCache:
    """Thread-local cache for precomputed indicator series.

    Designed for use within a single worker process during frontier sweeps.
    """

    def __init__(self) -> None:
        self._store: dict[PrecomputeCacheKey, dict[str, Any]] = {}
        self._hits: int = 0
        self._misses: int = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def get_or_compute(
        self,
        key: PrecomputeCacheKey,
        compute_fn: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Return cached result or compute and cache."""
        if key in self._store:
            self._hits += 1
            logger.debug("precompute_cache_hit key=%s hits=%d", key, self._hits)
            return self._store[key]

        self._misses += 1
        result = compute_fn()
        self._store[key] = result
        logger.debug(
            "precompute_cache_miss key=%s misses=%d store_size=%d",
            key,
            self._misses,
            len(self._store),
        )
        return result

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(hourly_df: pd.DataFrame, cfg: Any) -> PrecomputeCacheKey:
        """Build a cache key from data + indicator config parameters."""
        n = len(hourly_df)
        close = hourly_df["close"]
        if n > 0:
            first_close = float(close.iloc[0])
            last_close = float(close.iloc[-1])
        else:
            first_close = 0.0
            last_close = 0.0

        fingerprint = hash((n, first_close, last_close))

        return PrecomputeCacheKey(
            n_rows=n,
            data_fingerprint=fingerprint,
            adx_window=int(cfg.adx_window),
            chop_window=int(cfg.chop_window),
            realized_vol_window=int(cfg.realized_vol_window),
            bb_window=int(cfg.bb_window),
            donchian_window=int(cfg.donchian_window),
            atr_window=int(cfg.atr_window),
        )
