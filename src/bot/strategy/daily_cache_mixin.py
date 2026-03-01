"""Shared daily-cache helpers used by all macro-gated strategies.

Extracts _to_timestamp_col, _day_cache_key, _ensure_daily_index_cache,
_closed_daily_cached, _latest_daily_ts_cached, and _latest_daily_feature
into a mixin so that bug fixes apply in one place.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_MAX_DAILY_CACHE_ENTRIES = 30


class DailyCacheMixin:
    """Mixin providing daily-bar cache helpers for strategies."""

    def _init_daily_cache(self) -> None:
        self._daily_ts_cache: dict[int, pd.DataFrame] = {}
        self._daily_last_ts_cache: dict[int, pd.Timestamp | None] = {}
        self._daily_index_cache_sig: tuple[int, int] | None = None
        self._daily_index_values: np.ndarray | None = None

    def _clear_daily_cache(self) -> None:
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()
        self._daily_index_cache_sig = None
        self._daily_index_values = None

    @staticmethod
    def _to_timestamp_col(df: pd.DataFrame) -> pd.Series:
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"], utc=True)
        return pd.to_datetime(df.index, utc=True)

    @staticmethod
    def _day_cache_key(decision_ts: pd.Timestamp) -> int:
        ts = pd.Timestamp(decision_ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.floor("D").value)

    def _ensure_daily_index_cache(self, daily_df: pd.DataFrame) -> None:
        if daily_df is None or daily_df.empty:
            self._daily_index_cache_sig = None
            self._daily_index_values = None
            return

        ts = self._to_timestamp_col(daily_df)
        if len(ts):
            ts_last = ts.iloc[-1] if isinstance(ts, pd.Series) else ts[-1]
            last_val = int(pd.Timestamp(ts_last).value)
        else:
            last_val = 0
        sig = (int(len(daily_df)), last_val)
        if self._daily_index_cache_sig == sig and self._daily_index_values is not None:
            return

        self._daily_index_values = ts.to_numpy(dtype="datetime64[ns]")
        self._daily_index_cache_sig = sig
        self._daily_ts_cache.clear()
        self._daily_last_ts_cache.clear()

    def _closed_daily_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.DataFrame:
        if daily_df is None or daily_df.empty:
            return pd.DataFrame(columns=daily_df.columns if daily_df is not None else [])

        self._ensure_daily_index_cache(daily_df)
        key = self._day_cache_key(decision_ts)
        if key in self._daily_ts_cache:
            return self._daily_ts_cache[key]

        cutoff = pd.Timestamp(decision_ts)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        closed_cutoff = cutoff.floor("D")

        idx_vals = self._daily_index_values
        if idx_vals is None:
            d = pd.DataFrame(columns=daily_df.columns)
        else:
            pos = int(np.searchsorted(idx_vals, closed_cutoff.to_numpy(), side="left"))
            d = daily_df.iloc[:pos]

        # Evict old entries to prevent unbounded cache growth (issue 6.3)
        if len(self._daily_ts_cache) >= _MAX_DAILY_CACHE_ENTRIES:
            oldest_key = next(iter(self._daily_ts_cache))
            del self._daily_ts_cache[oldest_key]

        self._daily_ts_cache[key] = d
        return d

    def _latest_daily_ts_cached(self, daily_df: pd.DataFrame, decision_ts: pd.Timestamp) -> pd.Timestamp | None:
        key = self._day_cache_key(decision_ts)
        if key in self._daily_last_ts_cache:
            return self._daily_last_ts_cache[key]

        closed = self._closed_daily_cached(daily_df, decision_ts)
        if closed.empty:
            self._daily_last_ts_cache[key] = None
            return None

        ts = self._to_timestamp_col(closed)
        ts_last = ts.iloc[-1] if isinstance(ts, pd.Series) else ts[-1]
        last = pd.to_datetime(ts_last, utc=True)

        if len(self._daily_last_ts_cache) >= _MAX_DAILY_CACHE_ENTRIES:
            oldest_key = next(iter(self._daily_last_ts_cache))
            del self._daily_last_ts_cache[oldest_key]

        self._daily_last_ts_cache[key] = last
        return last

    @staticmethod
    def _latest_daily_feature(daily_df: pd.DataFrame, column: str, default: float = 0.0) -> float:
        """Read the latest numeric value of a column from the daily DataFrame."""
        if daily_df is None or daily_df.empty or column not in daily_df.columns:
            return float(default)
        series = pd.to_numeric(daily_df[column], errors="coerce").dropna()
        if series.empty:
            return float(default)
        return float(series.iloc[-1])
