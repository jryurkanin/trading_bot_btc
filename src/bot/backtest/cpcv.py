"""Combinatorial Purged Cross-Validation (CPCV) for backtest evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import BotConfig
from ..system_log import get_system_logger
from .engine import BacktestEngine

logger = get_system_logger("backtest.cpcv")


@dataclass
class CPCVConfig:
    n_groups: int = 6
    n_test_groups: int = 2  # C(6,2) = 15 paths
    purge_bars: int = 24  # 1 day of hourly bars
    embargo_bars: int = 12  # 12 hours
    warmup_days: int = 400


@dataclass
class CPCVSplitResult:
    split_id: int
    test_group_indices: Tuple[int, ...]
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_bars: int
    test_bars: int
    purged_bars: int
    embargoed_bars: int
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPCVResult:
    n_groups: int
    n_test_groups: int
    n_paths: int
    n_completed: int
    splits: List[CPCVSplitResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)


def make_cpcv_groups(
    timestamps: pd.DatetimeIndex,
    n_groups: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a timestamp range into N equal-sized groups.

    Returns a list of (start, end) tuples.  Boundaries are inclusive-start,
    exclusive-end.
    """
    if len(timestamps) == 0:
        return []
    ts_sorted = timestamps.sort_values()
    n = len(ts_sorted)
    boundaries = np.linspace(0, n, n_groups + 1, dtype=int)
    groups = []
    for i in range(n_groups):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1] - 1
        if end_idx < start_idx:
            end_idx = start_idx
        groups.append((ts_sorted[start_idx], ts_sorted[end_idx]))
    return groups


def generate_cpcv_splits(
    groups: List[Tuple[pd.Timestamp, pd.Timestamp]],
    n_test_groups: int,
    purge_bars: int,
    embargo_bars: int,
    bar_duration_hours: int = 1,
) -> List[Dict[str, Any]]:
    """Generate all C(N, k) combinatorial splits with purge and embargo zones.

    Each split dict contains:
    - ``test_group_indices``: tuple of group indices used for testing
    - ``test_ranges``: list of (start, end) for test groups
    - ``purge_ranges``: list of (start, end) for purged zones
    - ``embargo_ranges``: list of (start, end) for embargo zones
    """
    n = len(groups)
    bar_td = timedelta(hours=bar_duration_hours)
    purge_td = bar_td * purge_bars
    embargo_td = bar_td * embargo_bars

    splits = []
    for combo in combinations(range(n), n_test_groups):
        test_ranges = [(groups[i][0], groups[i][1]) for i in combo]
        purge_ranges = []
        embargo_ranges = []

        for i in combo:
            g_start, g_end = groups[i]
            # Purge zone before test group
            purge_start = g_start - purge_td
            purge_ranges.append((purge_start, g_start))
            # Purge zone after test group
            purge_end = g_end + purge_td
            purge_ranges.append((g_end, purge_end))
            # Embargo zone after test group end
            embargo_start = g_end
            embargo_end = g_end + embargo_td
            embargo_ranges.append((embargo_start, embargo_end))

        splits.append({
            "test_group_indices": combo,
            "test_ranges": test_ranges,
            "purge_ranges": purge_ranges,
            "embargo_ranges": embargo_ranges,
        })

    return splits


def _ts_in_any_range(
    ts: pd.Timestamp,
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    """Check if a timestamp falls within any of the given ranges."""
    for start, end in ranges:
        if start <= ts <= end:
            return True
    return False


def _clone_cfg(cfg: BotConfig) -> BotConfig:
    if hasattr(cfg, "model_dump") and hasattr(BotConfig, "model_validate"):
        return BotConfig.model_validate(cfg.model_dump())
    if hasattr(cfg, "dict"):
        return BotConfig.parse_obj(cfg.dict())
    return BotConfig.parse_obj(dict(cfg.__dict__))


def run_cpcv(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: BotConfig,
    cpcv_config: CPCVConfig | None = None,
) -> CPCVResult:
    """Run combinatorial purged cross-validation.

    Evaluates every data point exactly ``n_test_groups`` times across all
    C(N, k) paths, eliminating selection bias from a single train/test split.
    """
    if cpcv_config is None:
        cpcv_config = CPCVConfig()

    h = hourly.copy()
    d = daily.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)

    timestamps = pd.DatetimeIndex(h["timestamp"])
    groups = make_cpcv_groups(timestamps, cpcv_config.n_groups)
    splits = generate_cpcv_splits(
        groups,
        cpcv_config.n_test_groups,
        cpcv_config.purge_bars,
        cpcv_config.embargo_bars,
    )

    warmup_td = timedelta(days=cpcv_config.warmup_days)
    n_paths = len(splits)

    logger.info(
        "cpcv_start n_groups=%d n_test_groups=%d n_paths=%d purge_bars=%d embargo_bars=%d",
        cpcv_config.n_groups,
        cpcv_config.n_test_groups,
        n_paths,
        cpcv_config.purge_bars,
        cpcv_config.embargo_bars,
    )

    split_results: List[CPCVSplitResult] = []
    all_metrics_keys: set = set()

    for split_id, split in enumerate(splits):
        test_ranges = split["test_ranges"]
        purge_ranges = split["purge_ranges"]
        embargo_ranges = split["embargo_ranges"]
        test_indices = split["test_group_indices"]

        # Overall test boundaries for reporting
        test_start = min(r[0] for r in test_ranges)
        test_end = max(r[1] for r in test_ranges)

        # Count excluded bars
        excluded_purge = 0
        excluded_embargo = 0
        for _, row_ts in h["timestamp"].items():
            if _ts_in_any_range(row_ts, purge_ranges):
                excluded_purge += 1
            if _ts_in_any_range(row_ts, embargo_ranges):
                excluded_embargo += 1

        # Run engine on each contiguous test segment
        combined_metrics: Dict[str, List[float]] = {}

        for t_start, t_end in test_ranges:
            # Provide full history for feature warmup (same as walkforward.py)
            prefetch_start = t_start - warmup_td
            hourly_segment = h[
                (h["timestamp"] >= prefetch_start) & (h["timestamp"] <= t_end)
            ]
            daily_segment = d[
                (d["timestamp"] >= prefetch_start) & (d["timestamp"] <= t_end)
            ]

            if hourly_segment.empty or daily_segment.empty:
                logger.debug(
                    "cpcv_skip_segment split=%d test_start=%s test_end=%s reason=empty",
                    split_id, t_start, t_end,
                )
                continue

            run_cfg = _clone_cfg(cfg)

            maker = run_cfg.backtest.maker_bps / 10_000.0
            taker = run_cfg.backtest.taker_bps / 10_000.0

            engine = BacktestEngine(
                product=run_cfg.data.product,
                hourly_candles=hourly_segment,
                daily_candles=daily_segment,
                start=t_start.to_pydatetime(),
                end=t_end.to_pydatetime(),
                config=run_cfg.backtest,
                fees=(maker, taker),
                slippage_bps=run_cfg.backtest.slippage_bps,
                use_spread_slippage=run_cfg.backtest.use_spread_slippage,
                regime_config=run_cfg.regime,
                risk_config=run_cfg.risk,
                execution_config=run_cfg.execution,
                fred_config=run_cfg.fred,
            )
            result = engine.run()

            for key, val in result.metrics.items():
                if isinstance(val, (int, float)) and val is not None:
                    combined_metrics.setdefault(key, []).append(float(val))

        # Average metrics across test segments in this split
        avg_metrics: Dict[str, Any] = {}
        for key, vals in combined_metrics.items():
            if vals:
                avg_metrics[key] = float(np.mean(vals))
                all_metrics_keys.add(key)

        test_bars = sum(
            len(h[(h["timestamp"] >= r[0]) & (h["timestamp"] <= r[1])])
            for r in test_ranges
        )
        train_bars = len(h) - test_bars - excluded_purge - excluded_embargo

        split_results.append(CPCVSplitResult(
            split_id=split_id,
            test_group_indices=test_indices,
            test_start=test_start,
            test_end=test_end,
            train_bars=max(0, train_bars),
            test_bars=test_bars,
            purged_bars=excluded_purge,
            embargoed_bars=excluded_embargo,
            metrics=avg_metrics,
        ))

        logger.info(
            "cpcv_split_complete split=%d/%d test_groups=%s sharpe=%.4f cagr=%.4f",
            split_id + 1,
            n_paths,
            test_indices,
            avg_metrics.get("sharpe", 0.0),
            avg_metrics.get("cagr", 0.0),
        )

    # Aggregate across all splits
    aggregate: Dict[str, Any] = {}
    for key in sorted(all_metrics_keys):
        vals = [s.metrics[key] for s in split_results if key in s.metrics]
        if vals:
            arr = np.array(vals, dtype=float)
            aggregate[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    logger.info(
        "cpcv_complete n_paths=%d n_completed=%d",
        n_paths,
        len(split_results),
    )

    return CPCVResult(
        n_groups=cpcv_config.n_groups,
        n_test_groups=cpcv_config.n_test_groups,
        n_paths=n_paths,
        n_completed=len(split_results),
        splits=split_results,
        aggregate_metrics=aggregate,
    )
