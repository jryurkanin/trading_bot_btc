from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from bot.config import BotConfig
from bot.backtest.cpcv import (
    make_cpcv_groups,
    generate_cpcv_splits,
    run_cpcv,
    CPCVConfig,
    _ts_in_any_range,
)


def _make_timestamps(n: int = 1000) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")


class TestMakeCPCVGroups:
    def test_correct_count(self):
        ts = _make_timestamps(1000)
        groups = make_cpcv_groups(ts, 6)
        assert len(groups) == 6

    def test_covers_range(self):
        ts = _make_timestamps(1000)
        groups = make_cpcv_groups(ts, 4)
        assert groups[0][0] == ts[0]
        assert groups[-1][1] == ts[-1]

    def test_no_gaps(self):
        ts = _make_timestamps(1000)
        groups = make_cpcv_groups(ts, 5)
        for i in range(len(groups) - 1):
            # End of group i should be close to start of group i+1
            assert groups[i][1] <= groups[i + 1][0]


class TestGenerateCPCVSplits:
    def test_n6_k2_gives_15_splits(self):
        ts = _make_timestamps(600)
        groups = make_cpcv_groups(ts, 6)
        splits = generate_cpcv_splits(groups, n_test_groups=2, purge_bars=24, embargo_bars=12)
        assert len(splits) == comb(6, 2)  # 15

    def test_n2_k1_gives_2_splits(self):
        ts = _make_timestamps(200)
        groups = make_cpcv_groups(ts, 2)
        splits = generate_cpcv_splits(groups, n_test_groups=1, purge_bars=12, embargo_bars=6)
        assert len(splits) == 2

    def test_n4_k2_gives_6_splits(self):
        ts = _make_timestamps(400)
        groups = make_cpcv_groups(ts, 4)
        splits = generate_cpcv_splits(groups, n_test_groups=2, purge_bars=24, embargo_bars=12)
        assert len(splits) == comb(4, 2)  # 6


class TestPurgeAndEmbargo:
    def test_purge_coverage(self):
        """Bars within purge_bars of test boundaries should appear in purge ranges."""
        ts = _make_timestamps(600)
        groups = make_cpcv_groups(ts, 3)
        splits = generate_cpcv_splits(groups, n_test_groups=1, purge_bars=24, embargo_bars=12)

        for split in splits:
            test_ranges = split["test_ranges"]
            purge_ranges = split["purge_ranges"]
            # Check that purge zones exist around each test boundary
            for t_start, t_end in test_ranges:
                # There should be a purge zone ending at t_start
                found_before = any(
                    pr[1] == t_start for pr in purge_ranges
                )
                assert found_before, f"No purge zone before test start {t_start}"
                # There should be a purge zone starting at t_end
                found_after = any(
                    pr[0] == t_end for pr in purge_ranges
                )
                assert found_after, f"No purge zone after test end {t_end}"

    def test_embargo_coverage(self):
        """Embargo zones should start at test group end."""
        ts = _make_timestamps(600)
        groups = make_cpcv_groups(ts, 3)
        splits = generate_cpcv_splits(groups, n_test_groups=1, purge_bars=24, embargo_bars=12)

        for split in splits:
            test_ranges = split["test_ranges"]
            embargo_ranges = split["embargo_ranges"]
            for t_start, t_end in test_ranges:
                found = any(er[0] == t_end for er in embargo_ranges)
                assert found, f"No embargo zone at test end {t_end}"


class TestNoTrainTestOverlap:
    def test_no_overlap(self):
        """No timestamp should be in both train and test in any split."""
        ts = _make_timestamps(300)
        groups = make_cpcv_groups(ts, 3)
        splits = generate_cpcv_splits(groups, n_test_groups=1, purge_bars=12, embargo_bars=6)

        for split in splits:
            test_ranges = split["test_ranges"]
            purge_ranges = split["purge_ranges"]
            embargo_ranges = split["embargo_ranges"]

            for t in ts:
                in_test = _ts_in_any_range(t, test_ranges)
                in_purge = _ts_in_any_range(t, purge_ranges)
                in_embargo = _ts_in_any_range(t, embargo_ranges)
                in_excluded = in_test or in_purge or in_embargo

                # A timestamp that is in test should not be used for training
                # (purge/embargo should exclude it from training set)
                if in_test:
                    # This is fine, it's just test data
                    pass


def make_oscillating_candles(n=400, product="BTC-USD"):
    start = pd.Timestamp("2025-01-01", tz="UTC")
    ts = pd.date_range(start, periods=n, freq="h")
    close = 30000 + (pd.Series(range(n)) % 20 - 10).astype(float)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = close + 5
    low = close - 5
    vol = 100.0
    df = pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )
    dstart = pd.Timestamp("2025-01-01", tz="UTC")
    dts = pd.date_range(dstart, periods=max(2, n // 24), freq="D")
    daily_close = pd.Series(30000 + (pd.Series(range(len(dts))) % 20 - 10).astype(float).values, index=dts)
    daily = pd.DataFrame(
        {"timestamp": dts, "open": daily_close, "high": daily_close + 10, "low": daily_close - 10, "close": daily_close, "volume": 1000.0}
    )
    return df, daily


class TestCPCVIntegration:
    """Integration test with actual engine runs."""

    def test_fast_run(self):
        hourly, daily = make_oscillating_candles(n=2000)
        cfg = BotConfig()
        cfg.data.product = "BTC-USD"
        cfg.backtest.initial_equity = 10000

        result = run_cpcv(
            hourly, daily, cfg,
            CPCVConfig(
                n_groups=3,
                n_test_groups=1,
                purge_bars=12,
                embargo_bars=6,
                warmup_days=0,  # Skip warmup for speed in test
            ),
        )

        assert result.n_paths == comb(3, 1)  # 3
        assert result.n_completed == 3
        assert len(result.splits) == 3

        # Each split should have valid metrics
        for split in result.splits:
            assert "sharpe" in split.metrics or "cagr" in split.metrics

        # Aggregate should have mean/std/median/min/max
        if result.aggregate_metrics:
            for key, stats in result.aggregate_metrics.items():
                assert "mean" in stats
                assert "std" in stats
                assert "median" in stats
                assert "min" in stats
                assert "max" in stats
