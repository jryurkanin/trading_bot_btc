from datetime import datetime, timezone

from bot.backtest.frontier_runtime import (
    build_checkpoint_fingerprint,
    checkpoint_fingerprint_mismatches,
    build_filter_rejections_payload,
)


def test_checkpoint_fingerprint_changes_with_grid_and_windows():
    windows_a = [
        {
            "name": "train",
            "start": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end": datetime(2025, 6, 30, 23, tzinfo=timezone.utc),
        },
        {
            "name": "val",
            "start": datetime(2025, 7, 1, tzinfo=timezone.utc),
            "end": datetime(2025, 9, 30, 23, tzinfo=timezone.utc),
        },
    ]
    windows_b = [
        {
            "name": "train",
            "start": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end": datetime(2025, 5, 31, 23, tzinfo=timezone.utc),
        },
        {
            "name": "val",
            "start": datetime(2025, 6, 1, tzinfo=timezone.utc),
            "end": datetime(2025, 9, 30, 23, tzinfo=timezone.utc),
        },
    ]

    fp1 = build_checkpoint_fingerprint(
        "macro_gate_benchmark",
        [{"target_ann_vol": 0.3}, {"target_ann_vol": 0.4}],
        windows_a,
    )
    fp2 = build_checkpoint_fingerprint(
        "macro_gate_benchmark",
        [{"target_ann_vol": 0.3}, {"target_ann_vol": 0.5}],
        windows_a,
    )
    fp3 = build_checkpoint_fingerprint(
        "macro_gate_benchmark",
        [{"target_ann_vol": 0.3}, {"target_ann_vol": 0.4}],
        windows_b,
    )

    assert fp1["strategy"] == "macro_gate_benchmark"
    assert fp1["grid_hash"] != fp2["grid_hash"]
    assert fp1["window_hash"] != fp3["window_hash"]


def test_checkpoint_mismatch_detects_missing_and_changed_fields():
    expected = {
        "strategy": "macro_only_v2",
        "grid_hash": "abc",
        "window_hash": "def",
    }

    mismatch = checkpoint_fingerprint_mismatches(
        {
            "strategy": "macro_only_v2",
            "grid_hash": "zzz",
            # window_hash intentionally missing
        },
        expected,
    )

    assert "grid_hash" in mismatch
    assert mismatch["grid_hash"]["reason"] == "mismatch"
    assert "window_hash" in mismatch
    assert mismatch["window_hash"]["reason"] == "missing"


def test_filter_rejections_payload_contract_shape():
    payload = build_filter_rejections_payload(
        run_id="run_abc",
        strategy="v5_adaptive",
        total_param_sets=12,
        rejection_counts={
            "accepted": 3,
            "drawdown_limit": 5,
            "turnover_limit": 4,
        },
    )

    assert payload == {
        "run_id": "run_abc",
        "strategy": "v5_adaptive",
        "total_param_sets": 12,
        "accepted_param_sets": 3,
        "rejections": {
            "drawdown_limit": 5,
            "turnover_limit": 4,
        },
    }
