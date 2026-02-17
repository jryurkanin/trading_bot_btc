from __future__ import annotations

import pandas as pd

from bot.backtest.macro_attribution import compute_macro_bucket_attribution


def test_macro_bucket_attribution_has_all_buckets_and_expected_fields():
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="h", tz="UTC")

    equity = pd.DataFrame(
        {
            "timestamp": ts,
            "equity": [10_000, 10_020, 10_030, 10_015, 10_060, 10_090],
            "exposure": [0.0, 0.1, 0.5, 0.45, 1.0, 0.9],
        }
    ).set_index("timestamp")

    decisions = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[2], ts[4]],
            "decision_applies_at": [ts[0], ts[2], ts[4]],
            "macro_state": ["OFF", "ON_HALF", "ON_FULL"],
            "macro_multiplier": [0.0, 0.5, 1.0],
        }
    ).set_index("timestamp")

    trades = pd.DataFrame(
        {
            "ts": [ts[0] + pd.Timedelta(minutes=30), ts[2] + pd.Timedelta(minutes=30), ts[4] + pd.Timedelta(minutes=30)],
            "fee": [1.0, 2.0, 3.0],
            "notional": [100.0, 200.0, 300.0],
        }
    )

    report, table = compute_macro_bucket_attribution(
        equity,
        decisions,
        trades,
        initial_equity=10_000.0,
    )

    buckets = report.get("buckets", {})
    assert set(buckets.keys()) == {"OFF", "ON_HALF", "ON_FULL"}

    assert set(table["bucket"].tolist()) == {"OFF", "ON_HALF", "ON_FULL"}
    assert all(c in table.columns for c in ["net_pnl", "fees", "turnover", "time_share", "avg_exposure", "trade_count"])

    total_share = float(table["time_share"].sum())
    assert 0.99 <= total_share <= 1.01

    # One trade per bucket in this synthetic setup.
    counts = {row["bucket"]: int(row["trade_count"]) for _, row in table.iterrows()}
    assert counts["OFF"] == 1
    assert counts["ON_HALF"] == 1
    assert counts["ON_FULL"] == 1


def test_macro_bucket_attribution_fred_aggregates_from_decisions():
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h", tz="UTC")

    equity = pd.DataFrame(
        {
            "timestamp": ts,
            "equity": [10_000, 10_010, 10_030, 10_040],
            "exposure": [0.0, 0.1, 0.8, 0.9],
        }
    ).set_index("timestamp")

    decisions = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[2]],
            "decision_applies_at": [ts[0], ts[2]],
            "macro_state": ["OFF", "ON_FULL"],
            "macro_multiplier": [0.0, 1.0],
            "fred_risk_off_score": [0.2, 0.8],
            "fred_comp_vix_z": [0.1, 1.1],
            "fred_vix_level": [15.0, 30.0],
        }
    ).set_index("timestamp")

    report, table = compute_macro_bucket_attribution(
        equity,
        decisions,
        trades_df=None,
        initial_equity=10_000.0,
    )

    assert "macro_bucket_from_decisions" in report.get("warnings", [])

    row_off = table[table["bucket"] == "OFF"].iloc[0]
    row_full = table[table["bucket"] == "ON_FULL"].iloc[0]

    assert abs(float(row_off["fred_risk_off_score_mean"]) - 0.2) < 1e-9
    assert abs(float(row_full["fred_risk_off_score_mean"]) - 0.8) < 1e-9
    assert abs(float(row_off["fred_vix_level_mean"]) - 15.0) < 1e-9
    assert abs(float(row_full["fred_vix_level_mean"]) - 30.0) < 1e-9
