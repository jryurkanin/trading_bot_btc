from __future__ import annotations

from pathlib import Path

import pandas as pd

from bot.analysis.pnl_decomposition import run_pnl_decomposition


def test_run_pnl_decomposition_handles_empty_trades_csv(tmp_path: Path):
    trades = tmp_path / "trades.csv"
    equity = tmp_path / "equity_curve.csv"

    # Empty file (0 bytes) should be treated as no trades.
    trades.write_text("", encoding="utf-8")

    eq = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "equity": [10_000.0, 10_010.0, 10_020.0],
        }
    )
    eq.to_csv(equity, index=False)

    out = run_pnl_decomposition(trades, equity, tmp_path)
    assert int(out["trade_count"]) == 0
    assert float(out["total_fees"]) == 0.0
    assert (tmp_path / "execution_quality.json").exists()
