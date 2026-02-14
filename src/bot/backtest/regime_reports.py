from __future__ import annotations

from typing import Dict

import pandas as pd

from .metrics import compute_metrics


def performance_by_regime(equity_curve: pd.DataFrame, trades_df: pd.DataFrame | None = None) -> Dict[str, dict]:
    if equity_curve.empty or "micro_regime" not in equity_curve.columns:
        return {}

    out: Dict[str, dict] = {}
    for regime, sub in equity_curve.groupby("micro_regime"):
        eq = sub["equity"]
        if eq.empty:
            continue
        tr = pd.DataFrame()
        m = compute_metrics(eq, tr)
        out[str(regime)] = m
    return out


def time_in_regime(equity_curve: pd.DataFrame) -> Dict[str, float]:
    if equity_curve.empty or "micro_regime" not in equity_curve.columns:
        return {}
    counts = equity_curve["micro_regime"].value_counts(dropna=True)
    total = counts.sum()
    if total <= 0:
        return {}
    return {str(k): float(v) / float(total) for k, v in counts.items()}


def regime_switch_count(equity_curve: pd.DataFrame) -> int:
    if equity_curve.empty or "micro_regime" not in equity_curve.columns:
        return 0
    regimes = equity_curve["micro_regime"].fillna("")
    return int((regimes != regimes.shift(1)).sum())


def turnover_at_regime_changes(equity_curve: pd.DataFrame, trades_df: pd.DataFrame) -> Dict[str, float]:
    if trades_df is None or trades_df.empty:
        return {"total_notional": 0.0, "changes_notional": 0.0}
    if "ts" not in trades_df.columns or "micro_regime" not in equity_curve.columns:
        return {"total_notional": float(trades_df["notional"].abs().sum()), "changes_notional": 0.0}

    eq = equity_curve.reset_index()
    if "timestamp" in eq.columns:
        eq = eq.rename(columns={"timestamp": "ts"})
    if "ts" not in eq.columns:
        eq = eq.reset_index().rename(columns={"index": "ts"})
    reg = eq[["ts", "micro_regime"]].copy().sort_values("ts")
    tr = trades_df[["ts", "notional"]].copy().sort_values("ts")

    tr = pd.merge_asof(tr, reg, on="ts", direction="backward")
    tr = tr.sort_values("ts")
    tr["prev_regime"] = tr["micro_regime"].shift(1)
    change = tr["micro_regime"].fillna("") != tr["prev_regime"].fillna("")
    total = float(tr["notional"].abs().sum())
    change_notional = float(tr.loc[change, "notional"].abs().sum())
    return {"total_notional": total, "changes_notional": change_notional}
