from __future__ import annotations

from typing import Dict

import pandas as pd

from .metrics import compute_cagr, compute_sharpe, compute_sortino, max_drawdown, profit_factor_from_pnl, turnover


def _to_utc_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def _to_epoch_seconds(series: pd.Series) -> pd.Series:
    return _to_utc_ts(series).astype("int64") // 1_000_000_000


def _regime_timeline(equity_curve: pd.DataFrame, decisions_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build ts -> micro_regime mapping aligned to execution timestamps."""
    if decisions_df is not None and not decisions_df.empty and "decision_applies_at" in decisions_df.columns and "micro_regime" in decisions_df.columns:
        reg = decisions_df[["decision_applies_at", "micro_regime"]].copy()
        reg = reg.rename(columns={"decision_applies_at": "ts"})
        reg["ts"] = _to_utc_ts(reg["ts"])
        reg["ts_epoch"] = _to_epoch_seconds(reg["ts"])
        reg = reg.sort_values("ts_epoch")
        if not reg.empty:
            return reg

    if equity_curve.empty or "micro_regime" not in equity_curve.columns:
        return pd.DataFrame(columns=["ts", "micro_regime"])

    eq = equity_curve.reset_index()
    if "timestamp" in eq.columns:
        ts_col = "timestamp"
    else:
        ts_col = eq.columns[0]
    reg = eq[[ts_col, "micro_regime"]].rename(columns={ts_col: "ts"}).copy()
    reg["ts"] = _to_utc_ts(reg["ts"])
    reg["ts_epoch"] = _to_epoch_seconds(reg["ts"])
    return reg.sort_values("ts_epoch")


def _align_regime_labels(equity_curve: pd.DataFrame, decisions_df: pd.DataFrame | None = None) -> pd.Series:
    if equity_curve.empty:
        return pd.Series(dtype=object)

    eq = equity_curve.reset_index()
    if "timestamp" in eq.columns:
        ts_col = "timestamp"
    else:
        ts_col = eq.columns[0]
    eq_ts = eq[[ts_col]].rename(columns={ts_col: "ts"}).copy()
    eq_ts["ts"] = _to_utc_ts(eq_ts["ts"])
    eq_ts["ts_epoch"] = _to_epoch_seconds(eq_ts["ts"])

    reg = _regime_timeline(equity_curve, decisions_df)
    if reg.empty:
        if "micro_regime" in equity_curve.columns:
            return equity_curve["micro_regime"].astype(str)
        return pd.Series(["UNKNOWN"] * len(equity_curve), index=equity_curve.index, dtype=object)

    out = pd.merge_asof(eq_ts.sort_values("ts_epoch"), reg[["ts_epoch", "micro_regime"]].sort_values("ts_epoch"), on="ts_epoch", direction="backward")
    labels = out["micro_regime"].fillna("UNKNOWN").astype(str)
    labels.index = equity_curve.index
    return labels


def _attach_regime_to_trades(
    trades_df: pd.DataFrame,
    equity_curve: pd.DataFrame,
    decisions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if trades_df is None or trades_df.empty or "ts" not in trades_df.columns:
        return pd.DataFrame(columns=list(trades_df.columns if trades_df is not None else []) + ["micro_regime"])

    tr = trades_df.copy()
    # prefer regime labels aligned from decisions/equity timeline; drop existing trade labels to avoid merge collisions
    if "micro_regime" in tr.columns:
        tr = tr.drop(columns=["micro_regime"])
    tr["ts"] = _to_utc_ts(tr["ts"])
    tr["ts_epoch"] = _to_epoch_seconds(tr["ts"])

    reg = _regime_timeline(equity_curve, decisions_df)
    if reg.empty:
        tr["micro_regime"] = "UNKNOWN"
        return tr

    merged = pd.merge_asof(
        tr.sort_values("ts_epoch"),
        reg[["ts_epoch", "micro_regime"]].sort_values("ts_epoch"),
        on="ts_epoch",
        direction="backward",
    )
    merged["micro_regime"] = merged["micro_regime"].fillna("UNKNOWN").astype(str)
    return merged


def performance_by_regime(
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame | None = None,
    decisions_df: pd.DataFrame | None = None,
    freq_per_year: int = 8760,
) -> Dict[str, dict]:
    if equity_curve.empty:
        return {}

    if "equity" not in equity_curve.columns:
        return {}

    eq = equity_curve.copy().sort_index()
    labels = _align_regime_labels(eq, decisions_df)

    ret = eq["equity"].pct_change().replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    pnl = eq["equity"].diff().replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    tr_reg = _attach_regime_to_trades(trades_df if trades_df is not None else pd.DataFrame(), eq, decisions_df)

    out: Dict[str, dict] = {}
    start_equity = float(eq["equity"].iloc[0]) if len(eq) else 1.0

    for regime in labels.dropna().astype(str).unique():
        mask = labels.astype(str) == str(regime)
        if not bool(mask.any()):
            continue

        ret_r = ret[mask]
        pnl_r = pnl[mask]

        # regime-only pseudo equity path from restricted returns
        eq_r = (1.0 + ret_r).cumprod()

        pf = profit_factor_from_pnl(pnl_r)
        tr_slice = tr_reg[tr_reg["micro_regime"].astype(str) == str(regime)] if not tr_reg.empty else pd.DataFrame()

        sharpe = compute_sharpe(ret_r, periods_per_year=freq_per_year) if len(ret_r.dropna()) >= 2 else None
        sortino = compute_sortino(ret_r, periods_per_year=freq_per_year) if len(ret_r.dropna()) >= 2 else None
        cagr = compute_cagr(eq_r, periods_per_year=freq_per_year) if len(eq_r.dropna()) >= 2 else None

        out[str(regime)] = {
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown(eq_r) if len(eq_r.dropna()) >= 1 else None,
            "profit_factor": pf["profit_factor"],
            "profit_factor_is_infinite": bool(pf["profit_factor_is_infinite"]),
            "profit_factor_gross_profit": float(pf["profit_factor_gross_profit"]),
            "profit_factor_gross_loss": float(pf["profit_factor_gross_loss"]),
            "turnover": turnover(tr_slice, start_equity),
        }

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

    # normalize to epoch seconds to avoid datetime precision mismatch issues in merge_asof
    reg["ts"] = pd.to_datetime(reg["ts"], utc=True).astype("int64") // 1_000_000_000
    tr["ts"] = pd.to_datetime(tr["ts"], utc=True).astype("int64") // 1_000_000_000

    tr = pd.merge_asof(tr.sort_values("ts"), reg.sort_values("ts"), on="ts", direction="backward")
    tr = tr.sort_values("ts")
    tr["prev_regime"] = tr["micro_regime"].shift(1)
    change = tr["micro_regime"].fillna("") != tr["prev_regime"].fillna("")
    total = float(tr["notional"].abs().sum())
    change_notional = float(tr.loc[change, "notional"].abs().sum())
    return {"total_notional": total, "changes_notional": change_notional}
