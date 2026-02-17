from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


BUCKETS = ("OFF", "ON_HALF", "ON_FULL")


def _to_utc(series_or_index: pd.Series | pd.Index) -> pd.Series:
    return pd.to_datetime(series_or_index, utc=True)


def _bucket_from_state_and_multiplier(state: str | float | int | None, multiplier: float | int | None) -> str:
    s = str(state or "").upper()
    if s in BUCKETS:
        return s

    try:
        m = float(multiplier or 0.0)
    except Exception:
        m = 0.0

    if m <= 0:
        return "OFF"
    if m >= 0.999999:
        return "ON_FULL"
    return "ON_HALF"


def _safe_float(x: float | int | np.floating | None, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(x: int | float | None, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_optional_float(x: float | int | np.floating | None) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


FRED_DECISION_FIELDS = [
    "fred_risk_off_score",
    "fred_penalty_multiplier",
    "fred_comp_vix_z",
    "fred_comp_hy_oas_z",
    "fred_comp_stlfsi_z",
    "fred_comp_nfci_z",
    "fred_vix_level",
    "fred_hy_oas_level",
    "fred_stlfsi_level",
    "fred_nfci_level",
]


def _align_bars_with_macro_bucket(equity_curve: pd.DataFrame, decisions_df: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []

    if equity_curve is None or equity_curve.empty:
        return pd.DataFrame(), ["equity_curve_empty"]

    eq = equity_curve.copy().reset_index().rename(columns={equity_curve.index.name or "index": "timestamp"})
    eq["timestamp"] = _to_utc(eq["timestamp"])
    eq = eq.sort_values("timestamp").reset_index(drop=True)
    eq["merge_key"] = (eq["timestamp"].astype("int64") // 1_000_000_000).astype("int64")

    if decisions_df is not None and not decisions_df.empty:
        dec = decisions_df.copy()
        if "timestamp" not in dec.columns:
            dec = dec.reset_index().rename(columns={dec.index.name or "index": "timestamp"})

        dec["timestamp"] = _to_utc(dec["timestamp"])
        applies_col = "decision_applies_at" if "decision_applies_at" in dec.columns else "timestamp"
        dec["decision_applies_at"] = _to_utc(dec[applies_col])
        dec = dec.sort_values("decision_applies_at").reset_index(drop=True)

        if "macro_state" not in dec.columns:
            warnings.append("macro_state_missing_in_decisions_used_multiplier")

        dec["macro_bucket"] = [
            _bucket_from_state_and_multiplier(state, mult)
            for state, mult in zip(dec.get("macro_state", "OFF"), dec.get("macro_multiplier", 0.0), strict=False)
        ]
        dec["merge_key"] = (dec["decision_applies_at"].astype("int64") // 1_000_000_000).astype("int64")

        keep_cols = ["merge_key", "macro_bucket"] + [c for c in FRED_DECISION_FIELDS if c in dec.columns]
        labels = dec[keep_cols].drop_duplicates(subset=["merge_key"], keep="last").sort_values("merge_key")

        aligned = pd.merge_asof(
            eq.sort_values("merge_key"),
            labels,
            on="merge_key",
            direction="backward",
        )
        aligned["macro_bucket"] = aligned["macro_bucket"].fillna("OFF")
        warnings.append("macro_bucket_from_decisions")
        return aligned, warnings

    # Fallback for legacy callers that do not provide decisions.
    if "macro_state" in eq.columns:
        mult_col = eq.get("macro_multiplier", pd.Series(0.0, index=eq.index))
        eq["macro_bucket"] = [
            _bucket_from_state_and_multiplier(state, mult)
            for state, mult in zip(eq["macro_state"], mult_col, strict=False)
        ]
        warnings.append("decisions_missing_used_equity_columns")
        return eq, warnings

    eq["macro_bucket"] = "OFF"
    warnings.append("decisions_missing_default_off")
    return eq, warnings


def _label_trades_with_bucket(trades_df: pd.DataFrame | None, decisions_df: pd.DataFrame | None) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["macro_bucket", "fee", "notional"])

    tr = trades_df.copy()
    if "ts" not in tr.columns:
        if "timestamp" in tr.columns:
            tr["ts"] = _to_utc(tr["timestamp"])
        else:
            tr = tr.reset_index().rename(columns={tr.index.name or "index": "ts"})
            tr["ts"] = _to_utc(tr["ts"])
    tr["ts"] = tr["ts"].astype("datetime64[ns, UTC]")
    if "ts" in tr.columns and tr["ts"].dtype != "datetime64[ns, UTC]":
        tr["ts"] = tr["ts"].astype("datetime64[ns, UTC]")

    if decisions_df is None or decisions_df.empty:
        out = tr[["ts"]].copy()
        out["macro_bucket"] = "OFF"
        out["fee"] = pd.to_numeric(out.get("fee", 0.0), errors="coerce").fillna(0.0)
        out["notional"] = pd.to_numeric(out.get("notional", 0.0), errors="coerce").fillna(0.0)
        return out

    dec = decisions_df.copy()
    if "decision_applies_at" in dec.columns:
        dec["_decision_ts"] = _to_utc(dec["decision_applies_at"])
    else:
        dec["_decision_ts"] = _to_utc(dec.index.to_series())

    if "macro_state" in dec.columns or "macro_multiplier" in dec.columns:
        dec["macro_bucket"] = [
            _bucket_from_state_and_multiplier(state, mult)
            for state, mult in zip(dec.get("macro_state", "OFF"), dec.get("macro_multiplier", 0.0), strict=False)
        ]
    else:
        dec["macro_bucket"] = "OFF"

    dec["_decision_ts"] = dec["_decision_ts"].astype("datetime64[ns, UTC]")

    dec_map = (
        dec[["_decision_ts", "macro_bucket"]]
        .dropna(subset=["_decision_ts"])
        .drop_duplicates(subset=["_decision_ts"], keep="last")
        .sort_values("_decision_ts")
    )

    out = pd.merge_asof(
        tr[["ts", "fee", "notional"]].sort_values("ts"),
        dec_map,
        left_on="ts",
        right_on="_decision_ts",
        direction="backward",
    )
    out["macro_bucket"] = out["macro_bucket"].fillna("OFF")
    out["fee"] = pd.to_numeric(out.get("fee", 0.0), errors="coerce").fillna(0.0)
    out["notional"] = pd.to_numeric(out.get("notional", 0.0), errors="coerce").fillna(0.0)
    out = out.drop(columns=["_decision_ts"], errors="ignore")
    return out


def _empty_bucket_row(bucket: str) -> dict[str, float | int | str | None]:
    return {
        "bucket": bucket,
        "time_bars": 0,
        "time_hours": 0.0,
        "time_share": 0.0,
        "avg_exposure": 0.0,
        "median_exposure": 0.0,
        "net_pnl": 0.0,
        "fees": 0.0,
        "turnover": 0.0,
        "trade_count": 0,
        "net_return": 0.0,
        "fred_risk_off_score_mean": None,
        "fred_risk_off_score_median": None,
        "fred_vix_level_mean": None,
        "fred_hy_oas_level_mean": None,
        "fred_stlfsi_level_mean": None,
        "fred_nfci_level_mean": None,
        "fred_comp_vix_z_mean": None,
        "fred_comp_hy_oas_z_mean": None,
        "fred_comp_stlfsi_z_mean": None,
        "fred_comp_nfci_z_mean": None,
    }


def compute_macro_bucket_attribution(
    equity_curve: pd.DataFrame,
    decisions_df: pd.DataFrame | None,
    trades_df: pd.DataFrame | None,
    *,
    initial_equity: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compute attribution by macro bucket (OFF/ON_HALF/ON_FULL)."""
    bars, warnings = _align_bars_with_macro_bucket(equity_curve, decisions_df)
    if bars.empty:
        table = pd.DataFrame([_empty_bucket_row(b) for b in BUCKETS])
        report = {
            "buckets": {b: {k: v for k, v in row.items() if k != "bucket"} for b, row in zip(BUCKETS, table.to_dict(orient="records"), strict=False)},
            "warnings": [*warnings, "empty_bars"],
        }
        return report, table

    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["equity"] = pd.to_numeric(bars.get("equity", 0.0), errors="coerce").fillna(0.0)
    bars["exposure"] = pd.to_numeric(bars.get("exposure", 0.0), errors="coerce").fillna(0.0)
    bars["pnl"] = bars["equity"].diff().fillna(0.0)

    dt_hours = bars["timestamp"].diff().dt.total_seconds().fillna(0.0) / 3600.0
    bars["time_hours"] = dt_hours.clip(lower=0.0)

    equity_prev = bars["equity"].shift(1).replace(0.0, np.nan)
    bars["turnover_bar"] = 0.0

    trades_labeled = _label_trades_with_bucket(trades_df, decisions_df)
    if not trades_labeled.empty:
        warnings.append("trade_labels_from_decisions")
    if trades_labeled.empty:
        warnings.append("no_trades_for_macro_bucket_attribution")

    # Attach fees/notional to nearest bar timestamp (as-of backward).
    if not trades_labeled.empty:
        trade_by_bucket = trades_labeled.groupby("macro_bucket", as_index=False).agg(
            fees=("fee", "sum"),
            notional=("notional", "sum"),
            trade_count=("macro_bucket", "size"),
        )
        assigned = int(trade_by_bucket["trade_count"].sum()) if not trade_by_bucket.empty else 0
        if assigned != len(trades_labeled):
            warnings.append("trade_bucket_count_mismatch")
    else:
        trade_by_bucket = pd.DataFrame(columns=["macro_bucket", "fees", "notional", "trade_count"])

    # turnover from notional/equity_prev on execution bars if trade table has timestamps.
    if not trades_labeled.empty and "ts" in trades_labeled.columns:
        exec_flow = trades_labeled[["ts", "notional"]].copy()
        exec_flow["merge_key"] = (exec_flow["ts"].astype("int64") // 1_000_000_000).astype("int64")
        bar_merge = bars[["merge_key", "timestamp", "equity"]].copy()
        bar_merge = bar_merge.sort_values("merge_key")
        flow_aligned = pd.merge_asof(
            exec_flow.sort_values("merge_key"),
            bar_merge,
            on="merge_key",
            direction="backward",
        )
        flow_aligned["equity"] = flow_aligned["equity"].replace(0.0, np.nan)
        flow_aligned["turnover_part"] = (flow_aligned["notional"].abs() / flow_aligned["equity"]).fillna(0.0)
        flow_turn = flow_aligned.groupby("merge_key", as_index=False)["turnover_part"].sum()
        bars = bars.merge(flow_turn, on="merge_key", how="left")
        bars["turnover_bar"] = bars["turnover_part"].fillna(0.0)
        bars = bars.drop(columns=["turnover_part"])

    total_bars = max(1, int(len(bars)))
    if initial_equity is None:
        initial_equity = float(bars["equity"].iloc[0]) if len(bars) else 0.0
    init_eq = max(1e-12, float(initial_equity))

    agg_map: dict[str, tuple[str, str]] = {
        "time_bars": ("macro_bucket", "size"),
        "time_hours": ("time_hours", "sum"),
        "avg_exposure": ("exposure", "mean"),
        "median_exposure": ("exposure", "median"),
        "net_pnl": ("pnl", "sum"),
        "turnover": ("turnover_bar", "sum"),
    }

    if "fred_risk_off_score" in bars.columns:
        bars["fred_risk_off_score"] = pd.to_numeric(bars["fred_risk_off_score"], errors="coerce")
        agg_map["fred_risk_off_score_mean"] = ("fred_risk_off_score", "mean")
        agg_map["fred_risk_off_score_median"] = ("fred_risk_off_score", "median")

    for fred_col in [
        "fred_vix_level",
        "fred_hy_oas_level",
        "fred_stlfsi_level",
        "fred_nfci_level",
        "fred_comp_vix_z",
        "fred_comp_hy_oas_z",
        "fred_comp_stlfsi_z",
        "fred_comp_nfci_z",
    ]:
        if fred_col in bars.columns:
            bars[fred_col] = pd.to_numeric(bars[fred_col], errors="coerce")
            agg_map[f"{fred_col}_mean"] = (fred_col, "mean")

    grouped = bars.groupby("macro_bucket", as_index=False).agg(**agg_map)

    grouped = grouped.merge(trade_by_bucket, left_on="macro_bucket", right_on="macro_bucket", how="left")
    grouped["fees"] = grouped.get("fees", 0.0).fillna(0.0)
    grouped["trade_count"] = grouped.get("trade_count", 0).fillna(0).astype(int)
    grouped["time_share"] = grouped["time_bars"].astype(float) / float(total_bars)
    grouped["net_return"] = grouped["net_pnl"].astype(float) / init_eq

    rows: list[dict[str, float | int | str | None]] = []
    present = set(grouped["macro_bucket"].tolist())
    for bucket in BUCKETS:
        if bucket not in present:
            rows.append(_empty_bucket_row(bucket))
            warnings.append(f"missing_bucket_{bucket}")
            continue

        row = grouped[grouped["macro_bucket"] == bucket].iloc[0]
        rows.append(
            {
                "bucket": bucket,
                "time_bars": _safe_int(row.get("time_bars"), 0),
                "time_hours": _safe_float(row.get("time_hours"), 0.0),
                "time_share": _safe_float(row.get("time_share"), 0.0),
                "avg_exposure": _safe_float(row.get("avg_exposure"), 0.0),
                "median_exposure": _safe_float(row.get("median_exposure"), 0.0),
                "net_pnl": _safe_float(row.get("net_pnl"), 0.0),
                "fees": _safe_float(row.get("fees"), 0.0),
                "turnover": _safe_float(row.get("turnover"), 0.0),
                "trade_count": _safe_int(row.get("trade_count"), 0),
                "net_return": _safe_float(row.get("net_return"), 0.0),
                "fred_risk_off_score_mean": _safe_optional_float(row.get("fred_risk_off_score_mean")),
                "fred_risk_off_score_median": _safe_optional_float(row.get("fred_risk_off_score_median")),
                "fred_vix_level_mean": _safe_optional_float(row.get("fred_vix_level_mean")),
                "fred_hy_oas_level_mean": _safe_optional_float(row.get("fred_hy_oas_level_mean")),
                "fred_stlfsi_level_mean": _safe_optional_float(row.get("fred_stlfsi_level_mean")),
                "fred_nfci_level_mean": _safe_optional_float(row.get("fred_nfci_level_mean")),
                "fred_comp_vix_z_mean": _safe_optional_float(row.get("fred_comp_vix_z_mean")),
                "fred_comp_hy_oas_z_mean": _safe_optional_float(row.get("fred_comp_hy_oas_z_mean")),
                "fred_comp_stlfsi_z_mean": _safe_optional_float(row.get("fred_comp_stlfsi_z_mean")),
                "fred_comp_nfci_z_mean": _safe_optional_float(row.get("fred_comp_nfci_z_mean")),
            }
        )

    table = pd.DataFrame(rows)
    report = {
        "buckets": {
            r["bucket"]: {k: v for k, v in r.items() if k != "bucket"}
            for r in rows
        },
        "warnings": sorted(set(warnings)),
    }
    return report, table
