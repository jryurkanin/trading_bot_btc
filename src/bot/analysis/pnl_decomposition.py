from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..backtest.reporting import write_strict_json, dumps_strict_json


@dataclass
class GuardrailResult:
    warnings: list[str]
    errors: list[str]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _side_slippage_bps(df: pd.DataFrame, side: str) -> pd.Series:
    sub = df[df["side"].astype(str).str.upper() == side.upper()].copy()
    if sub.empty:
        return pd.Series(dtype=float)

    if "mark_price" in sub.columns and "price" in sub.columns:
        mark = sub["mark_price"].astype(float)
        px = sub["price"].astype(float)
        if side.upper() == "BUY":
            bps = (px / mark - 1.0) * 10_000.0
        else:
            bps = (mark / px - 1.0) * 10_000.0
        bps = bps.replace([np.inf, -np.inf], np.nan).dropna()
        return bps

    return pd.Series(dtype=float)


def compute_pnl_decomposition(
    trades_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    min_trade_notional_usd: float = 50.0,
    max_allowed_slippage_bps: float = 50.0,
    ci_mode: bool = False,
) -> dict[str, Any]:
    if equity_curve_df.empty:
        raise ValueError("equity_curve_df is empty")

    eq = equity_curve_df.copy()
    if "equity" not in eq.columns:
        raise ValueError("equity_curve_df must include equity column")

    start_equity = _safe_float(eq["equity"].iloc[0], 0.0)
    end_equity = _safe_float(eq["equity"].iloc[-1], start_equity)
    net_pnl = end_equity - start_equity
    net_return = (end_equity / start_equity - 1.0) if start_equity > 0 else 0.0

    tr = trades_df.copy() if trades_df is not None else pd.DataFrame()
    if tr.empty:
        decomp = {
            "net_return": float(net_return),
            "gross_return": float(net_return),
            "net_pnl": float(net_pnl),
            "gross_pnl": float(net_pnl),
            "total_fees": 0.0,
            "total_slippage_cost": 0.0,
            "buy_slippage_bps": {"avg": 0.0, "median": 0.0},
            "sell_slippage_bps": {"avg": 0.0, "median": 0.0},
            "notional_distribution": {
                "pct_lt_10": 0.0,
                "pct_lt_25": 0.0,
                "pct_lt_50": 0.0,
                "median": 0.0,
                "mean": 0.0,
            },
            "trade_count": 0,
            "guardrails": {"warnings": [], "errors": []},
        }
        return decomp

    if "notional" not in tr.columns:
        tr["notional"] = 0.0
    tr["notional"] = tr["notional"].astype(float).abs()

    if "fee" not in tr.columns:
        tr["fee"] = 0.0
    tr["fee"] = tr["fee"].astype(float)

    # If slippage_cost is missing, estimate from mark/price where possible.
    if "slippage_cost" not in tr.columns:
        if all(c in tr.columns for c in ["mark_price", "price", "btc_qty", "side"]):
            qty = tr["btc_qty"].astype(float).abs()
            mark = tr["mark_price"].astype(float)
            px = tr["price"].astype(float)
            side = tr["side"].astype(str).str.upper()
            buy_cost = ((px - mark) * qty).clip(lower=0.0)
            sell_cost = ((mark - px) * qty).clip(lower=0.0)
            tr["slippage_cost"] = np.where(side == "BUY", buy_cost, sell_cost)
        else:
            tr["slippage_cost"] = 0.0
    tr["slippage_cost"] = tr["slippage_cost"].astype(float)

    total_fees = float(tr["fee"].sum())
    total_slippage_cost = float(tr["slippage_cost"].sum())

    gross_pnl = net_pnl + total_fees + total_slippage_cost
    gross_return = (gross_pnl / start_equity) if start_equity > 0 else 0.0

    buy_bps = _side_slippage_bps(tr, "BUY")
    sell_bps = _side_slippage_bps(tr, "SELL")

    notionals = tr["notional"]
    n = max(1, len(notionals))
    notional_distribution = {
        "pct_lt_10": float((notionals < 10).sum() / n),
        "pct_lt_25": float((notionals < 25).sum() / n),
        "pct_lt_50": float((notionals < 50).sum() / n),
        "median": float(notionals.median()),
        "mean": float(notionals.mean()),
    }

    guardrail = GuardrailResult(warnings=[], errors=[])

    if notional_distribution["median"] < float(min_trade_notional_usd):
        msg = f"median trade notional ${notional_distribution['median']:.2f} < min_trade_notional_usd ${min_trade_notional_usd:.2f}"
        (guardrail.errors if ci_mode else guardrail.warnings).append(msg)

    avg_buy = float(buy_bps.mean()) if len(buy_bps) else 0.0
    avg_sell = float(sell_bps.mean()) if len(sell_bps) else 0.0
    if avg_buy > max_allowed_slippage_bps:
        msg = f"avg BUY slippage {avg_buy:.2f} bps > max_allowed_slippage_bps {max_allowed_slippage_bps:.2f}"
        (guardrail.errors if ci_mode else guardrail.warnings).append(msg)
    if avg_sell > max_allowed_slippage_bps:
        msg = f"avg SELL slippage {avg_sell:.2f} bps > max_allowed_slippage_bps {max_allowed_slippage_bps:.2f}"
        (guardrail.errors if ci_mode else guardrail.warnings).append(msg)

    out = {
        "net_return": float(net_return),
        "gross_return": float(gross_return),
        "net_pnl": float(net_pnl),
        "gross_pnl": float(gross_pnl),
        "total_fees": float(total_fees),
        "total_slippage_cost": float(total_slippage_cost),
        "buy_slippage_bps": {
            "avg": avg_buy,
            "median": float(buy_bps.median()) if len(buy_bps) else 0.0,
        },
        "sell_slippage_bps": {
            "avg": avg_sell,
            "median": float(sell_bps.median()) if len(sell_bps) else 0.0,
        },
        "notional_distribution": notional_distribution,
        "trade_count": int(len(tr)),
        "guardrails": {
            "warnings": guardrail.warnings,
            "errors": guardrail.errors,
        },
    }

    if ci_mode and guardrail.errors:
        raise ValueError("; ".join(guardrail.errors))

    return out


def run_pnl_decomposition(
    trades_path: str | Path,
    equity_curve_path: str | Path,
    output_dir: str | Path,
    *,
    min_trade_notional_usd: float = 50.0,
    max_allowed_slippage_bps: float = 50.0,
    ci_mode: bool = False,
) -> dict[str, Any]:
    trades_path = Path(trades_path)
    equity_curve_path = Path(equity_curve_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if trades_path.exists():
        try:
            trades_df = pd.read_csv(trades_path)
        except pd.errors.EmptyDataError:
            trades_df = pd.DataFrame()
    else:
        trades_df = pd.DataFrame()

    equity_df = pd.read_csv(equity_curve_path)

    result = compute_pnl_decomposition(
        trades_df,
        equity_df,
        min_trade_notional_usd=min_trade_notional_usd,
        max_allowed_slippage_bps=max_allowed_slippage_bps,
        ci_mode=ci_mode,
    )

    out_path = write_strict_json(output_dir / "execution_quality.json", result)

    print("Execution quality summary")
    print(dumps_strict_json({
        "net_return": result["net_return"],
        "gross_return": result["gross_return"],
        "total_fees": result["total_fees"],
        "total_slippage_cost": result["total_slippage_cost"],
        "trade_count": result["trade_count"],
        "notional_distribution": result["notional_distribution"],
        "guardrails": result["guardrails"],
    }, indent=2))

    return result
