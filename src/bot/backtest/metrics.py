from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _to_pnl_dollars(equity: pd.Series) -> pd.Series:
    return equity.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_cagr(equity: pd.Series, periods_per_year: int = 8760) -> float:
    equity = equity.dropna()
    if equity.empty or len(equity) < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0:
        return 0.0
    years = max(len(equity) - 1, 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((end / start) ** (1 / years) - 1)


def compute_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 8760) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - (rf / periods_per_year)
    std = float(excess.std())
    if std <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def compute_sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 8760) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd_std = float(downside.std()) if len(downside) else 0.0
    if dd_std <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * (r.mean() - rf / periods_per_year) / dd_std)


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    peak = eq.expanding(min_periods=1).max()
    dd = (eq - peak) / peak.replace(0, np.nan)
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(dd.min() if not dd.empty else 0.0)


def profit_factor_from_pnl(pnl_dollars: pd.Series) -> dict[str, float | bool | None]:
    pnl = pnl_dollars.dropna()
    gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
    gross_loss = float(abs(pnl[pnl < 0].sum())) if len(pnl) else 0.0

    if gross_loss <= 1e-12:
        return {
            "profit_factor": None,
            "profit_factor_is_infinite": bool(gross_profit > 0),
            "profit_factor_gross_profit": gross_profit,
            "profit_factor_gross_loss": gross_loss,
        }

    pf = gross_profit / gross_loss
    return {
        "profit_factor": float(pf),
        "profit_factor_is_infinite": False,
        "profit_factor_gross_profit": gross_profit,
        "profit_factor_gross_loss": gross_loss,
    }


def profit_factor_trades_proxy(trades: pd.DataFrame) -> Optional[float]:
    if trades is None or trades.empty:
        return None
    if "notional" not in trades.columns or "side" not in trades.columns:
        return None

    fees = float(trades["fee"].sum()) if "fee" in trades.columns else 0.0
    signed_notional = trades["notional"].where(trades["side"].astype(str).str.upper() == "SELL", -trades["notional"])
    gross = float(signed_notional.sum() - fees)
    win = max(gross, 0.0)
    loss = -min(gross, 0.0)
    if loss <= 1e-12:
        return None
    return float(win / loss)


def turnover(trades: pd.DataFrame, equity_peak: float = 1.0) -> float:
    if trades is None or trades.empty:
        return 0.0
    gross = float(trades["notional"].abs().sum()) if "notional" in trades.columns else 0.0
    return float(gross / max(equity_peak, 1e-12))


def avg_exposure(decision_df: pd.DataFrame) -> float:
    if decision_df.empty or "target" not in decision_df.columns:
        return 0.0
    return float(decision_df["target"].mean())


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    exposure: pd.Series | None = None,
    freq_per_year: int = 8760,
) -> Dict[str, float | bool | None]:
    ret = _to_returns(equity)
    pnl_dollars = _to_pnl_dollars(equity)
    pf = profit_factor_from_pnl(pnl_dollars)

    out: Dict[str, float | bool | None] = {
        "cagr": compute_cagr(equity, freq_per_year),
        "sharpe": compute_sharpe(ret, periods_per_year=freq_per_year),
        "sortino": compute_sortino(ret, periods_per_year=freq_per_year),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": pf["profit_factor"],
        "profit_factor_is_infinite": bool(pf["profit_factor_is_infinite"]),
        "profit_factor_gross_profit": float(pf["profit_factor_gross_profit"]),
        "profit_factor_gross_loss": float(pf["profit_factor_gross_loss"]),
        "profit_factor_trades": profit_factor_trades_proxy(trades),
        "turnover": turnover(trades, float(equity.iloc[0]) if len(equity) else 1.0),
    }
    if exposure is not None:
        out["avg_exposure"] = float(exposure.mean()) if len(exposure) else 0.0
    return out
