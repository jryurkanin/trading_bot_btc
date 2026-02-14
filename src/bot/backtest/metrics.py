from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


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
    return (end / start) ** (1 / years) - 1


def compute_sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 8760) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    excess = r - (rf / periods_per_year)
    std = excess.std()
    if std <= 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def compute_sortino(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 8760) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd_std = downside.std()
    if dd_std <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * (r.mean() - rf / periods_per_year) / dd_std)


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    peak = eq.expanding(min_periods=1).max()
    dd = (eq - peak) / peak
    return float(dd.min() if not dd.empty else 0.0)


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    fees = trades["fee"].sum() if "fee" in trades.columns else 0.0
    pnl = 0.0
    # simple proxy: estimate from notional changes as signed impact (buy+ = cost, sell- = proceeds) not exact PnL
    # better than nothing for tests.
    notional = trades["notional"] if "notional" in trades.columns else pd.Series([], dtype=float)
    if notional.empty:
        return 0.0
    gross = (notional.where(trades["side"] == "SELL", -notional)).sum() - fees
    # no direct closed-trade PnL is available in this scaffold
    win = max(gross, 0)
    loss = -min(gross, 0)
    if loss <= 0:
        return float("inf") if win > 0 else 0.0
    return float(win / max(1e-12, loss))


def turnover(trades: pd.DataFrame, equity_peak: float = 1.0) -> float:
    if trades.empty:
        return 0.0
    gross = float(trades["notional"].abs().sum()) if "notional" in trades.columns else 0.0
    return gross / max(equity_peak, 1e-12)


def avg_exposure(decision_df: pd.DataFrame) -> float:
    if decision_df.empty or "target" not in decision_df.columns:
        return 0.0
    return float(decision_df["target"].mean())


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, exposure: pd.Series | None = None, freq_per_year: int = 8760) -> Dict[str, float]:
    ret = _to_returns(equity)
    out = {
        "cagr": compute_cagr(equity, freq_per_year),
        "sharpe": compute_sharpe(ret, periods_per_year=freq_per_year),
        "sortino": compute_sortino(ret, periods_per_year=freq_per_year),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(trades),
        "turnover": turnover(trades, equity.iloc[0] if len(equity) else 1.0),
    }
    if exposure is not None:
        out["avg_exposure"] = float(exposure.mean()) if len(exposure) else 0.0
    return out
