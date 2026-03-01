from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._bootstrap import circular_block_bootstrap_sample


def _to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _to_pnl_dollars(equity: pd.Series) -> pd.Series:
    return equity.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def compute_cagr(equity: pd.Series, periods_per_year: int = 8760) -> float:
    equity = equity.dropna()
    if equity.empty or len(equity) < 2:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0
    # Use calendar time if index is datetime, fall back to bar count
    if isinstance(equity.index, pd.DatetimeIndex):
        td = equity.index[-1] - equity.index[0]
        years = td.total_seconds() / (365.25 * 24 * 3600)
    else:
        years = max(len(equity) - 1, 1) / periods_per_year
    if years <= 1e-6:
        return 0.0
    try:
        return float((end_val / start_val) ** (1 / years) - 1)
    except (OverflowError, ValueError):
        return 0.0


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
    """Return maximum drawdown as a negative fraction (e.g. -0.20 for 20% drawdown).

    Convention: always negative or zero.  To compare with RiskState.drawdown
    (which uses the positive convention 0.20), use ``abs(max_drawdown(eq))``.
    """
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


# ---------------------------------------------------------------------------
# Sharpe ratio bootstrap confidence intervals
# ---------------------------------------------------------------------------

# Hardcoded z-values to avoid scipy dependency
_Z_VALUES: Dict[float, float] = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass
class SharpeBootstrapResult:
    point_estimate: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    bootstrap_mean: float
    bootstrap_std: float
    p_value_sharpe_leq_0: float
    hac_std_error: float
    hac_confidence_intervals: Dict[str, Tuple[float, float]]
    n_simulations: int
    block_length_used: int


def _newey_west_hac_se(
    excess_returns: np.ndarray,
    periods_per_year: int,
) -> float:
    """Newey-West HAC standard error of annualised Sharpe ratio.

    Bartlett kernel with Andrews (1991) automatic bandwidth:
    ``int(4 * (n / 100) ^ (2/9))``.
    """
    n = len(excess_returns)
    if n < 2:
        return 0.0

    mean_r = float(np.mean(excess_returns))
    std_r = float(np.std(excess_returns, ddof=1))
    if std_r <= 0:
        return 0.0

    # Bandwidth selection (Andrews 1991)
    bandwidth = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    bandwidth = max(bandwidth, 1)

    # Centred residuals
    resid = excess_returns - mean_r

    # Gamma_0
    gamma_0 = float(np.dot(resid, resid)) / n

    # Accumulate HAC variance with Bartlett weights
    hac_var = gamma_0
    for lag in range(1, bandwidth + 1):
        w = 1.0 - lag / (bandwidth + 1.0)
        gamma_lag = float(np.dot(resid[lag:], resid[:-lag])) / n
        hac_var += 2.0 * w * gamma_lag

    if hac_var <= 0:
        return 0.0

    # SE of the mean
    se_mean = float(np.sqrt(hac_var / n))

    # Delta-method: Sharpe = sqrt(T) * mean / std, so
    # se(Sharpe_annualised) ≈ sqrt(periods_per_year) * se_mean / std_r
    se_sharpe = float(np.sqrt(periods_per_year)) * se_mean / std_r
    return se_sharpe


def bootstrap_sharpe_confidence(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 8760,
    n_bootstrap: int = 10_000,
    block_length: Optional[int] = None,
    confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
    seed: int = 42,
) -> SharpeBootstrapResult:
    """Compute bootstrap confidence intervals for the annualised Sharpe ratio.

    Uses circular block bootstrap to preserve autocorrelation, plus
    Newey-West HAC-based analytic intervals.
    """
    r = returns.dropna().values.astype(float)
    n = len(r)

    # Point estimate via existing helper
    point_estimate = compute_sharpe(returns, rf=rf, periods_per_year=periods_per_year)

    # Default block length: int(n^(1/3))
    if block_length is None:
        block_length = max(1, int(n ** (1.0 / 3.0)))

    excess = r - rf / periods_per_year

    rng = np.random.default_rng(seed)
    boot_sharpes = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = circular_block_bootstrap_sample(excess, block_length, rng)
        std_s = float(np.std(sample, ddof=1))
        if std_s <= 0:
            boot_sharpes[i] = 0.0
        else:
            boot_sharpes[i] = float(np.sqrt(periods_per_year) * np.mean(sample) / std_s)

    # Percentile-based CIs
    cis: Dict[str, Tuple[float, float]] = {}
    for level in confidence_levels:
        alpha = (1.0 - level) / 2.0
        lo = float(np.percentile(boot_sharpes, 100 * alpha))
        hi = float(np.percentile(boot_sharpes, 100 * (1 - alpha)))
        cis[f"{int(level * 100)}%"] = (lo, hi)

    p_value = float(np.mean(boot_sharpes <= 0))

    # HAC standard error
    hac_se = _newey_west_hac_se(excess, periods_per_year)

    hac_cis: Dict[str, Tuple[float, float]] = {}
    for level in confidence_levels:
        z = _Z_VALUES.get(level, 1.96)
        lo = point_estimate - z * hac_se
        hi = point_estimate + z * hac_se
        hac_cis[f"{int(level * 100)}%"] = (lo, hi)

    return SharpeBootstrapResult(
        point_estimate=point_estimate,
        confidence_intervals=cis,
        bootstrap_mean=float(np.mean(boot_sharpes)),
        bootstrap_std=float(np.std(boot_sharpes, ddof=1)),
        p_value_sharpe_leq_0=p_value,
        hac_std_error=hac_se,
        hac_confidence_intervals=hac_cis,
        n_simulations=n_bootstrap,
        block_length_used=block_length,
    )
