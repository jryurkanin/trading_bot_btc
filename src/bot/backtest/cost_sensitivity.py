"""Transaction cost sensitivity analysis via full engine re-runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..config import BotConfig
from ..system_log import get_system_logger
from .engine import BacktestEngine

logger = get_system_logger("backtest.cost_sensitivity")


@dataclass
class CostSensitivityConfig:
    multiplier_min: float = 0.0
    multiplier_max: float = 3.0
    n_steps: int = 20  # produces n_steps + 1 points
    sweep_fees: bool = True
    sweep_spread: bool = True
    sweep_impact: bool = True


@dataclass
class CostSensitivityResult:
    combined_sweep: pd.DataFrame  # cost_multiplier, cagr, sharpe, sortino, max_drawdown, turnover, trade_count
    breakeven_sharpe_0: Optional[float] = None
    breakeven_sharpe_1: Optional[float] = None
    breakeven_cagr_0: Optional[float] = None


def _clone_cfg(cfg: BotConfig) -> BotConfig:
    """Clone a BotConfig using the Pydantic round-trip pattern."""
    if hasattr(cfg, "model_dump") and hasattr(BotConfig, "model_validate"):
        return BotConfig.model_validate(cfg.model_dump())
    if hasattr(cfg, "dict"):
        return BotConfig.parse_obj(cfg.dict())
    return BotConfig.parse_obj(dict(cfg.__dict__))


def _interpolate_breakeven(
    multipliers: np.ndarray, values: np.ndarray, threshold: float
) -> Optional[float]:
    """Find the multiplier where *values* crosses *threshold* via linear interpolation."""
    for i in range(len(values) - 1):
        v0, v1 = values[i], values[i + 1]
        if (v0 >= threshold and v1 < threshold) or (v0 > threshold and v1 <= threshold):
            # Linear interpolation
            denom = v0 - v1
            if abs(denom) < 1e-15:
                continue
            frac = (v0 - threshold) / denom
            return float(multipliers[i] + frac * (multipliers[i + 1] - multipliers[i]))
    return None


def run_cost_sensitivity(
    hourly: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: BotConfig,
    config: CostSensitivityConfig | None = None,
) -> CostSensitivityResult:
    """Sweep transaction cost multipliers and measure metric degradation.

    Re-runs the full backtest engine at each cost level so that cost changes
    affect trade decisions through the rebalance policy.
    """
    if config is None:
        config = CostSensitivityConfig()

    multipliers = np.linspace(config.multiplier_min, config.multiplier_max, config.n_steps + 1)

    # Baseline cost parameters
    base_maker_bps = cfg.backtest.maker_bps
    base_taker_bps = cfg.backtest.taker_bps
    base_spread_bps = cfg.execution.spread_bps
    base_impact_bps = cfg.execution.impact_bps

    rows = []
    for mult in multipliers:
        run_cfg = _clone_cfg(cfg)

        # Scale costs by multiplier
        if config.sweep_fees:
            run_cfg.backtest.maker_bps = float(base_maker_bps * mult)
            run_cfg.backtest.taker_bps = float(base_taker_bps * mult)
        if config.sweep_spread:
            run_cfg.execution.spread_bps = float(base_spread_bps * mult)
        if config.sweep_impact:
            run_cfg.execution.impact_bps = float(base_impact_bps * mult)

        maker = run_cfg.backtest.maker_bps / 10_000.0
        taker = run_cfg.backtest.taker_bps / 10_000.0

        # Determine start/end from config or candle range
        h = hourly.copy()
        h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
        d = daily.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)

        start = h["timestamp"].min().to_pydatetime() if run_cfg.backtest.start is None else pd.Timestamp(run_cfg.backtest.start, tz="UTC").to_pydatetime()
        end = h["timestamp"].max().to_pydatetime() if run_cfg.backtest.end is None else pd.Timestamp(run_cfg.backtest.end, tz="UTC").to_pydatetime()

        engine = BacktestEngine(
            product=run_cfg.data.product,
            hourly_candles=hourly,
            daily_candles=daily,
            start=start,
            end=end,
            config=run_cfg.backtest,
            fees=(maker, taker),
            slippage_bps=run_cfg.backtest.slippage_bps,
            use_spread_slippage=run_cfg.backtest.use_spread_slippage,
            regime_config=run_cfg.regime,
            risk_config=run_cfg.risk,
            execution_config=run_cfg.execution,
            fred_config=run_cfg.fred,
        )
        result = engine.run()

        rows.append({
            "cost_multiplier": float(mult),
            "cagr": result.metrics.get("cagr", 0.0),
            "sharpe": result.metrics.get("sharpe", 0.0),
            "sortino": result.metrics.get("sortino", 0.0),
            "max_drawdown": result.metrics.get("max_drawdown", 0.0),
            "turnover": result.metrics.get("turnover", 0.0),
            "trade_count": len(result.trades),
        })

        logger.info(
            "cost_sensitivity mult=%.3f sharpe=%.4f cagr=%.4f trades=%d",
            mult,
            rows[-1]["sharpe"],
            rows[-1]["cagr"],
            rows[-1]["trade_count"],
        )

    df = pd.DataFrame(rows)
    mults = df["cost_multiplier"].values
    sharpes = df["sharpe"].values.astype(float)
    cagrs = df["cagr"].values.astype(float)

    return CostSensitivityResult(
        combined_sweep=df,
        breakeven_sharpe_0=_interpolate_breakeven(mults, sharpes, 0.0),
        breakeven_sharpe_1=_interpolate_breakeven(mults, sharpes, 1.0),
        breakeven_cagr_0=_interpolate_breakeven(mults, cagrs, 0.0),
    )
