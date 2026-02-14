from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import BacktestConfig, RegimeConfig, RiskConfig, ExecutionConfig
from ..execution.risk import RiskManager, RiskState
from ..strategy.regime_switching_orchestrator import RegimeSwitchingOrchestrator, RegimeDecisionBundle


@dataclass
class ExecutionEvent:
    ts: datetime
    side: str
    target_before: float
    target_after: float
    fraction_delta: float
    price: float
    btc_qty: float
    notional: float
    fee: float
    fee_rate: float
    is_taker: bool
    micro_regime: str
    strategy: str


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    decisions: pd.DataFrame
    metrics: Dict[str, Any]
    regime_stats: Dict[str, Any]
    diagnostics: Dict[str, Any]


@dataclass
class BacktestEngine:
    product: str
    hourly_candles: pd.DataFrame
    daily_candles: pd.DataFrame
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    config: BacktestConfig | None = None
    fees: tuple[float, float] = (10e-4, 25e-4)
    slippage_bps: float = 5.0
    use_spread_slippage: bool = True
    regime_config: Optional[RegimeConfig] = None
    risk_config: Optional[RiskConfig] = None
    execution_config: Optional[ExecutionConfig] = None

    def run(self) -> BacktestResult:
        cfg = self.config or BacktestConfig()
        reg_cfg = self.regime_config or RegimeConfig()
        risk_cfg = self.risk_config or RiskConfig()
        _ = self.execution_config or ExecutionConfig()  # reserved for future execution-model params

        hourly = self.hourly_candles.copy()
        daily = self.daily_candles.copy()
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)

        def _utc_ts(v: datetime) -> pd.Timestamp:
            ts = pd.Timestamp(v)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        if self.start:
            start_ts = _utc_ts(self.start)
            hourly = hourly[hourly["timestamp"] >= start_ts]
            daily = daily[daily["timestamp"] >= start_ts]
        if self.end:
            end_ts = _utc_ts(self.end)
            hourly = hourly[hourly["timestamp"] <= end_ts]
            daily = daily[daily["timestamp"] <= end_ts]

        if hourly.empty:
            raise ValueError("No hourly candles in backtest window")
        if daily.empty:
            raise ValueError("No daily candles in backtest window")

        hourly = hourly.set_index("timestamp").sort_index()
        daily = daily.set_index("timestamp").sort_index()

        # align to closed bars, drop potential current/partial bar
        now_utc = pd.Timestamp.now(tz="UTC")
        hourly = hourly[hourly.index < now_utc.floor("h")]

        orchestrator = RegimeSwitchingOrchestrator(reg_cfg)
        risk_mgr = RiskManager(risk_cfg)
        risk_state = RiskState(equity_peak=cfg.initial_equity, current_equity=cfg.initial_equity)

        equity = cfg.initial_equity
        cash = cfg.initial_equity
        btc = 0.0
        events: List[ExecutionEvent] = []
        decisions_rows: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []

        for i, (ts, row) in enumerate(hourly.iterrows()):
            # make sure daily context available up to this timestamp
            daily_ctx = daily[daily.index <= ts]
            if daily_ctx.empty:
                continue
            hourly_ctx = hourly.iloc[: i + 1]

            bundle: RegimeDecisionBundle = orchestrator.compute_target_position(ts, hourly_ctx.reset_index(), daily_ctx.reset_index(), current_exposure=btc * row["close"] / max(equity, 1e-9) if equity > 0 else 0.0)
            target = bundle.final_target

            # apply stale data and drawdown breakers
            # In backtests, use simulated time (ts) for stale-data checks.
            target = risk_mgr.apply_caps(target, risk_state, ts.to_pydatetime(), ts.to_pydatetime(), timeframe_minutes=60)
            if target != bundle.final_target:
                bundle.final_target = target

            current_exposure = max(0.0, min(1.0, btc * row["close"] / equity if equity > 0 else 0.0))

            if target > current_exposure:
                side = "BUY"
                delta = target - current_exposure
                is_taker = True
                fee_rate = float(self.fees[1])
                # apply slippage as positive drift for buys
                fill_price = float(row["close"]) * (1 + self.slippage_bps / 1e4)
                if self.use_spread_slippage:
                    spread_adj = (row.get("high", row["close"]) - row.get("low", row["close"])) / max(1.0, row["close"]) / 2
                    fill_price *= 1 + min(1.0, spread_adj)
            elif target < current_exposure:
                side = "SELL"
                delta = target - current_exposure
                is_taker = True
                fee_rate = float(self.fees[1])
                fill_price = float(row["close"]) * (1 - self.slippage_bps / 1e4)
                if self.use_spread_slippage:
                    spread_adj = (row.get("high", row["close"]) - row.get("low", row["close"])) / max(1.0, row["close"]) / 2
                    fill_price *= 1 - min(1.0, spread_adj)
            else:
                side = "NONE"
                delta = 0.0

            if abs(delta) >= 1e-6 and equity > 0:
                desired_notional = abs(delta) * equity
                if side == "BUY":
                    max_notional = cash / (1.0 + fee_rate)
                    notional = max(0.0, min(desired_notional, max_notional))
                    btc_delta = notional / max(fill_price, 1e-12)
                    cash -= notional
                    btc += btc_delta
                else:
                    max_btc = max(0.0, btc)
                    btc_delta = min(desired_notional / max(fill_price, 1e-12), max_btc)
                    notional = btc_delta * fill_price
                    cash += notional
                    btc -= btc_delta

                if notional > 0 and btc_delta > 0:
                    fee = notional * fee_rate
                    cash -= fee
                    events.append(
                        ExecutionEvent(
                            ts=ts.to_pydatetime(),
                            side=side,
                            target_before=current_exposure,
                            target_after=target,
                            fraction_delta=delta,
                            price=fill_price,
                            btc_qty=btc_delta,
                            notional=notional,
                            fee=fee,
                            fee_rate=fee_rate,
                            is_taker=is_taker,
                            micro_regime=bundle.micro_regime.value,
                            strategy=bundle.strategy_name,
                        )
                    )
                equity = cash + btc * row["close"]
                risk_state.current_equity = equity
            else:
                equity = cash + btc * row["close"]

            risk_state.current_equity = equity
            if equity > risk_state.equity_peak:
                risk_state.equity_peak = equity

            dd = risk_state.drawdown
            post_exposure = max(0.0, min(1.0, (btc * row["close"] / equity) if equity > 0 else 0.0))
            equity_rows.append(
                {
                    "timestamp": ts,
                    "equity": equity,
                    "cash": cash,
                    "btc": btc,
                    "btc_price": row["close"],
                    "exposure": post_exposure,
                    "drawdown": dd,
                    "micro_regime": bundle.micro_regime.value,
                    "macro_risk_on": bundle.macro_risk_on,
                    "strategy": bundle.strategy_name,
                    "target": bundle.final_target,
                }
            )
            decisions_rows.append(
                {
                    "timestamp": ts,
                    "micro_regime": bundle.micro_regime.value,
                    "strategy": bundle.strategy_name,
                    "macro_risk_on": bundle.macro_risk_on,
                    "macro_reason": bundle.macro_reason,
                    "target": bundle.final_target,
                    "regime_target": bundle.regime_target,
                    "base_target": bundle.base_target,
                    "metadata": bundle.metadata,
                }
            )

        if not equity_rows:
            raise ValueError("Backtest produced no evaluable bars in the selected window")

        eq_df = pd.DataFrame(equity_rows).set_index("timestamp").sort_index()
        tr = pd.DataFrame([e.__dict__ for e in events])
        dec_raw = pd.DataFrame(decisions_rows)
        dec = dec_raw.set_index("timestamp").sort_index() if not dec_raw.empty else pd.DataFrame()

        # diagnostics
        from .metrics import compute_metrics
        from .regime_reports import performance_by_regime, regime_switch_count, time_in_regime, turnover_at_regime_changes

        if len(eq_df.index) >= 2:
            step_seconds = float((eq_df.index[1] - eq_df.index[0]).total_seconds())
            freq_per_year = int(round(365 * 24 * 3600 / max(step_seconds, 1.0)))
        else:
            freq_per_year = 8760
        metrics = compute_metrics(eq_df["equity"], tr, eq_df.get("exposure"), freq_per_year=freq_per_year)
        by_regime = performance_by_regime(eq_df, tr)
        in_regime = time_in_regime(eq_df)
        switches = regime_switch_count(eq_df)
        turn_reg = turnover_at_regime_changes(eq_df, tr)

        diagnostics = {
            "turnover": metrics.get("turnover", 0.0),
            "regime_switches": switches,
            "turnover_at_regime_changes": turn_reg,
        }

        return BacktestResult(
            equity_curve=eq_df,
            trades=tr,
            decisions=dec,
            metrics={**metrics, "by_regime": by_regime, "time_in_regime": in_regime, "turnover_at_regime_changes": turn_reg},
            regime_stats={"performance_by_regime": by_regime, "time_in_regime": in_regime, "regime_switches": switches},
            diagnostics=diagnostics,
        )
