from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import BacktestConfig, RegimeConfig, RiskConfig, ExecutionConfig
from ..execution.risk import RiskManager, RiskState
from ..execution.rebalance_policy import RebalancePolicy
from ..features import indicators
from ..features.indicators import donchian_channel, ema, atr as compute_atr, bollinger_bands
from ..features.regime import compute_adx, compute_chop
from ..strategy.regime_switching_orchestrator import RegimeSwitchingOrchestrator, RegimeDecisionBundle
from ..strategy.regime_switching_v4_core import V4CoreStrategy
from ..strategy.macro_gate_benchmark import MacroGateBenchmarkStrategy
from .fill_models import BacktestOrder, MarketState, make_fill_model
from .cost_model import CostModel


@dataclass
class ExecutionEvent:
    ts: datetime
    side: str
    target_before: float
    target_after: float
    fraction_delta: float
    price: float
    mark_price: float
    slippage_bps: float
    slippage_cost: float
    btc_qty: float
    notional: float
    fee: float
    fee_rate: float
    is_taker: bool
    is_maker: bool
    micro_regime: str
    strategy: str
    fill_model: str
    rebalance_reason: str


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
        exec_cfg = self.execution_config or ExecutionConfig()

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

        now_utc = pd.Timestamp.now(tz="UTC")
        hourly = hourly[hourly.index < now_utc.floor("h")]

        if len(hourly) < 2:
            raise ValueError("Need at least 2 hourly candles for no-lookahead fill simulation")

        # Precompute heavy regime inputs once so the backtest loop is O(n),
        # not O(n^2) from repeatedly recomputing rolling features.
        hourly_returns = hourly["close"].pct_change()
        vol_lookback_hours = max(24, reg_cfg.vol_lookback_days * 24)
        vol_min_periods = max(30, vol_lookback_hours // 4)

        realized_vol = indicators.realized_vol(hourly_returns, reg_cfg.realized_vol_window)
        hourly_precomputed: dict[str, Any] = {
            "realized_vol": realized_vol,
            "adx": compute_adx(hourly["high"], hourly["low"], hourly["close"], window=reg_cfg.adx_window),
            "chop": compute_chop(hourly["high"], hourly["low"], hourly["close"], window=reg_cfg.chop_window),
            "vol_thresholds": realized_vol.rolling(vol_lookback_hours, min_periods=vol_min_periods).quantile(
                reg_cfg.vol_high_threshold_quantile
            ),
        }

        # Precompute sub-strategy indicators for O(1) per-bar lookups.
        _donchian_low, _donchian_high = donchian_channel(hourly["high"], hourly["low"], reg_cfg.donchian_window)
        hourly_precomputed["donchian_high"] = _donchian_high
        hourly_precomputed["donchian_low"] = _donchian_low
        hourly_precomputed["atr"] = compute_atr(hourly["high"], hourly["low"], hourly["close"], reg_cfg.atr_window)
        hourly_precomputed["ema_fast"] = ema(hourly["close"], reg_cfg.ema_fast)
        hourly_precomputed["ema_slow"] = ema(hourly["close"], reg_cfg.ema_slow)
        _bb_mid, _bb_upper, _bb_lower = bollinger_bands(hourly["close"], reg_cfg.bb_window, reg_cfg.bb_stdev)
        hourly_precomputed["bb_mid"] = _bb_mid
        hourly_precomputed["bb_upper"] = _bb_upper
        hourly_precomputed["bb_lower"] = _bb_lower

        if reg_cfg.hmm_regime_enabled:
            hourly_precomputed["hmm_features"] = (
                pd.concat(
                    [
                        hourly["close"].pct_change().fillna(0.0),
                        hourly["high"].pct_change().fillna(0.0),
                        hourly["low"].pct_change().fillna(0.0),
                        hourly["volume"].pct_change().fillna(0.0),
                        hourly_precomputed["realized_vol"].fillna(0.0),
                    ],
                    axis=1,
                )
                .rename(
                    columns={
                        0: "close_chg",
                        1: "high_chg",
                        2: "low_chg",
                        3: "volume_chg",
                        4: "realized_vol",
                    }
                )
            )

        strategy_id = cfg.strategy if cfg.strategy else "regime_switching"
        if strategy_id == "regime_switching_v4_core":
            orchestrator = V4CoreStrategy(reg_cfg)
        elif strategy_id == "macro_gate_benchmark":
            orchestrator = MacroGateBenchmarkStrategy(reg_cfg)
        else:
            orchestrator = RegimeSwitchingOrchestrator(reg_cfg)
        risk_mgr = RiskManager(risk_cfg)
        risk_state = RiskState(equity_peak=cfg.initial_equity, current_equity=cfg.initial_equity)

        fill_model = make_fill_model(
            exec_cfg.fill_model,
            slippage_bps=self.slippage_bps,
            spread_bps=exec_cfg.spread_bps if self.use_spread_slippage else 0.0,
            impact_bps=exec_cfg.impact_bps,
        )
        cost_model = CostModel(
            maker_fee_rate=float(self.fees[0]),
            taker_fee_rate=float(self.fees[1]),
            spread_bps=exec_cfg.spread_bps if self.use_spread_slippage else 0.0,
            impact_bps=exec_cfg.impact_bps,
        )
        rebalance = RebalancePolicy(
            policy=exec_cfg.rebalance_policy,
            min_trade_notional_usd=exec_cfg.min_trade_notional_usd,
            min_exposure_delta=exec_cfg.min_exposure_delta,
            target_quantization_step=exec_cfg.target_quantization_step,
            min_time_between_trades_hours=exec_cfg.min_time_between_trades_hours,
            max_trades_per_day=exec_cfg.max_trades_per_day,
        )

        cash = float(cfg.initial_equity)
        btc = 0.0
        events: List[ExecutionEvent] = []
        decisions_rows: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []

        # Iterate on signal bar t and fill on bar t+1
        for i in range(1, len(hourly)):
            signal_ts = hourly.index[i - 1]
            exec_ts = hourly.index[i]
            bar_t = hourly.iloc[i - 1]
            bar_t1 = hourly.iloc[i]

            mark_signal = float(bar_t.get("close", 0.0))
            if mark_signal <= 0:
                continue

            equity_signal = cash + btc * mark_signal
            current_exposure = max(0.0, min(1.0, (btc * mark_signal / equity_signal) if equity_signal > 0 else 0.0))

            bundle: RegimeDecisionBundle = orchestrator.compute_target_position(
                timestamp=signal_ts,
                hourly_df=hourly,
                daily_df=daily,
                current_exposure=current_exposure,
                hourly_idx=i - 1,
                micro_precomputed=hourly_precomputed,
            )
            raw_target = float(bundle.final_target)

            risk_mgr.update_runtime_state(risk_state, equity_signal, signal_ts)
            target = risk_mgr.apply_caps(
                raw_target,
                risk_state,
                signal_ts.to_pydatetime(),
                signal_ts.to_pydatetime(),
                timeframe_minutes=60,
                current_fraction=current_exposure,
            )
            if target != raw_target:
                bundle.final_target = target

            should_trade, target_bucket, rebalance_reason = rebalance.should_rebalance(
                target,
                current_exposure,
                equity_signal,
                signal_ts,
            )

            if should_trade:
                delta = target_bucket - current_exposure
                side = "BUY" if delta > 0 else "SELL"
                desired_notional = abs(delta) * equity_signal
                qty = desired_notional / mark_signal if mark_signal > 0 else 0.0

                # Cap by available inventory/cash before attempting fill.
                if side == "BUY":
                    max_qty_by_cash = cash / (mark_signal * (1.0 + float(self.fees[1]))) if mark_signal > 0 else 0.0
                    qty = min(qty, max(0.0, max_qty_by_cash))
                else:
                    qty = min(qty, max(0.0, btc))

                order_type = "market"
                limit_price = None
                if fill_model.name == "bid_ask" and exec_cfg.maker_first:
                    spread_bps = exec_cfg.spread_bps if self.use_spread_slippage else 0.0
                    mark_open = float(bar_t1.get("open", mark_signal))
                    bid = mark_open * (1.0 - spread_bps / 20_000.0)
                    ask = mark_open * (1.0 + spread_bps / 20_000.0)
                    order_type = "limit"
                    limit_price = bid if side == "BUY" else ask

                order = BacktestOrder(
                    side=side,
                    qty=max(0.0, qty),
                    order_type=order_type,  # maker-first simulation for bid_ask model
                    limit_price=limit_price,
                    post_only=order_type == "limit",
                )
                market_state = MarketState(
                    spread_bps=exec_cfg.spread_bps if self.use_spread_slippage else 0.0,
                    impact_bps=exec_cfg.impact_bps,
                )
                fill = fill_model.fill(order, bar_t, bar_t1, market_state)

                if fill.filled and fill.qty > 0:
                    notional = float(fill.qty * fill.price)
                    fee_rate = cost_model.fee_rate(fill.is_maker)
                    fee = cost_model.fee(notional, fill.is_maker)

                    # Enforce min notional after fill pricing.
                    if notional >= exec_cfg.min_trade_notional_usd:
                        qty_exec = float(fill.qty)

                        if side == "BUY":
                            max_notional = cash / (1.0 + fee_rate)
                            notional = min(notional, max_notional)
                            qty_exec = notional / max(fill.price, 1e-12)
                            fee = cost_model.fee(notional, fill.is_maker)
                            cash -= notional + fee
                            btc += qty_exec
                        else:
                            qty_exec = min(qty_exec, max(0.0, btc))
                            notional = qty_exec * fill.price
                            fee = cost_model.fee(notional, fill.is_maker)
                            btc -= qty_exec
                            cash += notional - fee

                        if qty_exec > 0 and notional > 0:
                            slippage_cost = cost_model.slippage_cost(side, fill.price, fill.mark_price, qty_exec)
                            slippage_bps = cost_model.slippage_bps(side, fill.price, fill.mark_price)
                            events.append(
                                ExecutionEvent(
                                    ts=exec_ts.to_pydatetime(),
                                    side=side,
                                    target_before=current_exposure,
                                    target_after=target_bucket,
                                    fraction_delta=delta,
                                    price=float(fill.price),
                                    mark_price=float(fill.mark_price),
                                    slippage_bps=float(slippage_bps),
                                    slippage_cost=float(slippage_cost),
                                    btc_qty=float(qty_exec),
                                    notional=float(notional),
                                    fee=float(fee),
                                    fee_rate=float(fee_rate),
                                    is_taker=not fill.is_maker,
                                    is_maker=bool(fill.is_maker),
                                    micro_regime=bundle.micro_regime.value,
                                    strategy=bundle.strategy_name,
                                    fill_model=fill_model.name,
                                    rebalance_reason=rebalance_reason,
                                )
                            )
                            rebalance.on_trade(signal_ts, target_bucket)

            equity_exec = cash + btc * float(bar_t1.get("close", mark_signal))
            risk_state.current_equity = equity_exec
            if equity_exec > risk_state.equity_peak:
                risk_state.equity_peak = equity_exec

            drawdown = risk_state.drawdown
            mark_exec = float(bar_t1.get("close", mark_signal))
            post_exposure = max(0.0, min(1.0, (btc * mark_exec / equity_exec) if equity_exec > 0 else 0.0))

            macro_state = str(bundle.metadata.get("macro_state", "OFF"))
            macro_multiplier = float(bundle.metadata.get("macro_multiplier", 0.0) or 0.0)
            macro_score = float(bundle.metadata.get("macro_score", 0.0) or 0.0)
            trend_boost_active = int(bundle.metadata.get("trend_boost_active", 0) or 0)
            boost_multiplier_applied = float(bundle.metadata.get("boost_multiplier_applied", 1.0) or 1.0)

            equity_rows.append(
                {
                    "timestamp": exec_ts,
                    "equity": equity_exec,
                    "cash": cash,
                    "btc": btc,
                    "btc_price": mark_exec,
                    "exposure": post_exposure,
                    "drawdown": drawdown,
                    "micro_regime": bundle.micro_regime.value,
                    "macro_risk_on": bundle.macro_risk_on,
                    "macro_state": macro_state,
                    "macro_multiplier": macro_multiplier,
                    "macro_score": macro_score,
                    "trend_boost_active": trend_boost_active,
                    "strategy": bundle.strategy_name,
                    "target": target_bucket,
                    "raw_target": raw_target,
                }
            )
            decisions_rows.append(
                {
                    "timestamp": signal_ts,
                    "decision_applies_at": exec_ts,
                    "micro_regime": bundle.micro_regime.value,
                    "strategy": bundle.strategy_name,
                    "macro_risk_on": bundle.macro_risk_on,
                    "macro_state": macro_state,
                    "macro_multiplier": macro_multiplier,
                    "macro_score": macro_score,
                    "boost_active": trend_boost_active,
                    "boost_multiplier_applied": boost_multiplier_applied,
                    "macro_reason": bundle.macro_reason,
                    "target": target_bucket,
                    "raw_target": raw_target,
                    "rebalance_reason": rebalance_reason,
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

        from .metrics import compute_metrics
        from .regime_reports import performance_by_regime, regime_switch_count, time_in_regime, turnover_at_regime_changes

        if len(eq_df.index) >= 2:
            step_seconds = float((eq_df.index[1] - eq_df.index[0]).total_seconds())
            freq_per_year = int(round(365 * 24 * 3600 / max(step_seconds, 1.0)))
        else:
            freq_per_year = 8760

        metrics = compute_metrics(eq_df["equity"], tr, eq_df.get("exposure"), freq_per_year=freq_per_year)
        by_regime = performance_by_regime(eq_df, tr, decisions_df=dec, freq_per_year=freq_per_year)
        in_regime = time_in_regime(eq_df)
        switches = regime_switch_count(eq_df)
        turn_reg = turnover_at_regime_changes(eq_df, tr)

        diagnostics = {
            "turnover": metrics.get("turnover", 0.0),
            "regime_switches": switches,
            "turnover_at_regime_changes": turn_reg,
            "fill_model": fill_model.name,
            "rebalance_policy": exec_cfg.rebalance_policy,
            "trade_count": int(len(tr)),
        }

        return BacktestResult(
            equity_curve=eq_df,
            trades=tr,
            decisions=dec,
            metrics={**metrics, "by_regime": by_regime, "time_in_regime": in_regime, "turnover_at_regime_changes": turn_reg},
            regime_stats={"performance_by_regime": by_regime, "time_in_regime": in_regime, "regime_switches": switches},
            diagnostics=diagnostics,
        )
