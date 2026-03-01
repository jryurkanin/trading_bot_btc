from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import BacktestConfig, RegimeConfig, RiskConfig, ExecutionConfig, FredConfig
from ..execution.risk import RiskManager, RiskState
from ..execution.rebalance_policy import RebalancePolicy
from ..features import indicators
from ..features.indicators import donchian_channel, ema, atr as compute_atr, bollinger_bands
from ..acceleration.cuda_backend import resolve_acceleration_backend
from ..features.regime import compute_adx, compute_chop
from ..features.fred_features import build_fred_daily_overlay_features
from ..strategy.regime_switching_orchestrator import RegimeDecisionBundle, RegimeSwitchingOrchestrator
from ..strategy.macro_gate_benchmark import MacroGateBenchmarkStrategy
from ..strategy.macro_only_v2 import MacroOnlyV2Strategy
from ..strategy.regime_switching_v4_core import V4CoreStrategy
from ..strategy.v5_adaptive import V5AdaptiveStrategy
from ..system_log import setup_system_logger, get_system_logger
from .fill_models import BacktestOrder, MarketState, make_fill_model
from .cost_model import CostModel


logger = get_system_logger("backtest.engine")


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
    fred_config: Optional[FredConfig] = None

    def run(self) -> BacktestResult:
        log_path = setup_system_logger()
        logger.info(
            "engine_run_start log_path=%s product=%s start=%s end=%s",
            log_path,
            self.product,
            self.start,
            self.end,
        )

        cfg = self.config or BacktestConfig()
        reg_cfg = self.regime_config or RegimeConfig()
        risk_cfg = self.risk_config or RiskConfig()
        exec_cfg = self.execution_config or ExecutionConfig()
        fred_cfg = self.fred_config or FredConfig()

        hourly = self.hourly_candles.copy()
        daily = self.daily_candles.copy()
        hourly["timestamp"] = pd.to_datetime(hourly["timestamp"], utc=True)
        daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)

        def _utc_ts(v: datetime) -> pd.Timestamp:
            ts = pd.Timestamp(v)
            if ts.tzinfo is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        trade_start_ts = _utc_ts(self.start) if self.start else None

        # Keep pre-start history for feature warmup (daily macro + hourly rolling
        # indicators), but only execute trades for bars with exec_ts >= start.
        if self.end:
            end_ts = _utc_ts(self.end)
            hourly = hourly[hourly["timestamp"] <= end_ts]
            daily = daily[daily["timestamp"] <= end_ts]

        if hourly.empty:
            raise ValueError("No hourly candles in backtest window")
        if daily.empty:
            raise ValueError("No daily candles in backtest window")

        fred_report: dict[str, Any] = {
            "enabled": bool(getattr(fred_cfg, "enabled", False)),
            "series_used": [],
            "series_lags_hours": {},
            "warnings": [],
        }

        if bool(getattr(fred_cfg, "enabled", False)):
            try:
                fred_build = build_fred_daily_overlay_features(daily, fred_cfg)
                daily = fred_build.daily_features
                fred_report = dict(fred_build.report)
            except Exception as exc:
                logger.warning("FRED overlay build failed; continuing without FRED features: %s", exc)
                fred_report = {
                    "enabled": True,
                    "series_used": [],
                    "series_lags_hours": {},
                    "warnings": [f"fred_overlay_failed:{exc.__class__.__name__}"],
                }

        hourly = hourly.set_index("timestamp").sort_index()
        daily = daily.set_index("timestamp").sort_index()

        now_utc = pd.Timestamp.now(tz="UTC")
        hourly = hourly[hourly.index < now_utc.floor("h")]

        if len(hourly) < 2:
            raise ValueError("Need at least 2 hourly candles for no-lookahead fill simulation")

        if trade_start_ts is not None:
            start_i = int(hourly.index.searchsorted(trade_start_ts, side="left"))
            start_i = max(1, start_i)
        else:
            start_i = 1

        if start_i >= len(hourly):
            raise ValueError("No hourly candles in backtest window")

        logger.info(
            "engine_input_ready product=%s strategy=%s hourly_rows=%d daily_rows=%d start_i=%d",
            self.product,
            cfg.strategy,
            len(hourly),
            len(daily),
            start_i,
        )

        strategy_id = cfg.strategy if cfg.strategy else "macro_gate_benchmark"
        if strategy_id == "macro_gate_benchmark":
            orchestrator = MacroGateBenchmarkStrategy(reg_cfg)
        elif strategy_id == "macro_only_v2":
            orchestrator = MacroOnlyV2Strategy(reg_cfg)
        elif strategy_id == "macro_gate_state":
            orchestrator = RegimeSwitchingOrchestrator(reg_cfg)
        elif strategy_id == "regime_switching_v4_core":
            orchestrator = V4CoreStrategy(reg_cfg)
        elif strategy_id == "regime_switching_orchestrator":
            orchestrator = RegimeSwitchingOrchestrator(reg_cfg)
        elif strategy_id == "v5_adaptive":
            orchestrator = V5AdaptiveStrategy(reg_cfg)
        else:
            valid = ", ".join(sorted(["macro_gate_benchmark", "macro_only_v2", "macro_gate_state", "regime_switching_v4_core", "regime_switching_orchestrator", "v5_adaptive"]))
            raise ValueError(f"Unsupported strategy '{strategy_id}'. Supported: [{valid}]")

        logger.info("engine_strategy_selected strategy_id=%s orchestrator=%s", strategy_id, orchestrator.__class__.__name__)

        # Precompute regime inputs once so per-bar decision logic stays O(n).
        # Prefer strategy-provided precompute hooks to avoid unnecessary work.
        acc_ctx = resolve_acceleration_backend(getattr(cfg, "acceleration_backend", "auto"))
        acc_backend = "cpu"
        if (
            acc_ctx.backend == "cuda"
            and len(hourly) >= int(getattr(cfg, "acceleration_min_bars", 2048) or 2048)
        ):
            acc_backend = "cuda"

        hourly_precomputed: dict[str, Any] = {
            "acceleration_backend": acc_backend,
            "acceleration_requested": getattr(cfg, "acceleration_backend", "auto"),
            "acceleration_cuda_available": int(acc_ctx.cuda_available),
            "acceleration_device": acc_ctx.device_name,
            "acceleration_fallback_reason": acc_ctx.reason,
        }

        precompute_fn = getattr(orchestrator.__class__, "get_precomputed_features", None)
        if callable(precompute_fn):
            try:
                extra = precompute_fn(hourly, reg_cfg, backend=acc_backend)
                if isinstance(extra, dict):
                    hourly_precomputed.update(extra)
            except Exception as exc:
                logger.warning(
                    "engine_precompute_hook_failed strategy=%s orchestrator=%s err=%s",
                    strategy_id,
                    orchestrator.__class__.__name__,
                    exc,
                )

        # Legacy orchestrator sub-strategies depend on these series to avoid
        # expensive per-bar indicator recomputation.
        if strategy_id in {"macro_gate_state", "regime_switching_orchestrator"}:
            if "donchian_high" not in hourly_precomputed or "donchian_low" not in hourly_precomputed:
                _donchian_low, _donchian_high = donchian_channel(
                    hourly["high"],
                    hourly["low"],
                    reg_cfg.donchian_window,
                    backend=acc_backend,
                )
                hourly_precomputed.setdefault("donchian_high", _donchian_high)
                hourly_precomputed.setdefault("donchian_low", _donchian_low)

            if "atr" not in hourly_precomputed:
                hourly_precomputed["atr"] = compute_atr(
                    hourly["high"],
                    hourly["low"],
                    hourly["close"],
                    reg_cfg.atr_window,
                    backend=acc_backend,
                )

            if "ema_fast" not in hourly_precomputed:
                hourly_precomputed["ema_fast"] = ema(hourly["close"], reg_cfg.ema_fast, backend=acc_backend)
            if "ema_slow" not in hourly_precomputed:
                hourly_precomputed["ema_slow"] = ema(hourly["close"], reg_cfg.ema_slow, backend=acc_backend)

            if not {"bb_mid", "bb_upper", "bb_lower"}.issubset(hourly_precomputed.keys()):
                _bb_mid, _bb_upper, _bb_lower = bollinger_bands(
                    hourly["close"],
                    reg_cfg.bb_window,
                    reg_cfg.bb_stdev,
                    backend=acc_backend,
                )
                hourly_precomputed.setdefault("bb_mid", _bb_mid)
                hourly_precomputed.setdefault("bb_upper", _bb_upper)
                hourly_precomputed.setdefault("bb_lower", _bb_lower)

            if reg_cfg.hmm_regime_enabled and "hmm_features" not in hourly_precomputed:
                rv_seed = hourly_precomputed.get("realized_vol")
                if not isinstance(rv_seed, pd.Series):
                    rv_seed = indicators.realized_vol(
                        hourly["close"].pct_change(),
                        reg_cfg.realized_vol_window,
                        backend=acc_backend,
                    )
                    hourly_precomputed["realized_vol"] = rv_seed
                hourly_precomputed["hmm_features"] = (
                    pd.concat(
                        [
                            hourly["close"].pct_change().fillna(0.0),
                            hourly["high"].pct_change().fillna(0.0),
                            hourly["low"].pct_change().fillna(0.0),
                            hourly["volume"].pct_change().fillna(0.0),
                            rv_seed.fillna(0.0),
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

        # Compatibility guards for any path that expects realized_vol/vol_thresholds.
        if not isinstance(hourly_precomputed.get("realized_vol"), pd.Series):
            hourly_returns = hourly["close"].pct_change()
            hourly_precomputed["realized_vol"] = indicators.realized_vol(
                hourly_returns,
                reg_cfg.realized_vol_window,
                backend=acc_backend,
            )

        if "vol_thresholds" not in hourly_precomputed:
            vol_lookback_hours = max(24, reg_cfg.vol_lookback_days * 24)
            vol_min_periods = max(30, vol_lookback_hours // 4)
            hourly_precomputed["vol_thresholds"] = hourly_precomputed["realized_vol"].rolling(
                vol_lookback_hours,
                min_periods=vol_min_periods,
            ).quantile(reg_cfg.vol_high_threshold_quantile)

        logger.info(
            "engine_acceleration requested=%s effective=%s cuda_available=%s device=%s fallback_reason=%s precomputed_keys=%s",
            hourly_precomputed.get("acceleration_requested"),
            hourly_precomputed.get("acceleration_backend"),
            hourly_precomputed.get("acceleration_cuda_available"),
            hourly_precomputed.get("acceleration_device"),
            hourly_precomputed.get("acceleration_fallback_reason"),
            sorted(hourly_precomputed.keys()),
        )

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

        def _meta_float(meta: dict[str, Any], key: str, default: float) -> float:
            try:
                value = float(meta.get(key, default))
                if value != value:  # NaN-safe check
                    return float(default)
                return value
            except Exception:
                return float(default)

        # Iterate on signal bar t and fill on bar t+1. Use pre-start bars as
        # feature warmup context, but only execute/evaluate bars from start_i.
        for i in range(start_i, len(hourly)):
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

            logger.debug(
                "decision ts=%s strategy=%s micro=%s macro_state=%s risk_on=%s raw_target=%.6f capped_target=%.6f target_bucket=%.6f exposure=%.6f should_trade=%s reason=%s",
                signal_ts,
                bundle.strategy_name,
                bundle.micro_regime.value,
                bundle.metadata.get("macro_state", "OFF"),
                bundle.macro_risk_on,
                raw_target,
                target,
                target_bucket,
                current_exposure,
                should_trade,
                rebalance_reason,
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

                logger.debug(
                    "trade_attempt ts=%s side=%s qty=%.8f desired_notional=%.6f delta=%.6f order_type=%s limit_price=%s",
                    exec_ts,
                    side,
                    qty,
                    desired_notional,
                    delta,
                    order_type,
                    limit_price,
                )

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
                            logger.info(
                                "trade_filled ts=%s strategy=%s side=%s qty=%.8f price=%.6f notional=%.6f fee=%.6f maker=%s slippage_bps=%.4f exposure_before=%.6f exposure_after=%.6f",
                                exec_ts,
                                bundle.strategy_name,
                                side,
                                qty_exec,
                                float(fill.price),
                                float(notional),
                                float(fee),
                                bool(fill.is_maker),
                                float(slippage_bps),
                                current_exposure,
                                target_bucket,
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
            macro_multiplier = _meta_float(bundle.metadata, "macro_multiplier", 0.0)
            macro_score = _meta_float(bundle.metadata, "macro_score", 0.0)
            macro_score_raw = _meta_float(bundle.metadata, "macro_score_raw", macro_score)
            macro_score_after_fred = _meta_float(bundle.metadata, "macro_score_after_fred", macro_score)
            fred_risk_off_score = _meta_float(bundle.metadata, "fred_risk_off_score", 0.0)
            fred_penalty_multiplier = _meta_float(bundle.metadata, "fred_penalty_multiplier", 1.0)
            fred_comp_vix_z = _meta_float(bundle.metadata, "fred_comp_vix_z", float("nan"))
            fred_comp_hy_oas_z = _meta_float(bundle.metadata, "fred_comp_hy_oas_z", float("nan"))
            fred_comp_stlfsi_z = _meta_float(bundle.metadata, "fred_comp_stlfsi_z", float("nan"))
            fred_comp_nfci_z = _meta_float(bundle.metadata, "fred_comp_nfci_z", float("nan"))
            fred_vix_level = _meta_float(bundle.metadata, "fred_vix_level", float("nan"))
            fred_hy_oas_level = _meta_float(bundle.metadata, "fred_hy_oas_level", float("nan"))
            fred_stlfsi_level = _meta_float(bundle.metadata, "fred_stlfsi_level", float("nan"))
            fred_nfci_level = _meta_float(bundle.metadata, "fred_nfci_level", float("nan"))
            trend_boost_active = int(bundle.metadata.get("trend_boost_active", 0) or 0)
            boost_multiplier_applied = _meta_float(bundle.metadata, "boost_multiplier_applied", 1.0)

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
                    "macro_score": macro_score_after_fred,
                    "macro_score_raw": macro_score_raw,
                    "macro_score_after_fred": macro_score_after_fred,
                    "fred_risk_off_score": fred_risk_off_score,
                    "fred_penalty_multiplier": fred_penalty_multiplier,
                    "fred_comp_vix_z": fred_comp_vix_z,
                    "fred_comp_hy_oas_z": fred_comp_hy_oas_z,
                    "fred_comp_stlfsi_z": fred_comp_stlfsi_z,
                    "fred_comp_nfci_z": fred_comp_nfci_z,
                    "fred_vix_level": fred_vix_level,
                    "fred_hy_oas_level": fred_hy_oas_level,
                    "fred_stlfsi_level": fred_stlfsi_level,
                    "fred_nfci_level": fred_nfci_level,
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
                    "macro_score": macro_score_after_fred,
                    "macro_score_raw": macro_score_raw,
                    "macro_score_after_fred": macro_score_after_fred,
                    "fred_risk_off_score": fred_risk_off_score,
                    "fred_penalty_multiplier": fred_penalty_multiplier,
                    "fred_comp_vix_z": fred_comp_vix_z,
                    "fred_comp_hy_oas_z": fred_comp_hy_oas_z,
                    "fred_comp_stlfsi_z": fred_comp_stlfsi_z,
                    "fred_comp_nfci_z": fred_comp_nfci_z,
                    "fred_vix_level": fred_vix_level,
                    "fred_hy_oas_level": fred_hy_oas_level,
                    "fred_stlfsi_level": fred_stlfsi_level,
                    "fred_nfci_level": fred_nfci_level,
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
            "acceleration_backend": hourly_precomputed.get("acceleration_backend", "cpu"),
            "acceleration_requested": hourly_precomputed.get("acceleration_requested", "auto"),
            "acceleration_cuda_available": int(hourly_precomputed.get("acceleration_cuda_available", 0) or 0),
            "acceleration_device": hourly_precomputed.get("acceleration_device"),
            "acceleration_fallback_reason": hourly_precomputed.get("acceleration_fallback_reason"),
            "fred": fred_report,
        }

        logger.info(
            "engine_run_complete strategy=%s bars=%d trades=%d cagr=%s sharpe=%s max_drawdown=%s turnover=%s",
            strategy_id,
            len(eq_df),
            int(len(tr)),
            metrics.get("cagr"),
            metrics.get("sharpe"),
            metrics.get("max_drawdown"),
            metrics.get("turnover"),
        )

        return BacktestResult(
            equity_curve=eq_df,
            trades=tr,
            decisions=dec,
            metrics={**metrics, "by_regime": by_regime, "time_in_regime": in_regime, "turnover_at_regime_changes": turn_reg},
            regime_stats={"performance_by_regime": by_regime, "time_in_regime": in_regime, "regime_switches": switches},
            diagnostics=diagnostics,
        )
