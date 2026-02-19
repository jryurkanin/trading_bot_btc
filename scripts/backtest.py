#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

import pandas as pd

# Make local src discoverable when running directly from the repository root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.config import BotConfig, BacktestConfig
from bot.coinbase_client import RESTClientWrapper
from bot.data.candles import CandleQuery, CandleStore
from bot.backtest.engine import BacktestEngine
from bot.backtest.reporting import write_strict_json, dumps_strict_json
from bot.backtest.macro_attribution import compute_macro_bucket_attribution
from bot.analysis.pnl_decomposition import run_pnl_decomposition
from bot.system_log import setup_system_logger, get_system_logger


logger = get_system_logger("scripts.backtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tf", default="1h", choices=["1h", "1d"])
    p.add_argument(
        "--strategy",
        default="macro_gate_benchmark",
        choices=sorted(BacktestConfig.VALID_STRATEGIES),
    )
    p.add_argument("--config", default=None, help="Path to JSON/TOML/YAML config")
    p.add_argument("--acceleration-backend", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--fred-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--fred-max-risk-off-penalty", type=float, default=None)
    p.add_argument("--fred-risk-off-score-ema-span", type=int, default=None)
    p.add_argument("--fred-lag-stress-multiplier", type=float, default=None)
    p.add_argument("--fred-realtime-mode", choices=["lagged_latest", "vintage_dates"], default=None)
    p.add_argument("--initial-equity", type=float, default=10_000.0)
    p.add_argument("--maker-bps", type=float, default=10.0)
    p.add_argument("--taker-bps", type=float, default=25.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--impact-bps", type=float, default=None)
    p.add_argument("--fill-model", choices=["next_open", "bid_ask", "worst_case_bar"], default=None)
    p.add_argument("--rebalance-policy", choices=["signal_change_only", "band", "always"], default=None)
    p.add_argument("--min-trade-notional-usd", type=float, default=None)
    p.add_argument("--min-exposure-delta", type=float, default=None)
    p.add_argument("--target-quantization-step", type=float, default=None)
    p.add_argument("--min-time-between-trades-hours", type=float, default=None)
    p.add_argument("--max-trades-per-day", type=int, default=None)

    # macro scoring + trend boost (all optional, backward-compatible)
    p.add_argument("--macro-mode", choices=["binary", "score", "stateful_gate"], default=None)
    p.add_argument("--macro-score-transform", choices=["linear", "piecewise"], default=None)
    p.add_argument("--macro-score-floor", type=float, default=None)
    p.add_argument("--macro-score-min-to-trade", type=float, default=None)
    p.add_argument("--macro-enter-threshold", type=float, default=None)
    p.add_argument("--macro-exit-threshold", type=float, default=None)
    p.add_argument("--macro-full-threshold", type=float, default=None)
    p.add_argument("--macro-half-threshold", type=float, default=None)
    p.add_argument("--macro-confirm-days", type=int, default=None)
    p.add_argument("--macro-min-on-days", type=int, default=None)
    p.add_argument("--macro-min-off-days", type=int, default=None)
    p.add_argument("--macro-half-multiplier", type=float, default=None)
    p.add_argument("--macro-full-multiplier", type=float, default=None)

    # macro_only_v2 controls (optional overrides)
    p.add_argument("--macro2-signal-mode", choices=["sma200_band", "mom_6_12", "sma200_and_mom", "sma200_or_mom", "score4_legacy"], default=None)
    p.add_argument("--macro2-confirm-days", type=int, default=None)
    p.add_argument("--macro2-min-on-days", type=int, default=None)
    p.add_argument("--macro2-min-off-days", type=int, default=None)
    p.add_argument("--macro2-weight-off", type=float, default=None)
    p.add_argument("--macro2-weight-half", type=float, default=None)
    p.add_argument("--macro2-weight-full", type=float, default=None)
    p.add_argument("--macro2-vol-mode", choices=["none", "inverse_vol"], default=None)
    p.add_argument("--macro2-vol-lookback-days", type=int, default=None)
    p.add_argument("--macro2-vol-floor", type=float, default=None)
    p.add_argument("--macro2-target-ann-vol-half", type=float, default=None)
    p.add_argument("--macro2-target-ann-vol-full", type=float, default=None)
    p.add_argument("--macro2-dd-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--macro2-dd-threshold", type=float, default=None)
    p.add_argument("--macro2-dd-cooldown-days", type=int, default=None)
    p.add_argument("--macro2-dd-reentry-confirm-days", type=int, default=None)
    p.add_argument("--macro2-dd-safe-weight", type=float, default=None)
    p.add_argument("--macro2-sma200-entry-band", type=float, default=None)
    p.add_argument("--macro2-sma200-exit-band", type=float, default=None)
    p.add_argument("--macro2-mom-6m-days", type=int, default=None)
    p.add_argument("--macro2-mom-12m-days", type=int, default=None)

    p.add_argument("--trend-boost-enabled", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--trend-boost-multiplier", type=float, default=None)
    p.add_argument("--trend-boost-adx-threshold", type=float, default=None)
    p.add_argument("--trend-boost-macro-score-threshold", type=float, default=None)
    p.add_argument("--trend-boost-confirm-days", type=int, default=None)
    p.add_argument("--trend-boost-min-on-days", type=int, default=None)
    p.add_argument("--trend-boost-min-off-days", type=int, default=None)
    p.add_argument("--trend-boost-require-micro-trend", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--trend-boost-require-above-sma200", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--trend-boost-sma50-slope-lookback-days", type=int, default=None)


    p.add_argument("--ci-mode", action="store_true")
    p.add_argument("--no-spread", action="store_true")
    p.add_argument("--output", default="reports")
    return p.parse_args()


def parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _prefetch_start(start: datetime, cfg: BotConfig) -> datetime:
    # Keep enough history for no-lookahead warmup of daily and hourly rolling
    # features (macro momentum/SMA, vol quantiles, and optional FRED z-scores).
    warmup_days = max(
        400,
        int(getattr(cfg.regime, "mom_12m_days", 365) or 365) + 30,
        int(getattr(cfg.regime, "vol_lookback_days", 365) or 365) + 30,
        int(getattr(cfg.fred, "daily_z_lookback", 252) or 252) + 30,
    )
    return start - timedelta(days=warmup_days)


def main() -> int:
    args = parse_args()
    log_path = setup_system_logger()
    logger.info("backtest_start log_path=%s args=%s", log_path, vars(args))

    cfg = BotConfig.load(args.config)
    cfg.data.product = args.product
    cfg.backtest.initial_equity = args.initial_equity
    cfg.backtest.strategy = args.strategy
    if args.acceleration_backend is not None:
        cfg.backtest.acceleration_backend = args.acceleration_backend

    if args.fred_enabled is not None:
        cfg.fred.enabled = bool(args.fred_enabled)
    if args.fred_max_risk_off_penalty is not None:
        cfg.fred.max_risk_off_penalty = float(args.fred_max_risk_off_penalty)
    if args.fred_risk_off_score_ema_span is not None:
        cfg.fred.risk_off_score_ema_span = int(args.fred_risk_off_score_ema_span)
    if args.fred_lag_stress_multiplier is not None:
        cfg.fred.lag_stress_multiplier = float(args.fred_lag_stress_multiplier)
    if args.fred_realtime_mode is not None:
        cfg.fred.realtime_mode = str(args.fred_realtime_mode)

    # Keep macro benchmark policy behavior only.
    cfg.regime.trend_boost_enabled = False

    if args.fill_model:
        cfg.execution.fill_model = args.fill_model
    if args.rebalance_policy:
        cfg.execution.rebalance_policy = args.rebalance_policy
    if args.min_trade_notional_usd is not None:
        cfg.execution.min_trade_notional_usd = float(args.min_trade_notional_usd)
    if args.min_exposure_delta is not None:
        cfg.execution.min_exposure_delta = float(args.min_exposure_delta)
    if args.target_quantization_step is not None:
        cfg.execution.target_quantization_step = float(args.target_quantization_step)
    if args.min_time_between_trades_hours is not None:
        cfg.execution.min_time_between_trades_hours = float(args.min_time_between_trades_hours)
    if args.max_trades_per_day is not None:
        cfg.execution.max_trades_per_day = int(args.max_trades_per_day)
    if args.impact_bps is not None:
        cfg.execution.impact_bps = float(args.impact_bps)

    if args.macro_mode is not None:
        cfg.regime.macro_mode = args.macro_mode
    if args.macro_score_transform is not None:
        cfg.regime.macro_score_transform = args.macro_score_transform
    if args.macro_score_floor is not None:
        cfg.regime.macro_score_floor = float(args.macro_score_floor)
    if args.macro_score_min_to_trade is not None:
        cfg.regime.macro_score_min_to_trade = float(args.macro_score_min_to_trade)
    if args.macro_enter_threshold is not None:
        cfg.regime.macro_enter_threshold = float(args.macro_enter_threshold)
    if args.macro_exit_threshold is not None:
        cfg.regime.macro_exit_threshold = float(args.macro_exit_threshold)
    if args.macro_full_threshold is not None:
        cfg.regime.macro_full_threshold = float(args.macro_full_threshold)
    if args.macro_half_threshold is not None:
        cfg.regime.macro_half_threshold = float(args.macro_half_threshold)
    if args.macro_confirm_days is not None:
        cfg.regime.macro_confirm_days = int(args.macro_confirm_days)
    if args.macro_min_on_days is not None:
        cfg.regime.macro_min_on_days = int(args.macro_min_on_days)
    if args.macro_min_off_days is not None:
        cfg.regime.macro_min_off_days = int(args.macro_min_off_days)
    if args.macro_half_multiplier is not None:
        cfg.regime.macro_half_multiplier = float(args.macro_half_multiplier)
    if args.macro_full_multiplier is not None:
        cfg.regime.macro_full_multiplier = float(args.macro_full_multiplier)

    if args.macro2_signal_mode is not None:
        cfg.regime.macro2_signal_mode = args.macro2_signal_mode
    if args.macro2_confirm_days is not None:
        cfg.regime.macro2_confirm_days = int(args.macro2_confirm_days)
    if args.macro2_min_on_days is not None:
        cfg.regime.macro2_min_on_days = int(args.macro2_min_on_days)
    if args.macro2_min_off_days is not None:
        cfg.regime.macro2_min_off_days = int(args.macro2_min_off_days)
    if args.macro2_weight_off is not None:
        cfg.regime.macro2_weight_off = float(args.macro2_weight_off)
    if args.macro2_weight_half is not None:
        cfg.regime.macro2_weight_half = float(args.macro2_weight_half)
    if args.macro2_weight_full is not None:
        cfg.regime.macro2_weight_full = float(args.macro2_weight_full)
    if args.macro2_vol_mode is not None:
        cfg.regime.macro2_vol_mode = args.macro2_vol_mode
    if args.macro2_vol_lookback_days is not None:
        cfg.regime.macro2_vol_lookback_days = int(args.macro2_vol_lookback_days)
    if args.macro2_vol_floor is not None:
        cfg.regime.macro2_vol_floor = float(args.macro2_vol_floor)
    if args.macro2_target_ann_vol_half is not None:
        cfg.regime.macro2_target_ann_vol_half = float(args.macro2_target_ann_vol_half)
    if args.macro2_target_ann_vol_full is not None:
        cfg.regime.macro2_target_ann_vol_full = float(args.macro2_target_ann_vol_full)
    if args.macro2_dd_enabled is not None:
        cfg.regime.macro2_dd_enabled = bool(args.macro2_dd_enabled)
    if args.macro2_dd_threshold is not None:
        cfg.regime.macro2_dd_threshold = float(args.macro2_dd_threshold)
    if args.macro2_dd_cooldown_days is not None:
        cfg.regime.macro2_dd_cooldown_days = int(args.macro2_dd_cooldown_days)
    if args.macro2_dd_reentry_confirm_days is not None:
        cfg.regime.macro2_dd_reentry_confirm_days = int(args.macro2_dd_reentry_confirm_days)
    if args.macro2_dd_safe_weight is not None:
        cfg.regime.macro2_dd_safe_weight = float(args.macro2_dd_safe_weight)
    if args.macro2_sma200_entry_band is not None:
        cfg.regime.sma200_entry_band = float(args.macro2_sma200_entry_band)
    if args.macro2_sma200_exit_band is not None:
        cfg.regime.sma200_exit_band = float(args.macro2_sma200_exit_band)
    if args.macro2_mom_6m_days is not None:
        cfg.regime.mom_6m_days = int(args.macro2_mom_6m_days)
    if args.macro2_mom_12m_days is not None:
        cfg.regime.mom_12m_days = int(args.macro2_mom_12m_days)

    if args.trend_boost_enabled is not None:
        cfg.regime.trend_boost_enabled = bool(args.trend_boost_enabled)
    if args.trend_boost_multiplier is not None:
        cfg.regime.trend_boost_multiplier = float(args.trend_boost_multiplier)
    if args.trend_boost_adx_threshold is not None:
        cfg.regime.trend_boost_adx_threshold = float(args.trend_boost_adx_threshold)
    if args.trend_boost_macro_score_threshold is not None:
        cfg.regime.trend_boost_macro_score_threshold = float(args.trend_boost_macro_score_threshold)
    if args.trend_boost_confirm_days is not None:
        cfg.regime.trend_boost_confirm_days = int(args.trend_boost_confirm_days)
    if args.trend_boost_min_on_days is not None:
        cfg.regime.trend_boost_min_on_days = int(args.trend_boost_min_on_days)
    if args.trend_boost_min_off_days is not None:
        cfg.regime.trend_boost_min_off_days = int(args.trend_boost_min_off_days)
    if args.trend_boost_require_micro_trend is not None:
        cfg.regime.trend_boost_require_micro_trend = bool(args.trend_boost_require_micro_trend)
    if args.trend_boost_require_above_sma200 is not None:
        cfg.regime.trend_boost_require_above_sma200 = bool(args.trend_boost_require_above_sma200)
    if args.trend_boost_sma50_slope_lookback_days is not None:
        cfg.regime.trend_boost_sma50_slope_lookback_days = int(args.trend_boost_sma50_slope_lookback_days)


    if args.ci_mode:
        cfg.backtest.ci_mode = True

    store = CandleStore(cfg.data)
    start = parse_ts(args.start)
    end = parse_ts(args.end)
    prefetch_start = _prefetch_start(start, cfg)
    logger.info(
        "backtest_window product=%s strategy=%s start=%s end=%s prefetch_start=%s",
        args.product,
        args.strategy,
        start.isoformat(),
        end.isoformat(),
        prefetch_start.isoformat(),
    )

    client = RESTClientWrapper(cfg.coinbase, cfg.data)
    maker = args.maker_bps / 10000.0
    taker = args.taker_bps / 10000.0
    try:
        tx = client.get_transaction_summary(args.product)
        m = float(tx.maker_fee_rate)
        t = float(tx.taker_fee_rate)
        if m > 0:
            maker = m
        if t > 0:
            taker = t
    except Exception:
        pass

    hourly_tf = args.tf if args.tf in {"1h", "1d"} else "1h"
    hourly = store.get_candles(
        client=client,
        query=CandleQuery(product=args.product, timeframe=hourly_tf, start=prefetch_start, end=end),
    )
    # Keep long daily context for macro state and FRED z-score warmups.
    daily = store.get_candles(
        client=client,
        query=CandleQuery(product=args.product, timeframe="1d", start=prefetch_start, end=end),
    )
    logger.info(
        "backtest_candles_loaded product=%s hourly_rows=%d daily_rows=%d",
        args.product,
        len(hourly),
        len(daily),
    )

    engine = BacktestEngine(
        product=args.product,
        hourly_candles=hourly,
        daily_candles=daily,
        start=start,
        end=end,
        config=cfg.backtest,
        fees=(maker, taker),
        slippage_bps=args.slippage_bps,
        use_spread_slippage=not args.no_spread,
        regime_config=cfg.regime,
        risk_config=cfg.risk,
        execution_config=cfg.execution,
        fred_config=cfg.fred,
    )
    result = engine.run()
    logger.info(
        "backtest_complete strategy=%s trade_count=%d metrics=%s diagnostics=%s",
        args.strategy,
        len(result.trades),
        result.metrics,
        result.diagnostics,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    equity_csv = out / "equity_curve.csv"
    trades_csv = out / "trades.csv"
    decisions_csv = out / "decisions.csv"

    result.equity_curve.to_csv(equity_csv, index=True)
    result.trades.to_csv(trades_csv, index=False)
    result.decisions.to_csv(decisions_csv, index=True)

    # Re-load persisted artifacts for attribution to normalize dtypes across
    # timezone/object edge cases and keep report/csv outputs consistent.
    eq_for_attr = pd.read_csv(equity_csv, parse_dates=["timestamp"]).set_index("timestamp")

    dec_for_attr = pd.read_csv(decisions_csv)
    if "timestamp" in dec_for_attr.columns:
        dec_for_attr["timestamp"] = pd.to_datetime(dec_for_attr["timestamp"], utc=True, errors="coerce")
    if "decision_applies_at" in dec_for_attr.columns:
        dec_for_attr["decision_applies_at"] = pd.to_datetime(dec_for_attr["decision_applies_at"], utc=True, errors="coerce")
    if "timestamp" in dec_for_attr.columns:
        dec_for_attr = dec_for_attr.set_index("timestamp")

    tr_for_attr = pd.read_csv(trades_csv)
    if "ts" in tr_for_attr.columns:
        tr_for_attr["ts"] = pd.to_datetime(tr_for_attr["ts"], utc=True, errors="coerce")

    macro_bucket_report, macro_bucket_table = compute_macro_bucket_attribution(
        eq_for_attr,
        dec_for_attr,
        tr_for_attr,
        initial_equity=cfg.backtest.initial_equity,
    )
    macro_bucket_csv = out / "macro_bucket_attribution.csv"
    macro_bucket_table.to_csv(macro_bucket_csv, index=False)

    report = {
        "product": args.product,
        "start": args.start,
        "end": args.end,
        "strategy": args.strategy,
        "metrics": result.metrics,
        "regime_metrics": result.regime_stats,
        "diagnostics": result.diagnostics,
        "macro_bucket_attribution": macro_bucket_report,
        "artifacts": {
            "macro_bucket_attribution_csv": str(macro_bucket_csv),
        },
        "acceleration": {
            "requested_backend": cfg.backtest.acceleration_backend,
            "effective_backend": result.diagnostics.get("acceleration_backend"),
            "cuda_available": result.diagnostics.get("acceleration_cuda_available"),
            "device": result.diagnostics.get("acceleration_device"),
            "fallback_reason": result.diagnostics.get("acceleration_fallback_reason"),
        },
        "fred": {
            "enabled": bool(cfg.fred.enabled),
            **(result.diagnostics.get("fred", {}) if isinstance(result.diagnostics.get("fred"), dict) else {}),
        },
        "execution": {
            "fill_model": cfg.execution.fill_model,
            "rebalance_policy": cfg.execution.rebalance_policy,
            "min_trade_notional_usd": cfg.execution.min_trade_notional_usd,
            "min_exposure_delta": cfg.execution.min_exposure_delta,
            "target_quantization_step": cfg.execution.target_quantization_step,
            "min_time_between_trades_hours": cfg.execution.min_time_between_trades_hours,
            "max_trades_per_day": cfg.execution.max_trades_per_day,
            "impact_bps": cfg.execution.impact_bps,
        },
        "regime_config": {
            "macro_mode": cfg.regime.macro_mode,
            "macro_score_transform": cfg.regime.macro_score_transform,
            "macro_score_floor": cfg.regime.macro_score_floor,
            "macro_score_min_to_trade": cfg.regime.macro_score_min_to_trade,
            "macro_enter_threshold": cfg.regime.macro_enter_threshold,
            "macro_exit_threshold": cfg.regime.macro_exit_threshold,
            "macro_full_threshold": cfg.regime.macro_full_threshold,
            "macro_half_threshold": cfg.regime.macro_half_threshold,
            "macro_confirm_days": cfg.regime.macro_confirm_days,
            "macro_min_on_days": cfg.regime.macro_min_on_days,
            "macro_min_off_days": cfg.regime.macro_min_off_days,
            "macro_half_multiplier": cfg.regime.macro_half_multiplier,
            "macro_full_multiplier": cfg.regime.macro_full_multiplier,
            "trend_boost_enabled": cfg.regime.trend_boost_enabled,
            "trend_boost_multiplier": cfg.regime.trend_boost_multiplier,
            "trend_boost_adx_threshold": cfg.regime.trend_boost_adx_threshold,
            "trend_boost_macro_score_threshold": cfg.regime.trend_boost_macro_score_threshold,
            "trend_boost_confirm_days": cfg.regime.trend_boost_confirm_days,
            "trend_boost_min_on_days": cfg.regime.trend_boost_min_on_days,
            "trend_boost_min_off_days": cfg.regime.trend_boost_min_off_days,
            "trend_boost_require_micro_trend": cfg.regime.trend_boost_require_micro_trend,
            "trend_boost_require_above_sma200": cfg.regime.trend_boost_require_above_sma200,
            "trend_boost_sma50_slope_lookback_days": cfg.regime.trend_boost_sma50_slope_lookback_days,
            "macro2_signal_mode": cfg.regime.macro2_signal_mode,
            "macro2_confirm_days": cfg.regime.macro2_confirm_days,
            "macro2_min_on_days": cfg.regime.macro2_min_on_days,
            "macro2_min_off_days": cfg.regime.macro2_min_off_days,
            "macro2_weight_off": cfg.regime.macro2_weight_off,
            "macro2_weight_half": cfg.regime.macro2_weight_half,
            "macro2_weight_full": cfg.regime.macro2_weight_full,
            "macro2_vol_mode": cfg.regime.macro2_vol_mode,
            "macro2_vol_lookback_days": cfg.regime.macro2_vol_lookback_days,
            "macro2_vol_floor": cfg.regime.macro2_vol_floor,
            "macro2_target_ann_vol_half": cfg.regime.macro2_target_ann_vol_half,
            "macro2_target_ann_vol_full": cfg.regime.macro2_target_ann_vol_full,
            "macro2_dd_enabled": cfg.regime.macro2_dd_enabled,
            "macro2_dd_threshold": cfg.regime.macro2_dd_threshold,
            "macro2_dd_cooldown_days": cfg.regime.macro2_dd_cooldown_days,
            "macro2_dd_reentry_confirm_days": cfg.regime.macro2_dd_reentry_confirm_days,
            "macro2_dd_safe_weight": cfg.regime.macro2_dd_safe_weight,
            "sma200_entry_band": cfg.regime.sma200_entry_band,
            "sma200_exit_band": cfg.regime.sma200_exit_band,
            "mom_6m_days": cfg.regime.mom_6m_days,
            "mom_12m_days": cfg.regime.mom_12m_days,
        },
    }
    report_path = write_strict_json(out / "report.json", report)

    execution_quality = run_pnl_decomposition(
        out / "trades.csv",
        out / "equity_curve.csv",
        out,
        min_trade_notional_usd=cfg.execution.min_trade_notional_usd,
        max_allowed_slippage_bps=cfg.execution.max_allowed_slippage_bps,
        ci_mode=cfg.backtest.ci_mode,
    )

    print("Backtest completed")
    print(dumps_strict_json({"metrics": report["metrics"], "execution_quality": execution_quality}, indent=2))
    print(f"Equity curve: {out / 'equity_curve.csv'}")
    print(f"Macro bucket attribution: {macro_bucket_csv}")
    print(f"Report: {report_path}")
    logger.info(
        "backtest_artifacts output=%s report=%s equity=%s trades=%s decisions=%s macro_bucket=%s",
        out,
        report_path,
        equity_csv,
        trades_csv,
        decisions_csv,
        macro_bucket_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
