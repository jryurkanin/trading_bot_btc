#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Make local src discoverable when running directly from the repository root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.config import BotConfig
from bot.coinbase_client import RESTClientWrapper
from bot.data.candles import CandleQuery, CandleStore
from bot.backtest.engine import BacktestEngine
from bot.analysis.pnl_decomposition import run_pnl_decomposition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tf", default="1h", choices=["1h", "1d"])
    p.add_argument("--strategy", default="regime_switching")
    p.add_argument("--config", default=None, help="Path to JSON/TOML/YAML config")
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
    p.add_argument("--ci-mode", action="store_true")
    p.add_argument("--no-spread", action="store_true")
    p.add_argument("--output", default="reports")
    return p.parse_args()


def parse_ts(raw: str) -> datetime:
    ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def main() -> int:
    args = parse_args()
    cfg = BotConfig.load(args.config)
    cfg.data.product = args.product
    cfg.backtest.initial_equity = args.initial_equity

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
    if args.ci_mode:
        cfg.backtest.ci_mode = True

    store = CandleStore(cfg.data)
    start = parse_ts(args.start)
    end = parse_ts(args.end)

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
        query=CandleQuery(product=args.product, timeframe=hourly_tf, start=start, end=end),
    )
    # keep a daily context for regime and risk overlays
    daily = store.get_candles(
        client=client,
        query=CandleQuery(product=args.product, timeframe="1d", start=start, end=end),
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
    )
    result = engine.run()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(out / "equity_curve.csv", index=True)
    result.trades.to_csv(out / "trades.csv", index=False)
    result.decisions.to_csv(out / "decisions.csv", index=True)

    report = {
        "product": args.product,
        "start": args.start,
        "end": args.end,
        "strategy": args.strategy,
        "metrics": result.metrics,
        "regime_metrics": result.regime_stats,
        "diagnostics": result.diagnostics,
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
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    execution_quality = run_pnl_decomposition(
        out / "trades.csv",
        out / "equity_curve.csv",
        out,
        min_trade_notional_usd=cfg.execution.min_trade_notional_usd,
        max_allowed_slippage_bps=cfg.execution.max_allowed_slippage_bps,
        ci_mode=cfg.backtest.ci_mode,
    )

    print("Backtest completed")
    print(json.dumps({"metrics": report["metrics"], "execution_quality": execution_quality}, indent=2))
    print(f"Equity curve: {out / 'equity_curve.csv'}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
