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
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Backtest completed")
    print(json.dumps({"metrics": report["metrics"]}, indent=2))
    print(f"Equity curve: {out / 'equity_curve.csv'}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
