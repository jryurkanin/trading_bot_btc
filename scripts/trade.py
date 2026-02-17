#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make local src discoverable when running directly from repository root
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bot.config import BotConfig
from bot.coinbase_client import RESTClientWrapper
from bot.execution.state_store import BotStateStore
from bot.live.runner import LiveRunner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="paper mode")
    mode.add_argument("--live", action="store_true", help="live exchange mode")
    p.add_argument("--product", default="BTC-USD")
    p.add_argument(
        "--strategy",
        default=None,
        choices=[
            "macro_gate_benchmark",
            "macro_only_v2",
            "regime_switching_v3",
            "regime_switching_v4_core",
            "v5_adaptive",
        ],
        help="override strategy for paper/live runner",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--cycles", type=int, default=None, help="number of loops to run")
    p.add_argument("--sandbox", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = BotConfig.load(args.config)
    cfg.data.product = args.product
    if args.strategy:
        cfg.backtest.strategy = args.strategy
    cfg.coinbase.use_sandbox = bool(args.sandbox) if args.sandbox else cfg.coinbase.use_sandbox

    if args.live and not args.paper:
        if not cfg.coinbase.api_key or not cfg.coinbase.api_secret:
            raise RuntimeError("Live mode needs COINBASE_API_KEY/COINBASE_API_SECRET")

    client = RESTClientWrapper(cfg.coinbase, cfg.data)
    state = BotStateStore(".trading_bot_cache/live_state.sqlite")
    run_paper = True if args.paper or not args.live else False
    runner = LiveRunner(cfg, client=client, state_store=state, paper=run_paper, cycles=args.cycles)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
