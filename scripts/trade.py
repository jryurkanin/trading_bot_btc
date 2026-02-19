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
from bot.system_log import setup_system_logger, get_system_logger

logger = get_system_logger("scripts.trade")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="paper mode")
    mode.add_argument("--live", action="store_true", help="live exchange mode")
    p.add_argument("--product", default="BTC-USD")
    p.add_argument("--config", default=None)
    p.add_argument("--cycles", type=int, default=None, help="number of loops to run")
    p.add_argument("--sandbox", action="store_true")
    p.add_argument(
        "--log-path",
        default=None,
        help="optional system log path (default: BOT_SYSTEM_LOG_PATH or system_log.log)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_path = setup_system_logger(args.log_path)
    logger.info("trade_start log_path=%s args=%s", log_path, vars(args))

    try:
        cfg = BotConfig.load(args.config)
        cfg.data.product = args.product
        cfg.coinbase.use_sandbox = bool(args.sandbox) if args.sandbox else cfg.coinbase.use_sandbox

        if args.live and not args.paper:
            if not cfg.coinbase.api_key or not cfg.coinbase.api_secret:
                raise RuntimeError("Live mode needs COINBASE_API_KEY/COINBASE_API_SECRET")

        client = RESTClientWrapper(cfg.coinbase, cfg.data)
        state = BotStateStore(".trading_bot_cache/live_state.sqlite")
        run_paper = True if args.paper or not args.live else False

        logger.info(
            "trade_runner_start mode=%s product=%s sandbox=%s cycles=%s strategy=%s",
            "paper" if run_paper else "live",
            cfg.data.product,
            cfg.coinbase.use_sandbox,
            args.cycles,
            cfg.backtest.strategy,
        )

        runner = LiveRunner(cfg, client=client, state_store=state, paper=run_paper, cycles=args.cycles)
        runner.run()

        logger.info("trade_complete mode=%s cycles=%s", "paper" if run_paper else "live", args.cycles)
        return 0
    except Exception:
        logger.exception("trade_failed")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
