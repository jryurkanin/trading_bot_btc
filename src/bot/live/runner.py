from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import logging
import time

import pandas as pd

from ..coinbase_client import RESTClientWrapper
from ..config import BotConfig
from ..data.candles import CandleStore, CandleQuery, align_closed_candles
from ..execution.state_store import BotStateStore
from ..execution.risk import RiskManager, RiskState
from ..strategy.regime_switching_orchestrator import RegimeSwitchingOrchestrator
from .paper import PaperTrader
from ..execution.order_router import OrderRouter

logger = logging.getLogger("trading_bot.live")


@dataclass
class RunnerDecision:
    timestamp: pd.Timestamp
    target_fraction: float
    micro_regime: str
    strategy: str
    filled: bool
    fills: list
    reason: str


class LiveRunner:
    def __init__(
        self,
        cfg: BotConfig,
        client: Optional[RESTClientWrapper] = None,
        state_store: Optional[BotStateStore] = None,
        paper: bool = True,
        cycles: Optional[int] = None,
    ):
        self.cfg = cfg
        self.client = client or RESTClientWrapper(cfg.coinbase, cfg.data)
        self.state = state_store or BotStateStore(".trading_bot_cache/live_state.sqlite")
        self.store = CandleStore(cfg.data)
        self.orchestrator = RegimeSwitchingOrchestrator(cfg.regime)
        self.risk_mgr = RiskManager(cfg.risk)
        self.risk_state = RiskState(equity_peak=cfg.backtest.initial_equity, current_equity=cfg.backtest.initial_equity)
        self.paper_mode = paper
        self.cycles = cycles

        self.paper_trader = PaperTrader(self.state, cfg.execution.maker_bps, cfg.execution.taker_bps, cfg.execution.max_slippage_bps, cfg.execution.spread_bps)
        self.order_router = OrderRouter(self.client, cfg.execution)
        self.last_daily_update: Optional[pd.Timestamp] = None

        self.summary_path = Path(cfg.backtest.output_dir) / "daily_summary.log"
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_candles(self, product: str, timeframe: str, lookback_hours: int) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        if timeframe == "1h":
            start = now - timedelta(hours=lookback_hours)
            end = now
        else:
            start = now - timedelta(days=max(2, lookback_hours // 24))
            end = now
        df = self.store.get_candles(self.client, CandleQuery(product=product, timeframe=timeframe, start=start.to_pydatetime(), end=end.to_pydatetime(), force_refresh=False))
        return align_closed_candles(df, timeframe)

    def _reconcile(self):
        open_orders = self.state.list_open_orders()
        # production impl would query fills and cancel stale order IDs
        for row in open_orders:
            try:
                order_id = row[0]
                payload = self.client.list_orders()
                # if order gone, drop local
                if not any(isinstance(x, dict) and (x.get("client_order_id") == order_id or x.get("order_id") == order_id) for x in payload):
                    self.state.drop_open_order(order_id)
            except Exception:
                continue

    def _maybe_daily_refresh(self, now: pd.Timestamp) -> tuple[pd.DataFrame, bool]:
        if self.last_daily_update is not None and self.last_daily_update.normalize() == now.normalize():
            return pd.DataFrame(), False
        # do a fresh daily pull (lookback for initial state)
        daily_df = self._get_candles(self.cfg.data.product, "1d", lookback_hours=30 * 24)
        self.last_daily_update = now
        return daily_df, True

    def step_once(self, cycle_index: int = 0) -> RunnerDecision:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        hourly = self._get_candles(self.cfg.data.product, "1h", lookback_hours=720)

        # ensure latest daily data
        if self.last_daily_update is None or self.last_daily_update.date() != now.date():
            daily = self._get_candles(self.cfg.data.product, "1d", lookback_hours=90 * 24)
            self.last_daily_update = now
        else:
            # cache one day of daily data is enough if already loaded
            daily = self._get_candles(self.cfg.data.product, "1d", lookback_hours=90 * 24)

        if hourly.empty:
            return RunnerDecision(now, 0.0, "NEUTRAL", "none", False, [], "no_hourly_data")

        current_hourly = hourly.copy()
        latest_close = float(current_hourly["close"].iloc[-1])
        latest_high = float(current_hourly["high"].iloc[-1])
        latest_low = float(current_hourly["low"].iloc[-1])

        if self.paper_mode:
            current_equity = self.paper_trader.get_portfolio().equity(latest_close)
            # current exposure as BTC value / equity
            current_fraction = 0.0
            if current_equity > 1e-9:
                current_fraction = max(0.0, self.paper_trader.get_portfolio().btc * latest_close / current_equity)
        else:
            # from real account
            balances = self.client.get_accounts()
            quote = self.cfg.data.product.split("-")[-1]
            btc = [a.balance for a in balances if a.currency == self.cfg.data.product.split("-")[0]]
            usd = [a.balance for a in balances if a.currency == quote]
            btc_pos = float(btc[0]) if btc else 0.0
            usd_pos = float(usd[0]) if usd else 0.0
            current_equity = usd_pos + btc_pos * latest_close
            current_fraction = btc_pos * latest_close / max(current_equity, 1e-12)

        bundle = self.orchestrator.compute_target_position(now, current_hourly.reset_index(), daily.reset_index(), current_exposure=current_fraction)
        target = bundle.final_target
        target = self.risk_mgr.apply_caps(target, self.risk_state, current_hourly["timestamp"].iloc[-1].to_pydatetime(), now.to_pydatetime(), timeframe_minutes=60)

        filled = False
        fills = []

        if self.paper_mode:
            fills = self.paper_trader.execute_fraction(target, now.to_pydatetime(), latest_close, latest_high, latest_low)
            if fills:
                filled = True
        else:
            # live mode: route real orders
            orders = self.order_router.target_to_order(
                product=self.cfg.data.product,
                current_fraction=current_fraction,
                target_fraction=target,
                equity_usd=current_equity,
                price=latest_close,
                latest_bid=latest_close,
                latest_ask=latest_close,
            )
            for o in orders:
                try:
                    quote = self.order_router.place_limit_with_fallback(
                        product=o.product,
                        side=o.side,
                        size=o.size,
                        bid=latest_low,
                        ask=latest_high,
                        now=now.to_pydatetime(),
                        fallback_to_market=self.cfg.execution.fallback_to_market,
                    )
                    self.state.put_open_order(o.client_order_id or "", o.product, o.side, o.size, o.order_type, o.price, int(now.timestamp()))
                    filled = True
                    fills.append(quote)
                except Exception as exc:
                    logger.exception("order placement failed: %s", exc)

        self.risk_state.current_equity = current_equity
        if current_equity > self.risk_state.equity_peak:
            self.risk_state.equity_peak = current_equity

        decision_payload = {
            "timestamp": str(now),
            "target": float(target),
            "micro_regime": bundle.micro_regime.value,
            "macro_risk_on": bundle.macro_risk_on,
            "strategy": bundle.strategy_name,
            "reason": bundle.macro_reason,
            "metadata": bundle.metadata,
            "cycle": cycle_index,
        }
        self.state.log_decision(int(now.timestamp()), self.cfg.data.product, decision_payload)

        # daily summary file
        if now.hour == 0:
            with self.summary_path.open("a", encoding="utf-8") as f:
                f.write(f"{now.isoformat()} {bundle.micro_regime.value} target={target:.4f} strategy={bundle.strategy_name}\n")

        return RunnerDecision(
            timestamp=now,
            target_fraction=float(target),
            micro_regime=bundle.micro_regime.value,
            strategy=bundle.strategy_name,
            filled=filled,
            fills=fills,
            reason=bundle.macro_reason,
        )

    def run(self):
        cycles = self.cycles
        current = 0
        logger.info("Starting live runner (paper=%s)", self.paper_mode)
        while cycles is None or current < cycles:
            self._reconcile()
            try:
                decision = self.step_once(current)
            except Exception:
                logger.exception("run cycle failed at %d", current)
            else:
                logger.info("cycle=%d target=%s regime=%s strategy=%s filled=%s", current, decision.target_fraction, decision.micro_regime, decision.strategy, decision.filled)
            current += 1

            # wait until next hour edge
            if cycles is not None and current >= cycles:
                break
            now = datetime.now(tz=timezone.utc)
            next_hour = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
            delay = (next_hour - now).total_seconds()
            if delay > 0:
                if cycles is not None:
                    time.sleep(min(delay, 1.0))
                else:
                    time.sleep(delay)
