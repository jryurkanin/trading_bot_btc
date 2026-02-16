from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
import json
import logging
import time

import pandas as pd

from ..coinbase_client import RESTClientWrapper
from ..config import BotConfig
from ..data.candles import CandleStore, CandleQuery, align_closed_candles
from ..execution.state_store import BotStateStore
from ..execution.risk import RiskManager, RiskState
from ..strategy.macro_gate_benchmark import MacroGateBenchmarkStrategy
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
        self.orchestrator = MacroGateBenchmarkStrategy(cfg.regime)
        self.risk_mgr = RiskManager(cfg.risk)
        self.risk_state = RiskState(equity_peak=cfg.backtest.initial_equity, current_equity=cfg.backtest.initial_equity)
        self.paper_mode = paper
        self.cycles = cycles

        self.paper_trader = PaperTrader(self.state, cfg.execution.maker_bps, cfg.execution.taker_bps, cfg.execution.max_slippage_bps, cfg.execution.spread_bps)
        self.order_router = OrderRouter(self.client, cfg.execution)
        self.last_daily_update: Optional[pd.Timestamp] = None

        self.summary_path = Path(cfg.backtest.output_dir) / "daily_summary.log"
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

        self.health_path = Path(cfg.runtime.healthcheck_file)
        self.health_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_path = Path(cfg.runtime.alert_file)
        self.alert_path.parent.mkdir(parents=True, exist_ok=True)

        self.consecutive_cycle_failures = 0
        self.consecutive_order_failures = 0

        saved_risk = self.state.get_kv("risk_runtime", {})
        if isinstance(saved_risk, dict):
            self.risk_state.equity_peak = float(saved_risk.get("equity_peak", self.risk_state.equity_peak))
            self.risk_state.current_equity = float(saved_risk.get("current_equity", self.risk_state.current_equity))
            self.risk_state.day_start_equity = float(saved_risk.get("day_start_equity", self.risk_state.day_start_equity))
            self.risk_state.consecutive_losses = int(saved_risk.get("consecutive_losses", 0) or 0)
            anchor_raw = saved_risk.get("day_anchor")
            if anchor_raw:
                try:
                    self.risk_state.day_anchor = pd.Timestamp(anchor_raw)
                except Exception:
                    self.risk_state.day_anchor = None

        saved_orchestrator = self.state.get_kv("orchestrator_runtime", {})
        if isinstance(saved_orchestrator, dict):
            self.orchestrator.load_runtime_state(saved_orchestrator)

    def _write_health(self, status: str, payload: dict[str, Any]) -> None:
        doc = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "status": status,
            "paper_mode": self.paper_mode,
            "consecutive_cycle_failures": self.consecutive_cycle_failures,
            "consecutive_order_failures": self.consecutive_order_failures,
            "payload": payload,
        }
        self.health_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    def _alert(self, message: str, payload: Optional[dict[str, Any]] = None) -> None:
        payload = payload or {}
        line = json.dumps(
            {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "message": message,
                "payload": payload,
            }
        )
        with self.alert_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.error("ALERT: %s | %s", message, payload)

    def _get_candles(self, product: str, timeframe: str, lookback_hours: int) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        if timeframe == "1h":
            start = now - timedelta(hours=lookback_hours)
            end = now
        else:
            start = now - timedelta(days=max(2, lookback_hours // 24))
            end = now
        df = self.store.get_candles(
            self.client,
            CandleQuery(product=product, timeframe=timeframe, start=start.to_pydatetime(), end=end.to_pydatetime(), force_refresh=False),
        )
        return align_closed_candles(df, timeframe)

    @staticmethod
    def _order_status(row: dict[str, Any]) -> str:
        for key in ["status", "order_status", "state"]:
            v = row.get(key)
            if v is not None:
                return str(v).upper()
        return "UNKNOWN"

    @staticmethod
    def _order_filled_size(row: dict[str, Any]) -> float:
        for key in ["filled_size", "filled_base_size", "filled_quantity", "filled_amount"]:
            if key in row and row[key] is not None:
                try:
                    return float(row[key])
                except Exception:
                    pass
        completion = row.get("completion_percentage")
        size = row.get("size") or row.get("base_size")
        if completion is not None and size is not None:
            try:
                pct = float(completion) / 100.0
                return max(0.0, pct * float(size))
            except Exception:
                pass
        return 0.0

    def _reconcile(self):
        open_orders = self.state.list_open_orders_dict()
        if not open_orders:
            return

        try:
            remote_orders = self.client.list_orders(product_id=self.cfg.data.product)
        except Exception as exc:
            logger.exception("reconcile failed listing orders: %s", exc)
            return

        remote_map: dict[str, dict[str, Any]] = {}
        for row in remote_orders:
            if not isinstance(row, dict):
                continue
            cid = row.get("client_order_id") or row.get("clientOrderId")
            oid = row.get("order_id") or row.get("id")
            if cid:
                remote_map[str(cid)] = row
            if oid:
                remote_map[str(oid)] = row

        now_ts = int(time.time())
        timeout_s = int(self.cfg.execution.order_timeout_s)

        for local in open_orders:
            order_id = str(local["client_order_id"])
            local_size = float(local.get("size") or 0.0)
            created_at = int(local.get("created_at") or now_ts)
            replace_count = int(local.get("replace_count") or 0)
            side = str(local.get("side") or "BUY")
            product = str(local.get("product") or self.cfg.data.product)

            remote = remote_map.get(order_id)
            if remote is None:
                self.state.log_order_event(order_id, "missing_on_exchange", {"action": "drop_local"}, ts=now_ts)
                self.state.drop_open_order(order_id)
                continue

            status = self._order_status(remote)
            filled_size = self._order_filled_size(remote)
            self.state.update_open_order(order_id, status=status, filled_size=filled_size, ts=now_ts)

            terminal = status in {"FILLED", "CANCELLED", "CANCELED", "FAILED", "EXPIRED", "REJECTED"}
            if filled_size > 0 and filled_size < local_size:
                self.state.log_order_event(
                    order_id,
                    "partial_fill",
                    {"filled_size": filled_size, "size": local_size, "status": status},
                    ts=now_ts,
                )

            if terminal:
                self.state.log_order_event(order_id, "terminal", {"status": status, "filled_size": filled_size}, ts=now_ts)
                self.state.drop_open_order(order_id)
                continue

            age_s = now_ts - created_at
            if not self.cfg.execution.cancel_replace_on_timeout or age_s < timeout_s:
                continue

            # deterministic cancel/replace on timeout
            self.order_router.cancel_order(order_id)
            self.state.log_order_event(order_id, "cancel_timeout", {"age_s": age_s, "status": status}, ts=now_ts)

            remaining = max(0.0, local_size - filled_size)
            if remaining <= 0 or not self.cfg.execution.replace_with_market_on_timeout:
                self.state.drop_open_order(order_id)
                continue

            replace_id = self.order_router.make_order_id(product, side, remaining, datetime.now(tz=timezone.utc))
            try:
                self.client.create_order(
                    product_id=product,
                    side=side,
                    size=f"{remaining:.8f}",
                    client_order_id=replace_id,
                    order_type="market",
                )
                self.state.put_open_order(
                    replace_id,
                    product,
                    side,
                    remaining,
                    "market",
                    None,
                    now_ts,
                    status="submitted",
                    filled_size=0.0,
                    replace_count=replace_count + 1,
                    metadata={"replaces": order_id},
                )
                self.state.log_order_event(order_id, "replaced_with_market", {"replacement_id": replace_id, "remaining": remaining}, ts=now_ts)
            except Exception as exc:
                self.state.log_order_event(order_id, "replace_failed", {"error": str(exc), "remaining": remaining}, ts=now_ts)
                self.consecutive_order_failures += 1
            finally:
                self.state.drop_open_order(order_id)

    def step_once(self, cycle_index: int = 0) -> RunnerDecision:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        hourly = self._get_candles(self.cfg.data.product, "1h", lookback_hours=720)

        if hourly.empty:
            self._alert("no_hourly_data", {"cycle": cycle_index})
            return RunnerDecision(now, 0.0, "NEUTRAL", "none", False, [], "no_hourly_data")

        latest_ts = pd.Timestamp(hourly["timestamp"].iloc[-1])
        age_min = (now - latest_ts).total_seconds() / 60.0
        if age_min > self.cfg.runtime.stale_feed_alert_minutes:
            self._alert("stale_hourly_feed", {"age_minutes": age_min})

        # ensure latest daily data
        if self.last_daily_update is None or self.last_daily_update.date() != now.date():
            daily = self._get_candles(self.cfg.data.product, "1d", lookback_hours=90 * 24)
            self.last_daily_update = now
        else:
            daily = self._get_candles(self.cfg.data.product, "1d", lookback_hours=90 * 24)

        current_hourly = hourly.copy()
        latest_close = float(current_hourly["close"].iloc[-1])
        latest_high = float(current_hourly["high"].iloc[-1])
        latest_low = float(current_hourly["low"].iloc[-1])

        if self.paper_mode:
            current_equity = self.paper_trader.get_portfolio().equity(latest_close)
            current_fraction = 0.0
            if current_equity > 1e-9:
                current_fraction = max(0.0, self.paper_trader.get_portfolio().btc * latest_close / current_equity)
        else:
            balances = self.client.get_accounts()
            quote = self.cfg.data.product.split("-")[-1]
            btc = [a.balance for a in balances if a.currency == self.cfg.data.product.split("-")[0]]
            usd = [a.balance for a in balances if a.currency == quote]
            btc_pos = float(btc[0]) if btc else 0.0
            usd_pos = float(usd[0]) if usd else 0.0
            current_equity = usd_pos + btc_pos * latest_close
            current_fraction = btc_pos * latest_close / max(current_equity, 1e-12)

        self.risk_mgr.update_runtime_state(self.risk_state, current_equity, now)

        bundle = self.orchestrator.compute_target_position(now, current_hourly.reset_index(), daily.reset_index(), current_exposure=current_fraction)
        target = bundle.final_target
        target = self.risk_mgr.apply_caps(
            target,
            self.risk_state,
            current_hourly["timestamp"].iloc[-1].to_pydatetime(),
            now.to_pydatetime(),
            timeframe_minutes=60,
            current_fraction=current_fraction,
        )

        kill_reason = self.risk_mgr.kill_switch_reason(self.risk_state)

        filled = False
        fills = []

        if self.paper_mode:
            fills = self.paper_trader.execute_fraction(target, now.to_pydatetime(), latest_close, latest_high, latest_low)
            if fills:
                filled = True
                self.consecutive_order_failures = 0
        else:
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
                        client_order_id=o.client_order_id,
                    )
                    self.state.put_open_order(
                        o.client_order_id,
                        o.product,
                        o.side,
                        quote.get("submitted_size", o.size),
                        o.order_type,
                        o.price,
                        int(now.timestamp()),
                        status="submitted",
                        filled_size=0.0,
                        metadata={"mode": quote.get("mode", "limit")},
                    )
                    self.state.log_order_event(o.client_order_id, "submitted", {"quote": quote}, ts=int(now.timestamp()))
                    filled = True
                    fills.append(quote)
                    self.consecutive_order_failures = 0
                except Exception as exc:
                    logger.exception("order placement failed: %s", exc)
                    self.consecutive_order_failures += 1

        if self.consecutive_order_failures >= self.cfg.runtime.max_consecutive_order_failures:
            self._alert("consecutive_order_failures", {"count": self.consecutive_order_failures})

        self.risk_state.current_equity = current_equity
        if current_equity > self.risk_state.equity_peak:
            self.risk_state.equity_peak = current_equity

        self.state.set_kv(
            "risk_runtime",
            {
                "equity_peak": self.risk_state.equity_peak,
                "current_equity": self.risk_state.current_equity,
                "day_start_equity": self.risk_state.day_start_equity,
                "day_anchor": str(self.risk_state.day_anchor) if self.risk_state.day_anchor is not None else None,
                "consecutive_losses": self.risk_state.consecutive_losses,
                "kill_switch_reason": kill_reason,
            },
        )
        self.state.set_kv("orchestrator_runtime", self.orchestrator.runtime_state())

        decision_payload = {
            "timestamp": str(now),
            "target": float(target),
            "micro_regime": bundle.micro_regime.value,
            "macro_risk_on": bundle.macro_risk_on,
            "strategy": bundle.strategy_name,
            "reason": bundle.macro_reason,
            "metadata": bundle.metadata,
            "cycle": cycle_index,
            "kill_switch_reason": kill_reason,
        }
        self.state.log_decision(int(now.timestamp()), self.cfg.data.product, decision_payload)

        if now.hour == 0:
            with self.summary_path.open("a", encoding="utf-8") as f:
                f.write(f"{now.isoformat()} {bundle.micro_regime.value} target={target:.4f} strategy={bundle.strategy_name}\n")

        self._write_health(
            "ok",
            {
                "cycle": cycle_index,
                "target": target,
                "regime": bundle.micro_regime.value,
                "kill_switch_reason": kill_reason,
            },
        )

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
            except Exception as exc:
                self.consecutive_cycle_failures += 1
                logger.exception("run cycle failed at %d", current)
                self._write_health("error", {"cycle": current, "error": str(exc)})
                if self.consecutive_cycle_failures >= self.cfg.runtime.max_consecutive_cycle_failures:
                    self._alert("consecutive_cycle_failures", {"count": self.consecutive_cycle_failures, "cycle": current})
            else:
                self.consecutive_cycle_failures = 0
                logger.info(
                    "cycle=%d target=%s regime=%s strategy=%s filled=%s",
                    current,
                    decision.target_fraction,
                    decision.micro_regime,
                    decision.strategy,
                    decision.filled,
                )
            current += 1

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
