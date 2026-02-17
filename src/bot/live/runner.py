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
from ..features.fred_features import build_fred_daily_overlay_features
from ..strategy.macro_gate_benchmark import MacroGateBenchmarkStrategy
from ..strategy.macro_only_v2 import MacroOnlyV2Strategy
from ..strategy.regime_switching_orchestrator import RegimeSwitchingOrchestrator
from ..strategy.regime_switching_v4_core import V4CoreStrategy
from ..strategy.v5_adaptive import V5AdaptiveStrategy
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
    @staticmethod
    def _build_orchestrator(cfg: BotConfig):
        strategy_id = str(getattr(cfg.backtest, "strategy", "macro_gate_benchmark") or "macro_gate_benchmark")
        if strategy_id == "macro_gate_benchmark":
            return MacroGateBenchmarkStrategy(cfg.regime)
        if strategy_id == "macro_only_v2":
            return MacroOnlyV2Strategy(cfg.regime)
        if strategy_id in {"regime_switching_v3", "regime_switching_orchestrator", "regime_switching"}:
            return RegimeSwitchingOrchestrator(cfg.regime)
        if strategy_id in {"regime_switching_v4_core", "v4_core"}:
            return V4CoreStrategy(cfg.regime)
        if strategy_id == "v5_adaptive":
            return V5AdaptiveStrategy(cfg.regime)
        raise ValueError(f"Unsupported live strategy '{strategy_id}'")

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
        self.orchestrator = self._build_orchestrator(cfg)
        self.risk_mgr = RiskManager(cfg.risk)
        self.risk_state = RiskState(equity_peak=cfg.backtest.initial_equity, current_equity=cfg.backtest.initial_equity)
        self.paper_mode = paper
        self.cycles = cycles

        self.paper_trader = PaperTrader(self.state, cfg.execution.maker_bps, cfg.execution.taker_bps, cfg.execution.max_slippage_bps, cfg.execution.spread_bps)
        self.order_router = OrderRouter(self.client, cfg.execution)
        self.last_daily_update: Optional[pd.Timestamp] = None
        self._daily_features_cache: Optional[pd.DataFrame] = None
        self._daily_features_day: Optional[pd.Timestamp] = None
        self._last_fred_report: dict[str, Any] = {
            "enabled": bool(getattr(cfg.fred, "enabled", False)),
            "warnings": [],
        }

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
        if isinstance(saved_orchestrator, dict) and hasattr(self.orchestrator, "load_runtime_state"):
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
            CandleQuery(
                product=product,
                timeframe=timeframe,
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                force_refresh=bool(getattr(self.cfg.data, "force_refresh", False)),
            ),
        )
        return align_closed_candles(df, timeframe)

    def _apply_live_fred_overlay(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        if daily_df is None or daily_df.empty or not bool(getattr(self.cfg.fred, "enabled", False)):
            self._last_fred_report = {
                "enabled": bool(getattr(self.cfg.fred, "enabled", False)),
                "warnings": [],
            }
            return daily_df

        try:
            fred_build = build_fred_daily_overlay_features(daily_df, self.cfg.fred)
            self._last_fred_report = dict(fred_build.report)
            return fred_build.daily_features
        except Exception as exc:
            logger.warning("live FRED overlay build failed; continuing without FRED features: %s", exc)
            self._last_fred_report = {
                "enabled": True,
                "series_used": [],
                "warnings": [f"fred_overlay_failed:{exc.__class__.__name__}"],
            }
            return daily_df

    def _get_daily_features(self, product: str, lookback_days: int, now: pd.Timestamp) -> pd.DataFrame:
        force_refresh = bool(getattr(self.cfg.data, "force_refresh", False))
        refresh_needed = (
            force_refresh
            or self._daily_features_cache is None
            or self._daily_features_day is None
            or self._daily_features_day.date() != now.date()
        )

        if refresh_needed:
            daily_raw = self._get_candles(product, "1d", lookback_hours=lookback_days * 24)
            daily_features = self._apply_live_fred_overlay(daily_raw)
            self._daily_features_cache = daily_features
            self._daily_features_day = now
            self.last_daily_update = now

        if self._daily_features_cache is None:
            self._daily_features_cache = self._get_candles(product, "1d", lookback_hours=lookback_days * 24)
            self._daily_features_day = now
            self.last_daily_update = now

        return self._daily_features_cache.copy()

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
            metadata = dict(local.get("metadata") or {})

            remote = remote_map.get(order_id)
            if remote is None:
                missing_count = int(metadata.get("missing_count") or 0) + 1
                metadata["missing_count"] = missing_count
                self.state.update_open_order(order_id, metadata=metadata, ts=now_ts)
                self.state.log_order_event(
                    order_id,
                    "missing_on_exchange",
                    {"missing_count": missing_count},
                    ts=now_ts,
                )

                age_s = max(0, now_ts - created_at)
                if missing_count >= 3 and age_s >= timeout_s:
                    self.state.log_order_event(
                        order_id,
                        "drop_after_missing_threshold",
                        {"missing_count": missing_count, "age_s": age_s},
                        ts=now_ts,
                    )
                    self.state.drop_open_order(order_id)
                continue

            if "missing_count" in metadata:
                metadata.pop("missing_count", None)

            status = self._order_status(remote)
            filled_size = self._order_filled_size(remote)
            self.state.update_open_order(
                order_id,
                status=status,
                filled_size=filled_size,
                metadata=metadata,
                ts=now_ts,
            )

            terminal = status in {"FILLED", "DONE", "COMPLETED", "CANCELLED", "CANCELED", "FAILED", "EXPIRED", "REJECTED"}
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
            cancel_ok = self.order_router.cancel_order(order_id)
            if not cancel_ok:
                self.state.log_order_event(order_id, "cancel_timeout_failed", {"age_s": age_s, "status": status}, ts=now_ts)
                self.consecutive_order_failures += 1
                continue

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
                self.state.drop_open_order(order_id)
            except Exception as exc:
                self.state.log_order_event(order_id, "replace_failed", {"error": str(exc), "remaining": remaining}, ts=now_ts)
                self.consecutive_order_failures += 1
                # Original order has already been canceled successfully.
                self.state.drop_open_order(order_id)

    def step_once(self, cycle_index: int = 0) -> RunnerDecision:
        now = pd.Timestamp.now(tz="UTC").floor("h")
        # Need enough hourly history for realized vol and regime indicators,
        # and enough daily history for SMA200 (200d), momentum (365d), and
        # FRED z-score (252d) warmup.
        daily_lookback_days = max(
            400,
            int(getattr(self.cfg.regime, "mom_12m_days", 365) or 365) + 30,
            int(getattr(self.cfg.regime, "vol_lookback_days", 365) or 365) + 30,
            int(getattr(self.cfg.fred, "daily_z_lookback", 252) or 252) + 30,
        )
        hourly_lookback_hours = max(720, daily_lookback_days * 24)

        hourly = self._get_candles(self.cfg.data.product, "1h", lookback_hours=hourly_lookback_hours)

        if hourly.empty:
            self._alert("no_hourly_data", {"cycle": cycle_index})
            return RunnerDecision(now, 0.0, "NEUTRAL", "none", False, [], "no_hourly_data")

        latest_ts = pd.Timestamp(hourly["timestamp"].iloc[-1])
        age_min = (now - latest_ts).total_seconds() / 60.0
        if age_min > self.cfg.runtime.stale_feed_alert_minutes:
            self._alert("stale_hourly_feed", {"age_minutes": age_min})

        # Daily data (with optional FRED overlay) refreshed at daily cadence.
        daily = self._get_daily_features(self.cfg.data.product, daily_lookback_days, now)

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
            best_bid = latest_close
            best_ask = latest_close
            try:
                bbo = self.client.get_best_bid_ask(self.cfg.data.product)
                bid_raw = float(getattr(bbo, "bid", 0.0) or 0.0)
                ask_raw = float(getattr(bbo, "ask", 0.0) or 0.0)
                if bid_raw > 0:
                    best_bid = bid_raw
                if ask_raw > 0:
                    best_ask = ask_raw
                if best_bid > 0 and best_ask > 0 and best_ask < best_bid:
                    best_bid, best_ask = min(best_bid, best_ask), max(best_bid, best_ask)
            except Exception as exc:
                logger.warning("best_bid_ask fetch failed; using close fallback: %s", exc)

            orders = self.order_router.target_to_order(
                product=self.cfg.data.product,
                current_fraction=current_fraction,
                target_fraction=target,
                equity_usd=current_equity,
                price=latest_close,
                latest_bid=best_bid,
                latest_ask=best_ask,
            )
            for o in orders:
                try:
                    if self.cfg.execution.maker_first:
                        quote = self.order_router.place_maker_first(
                            product=o.product,
                            side=o.side,
                            size=o.size,
                            now=now.to_pydatetime(),
                        )
                    else:
                        quote = self.order_router.place_limit_with_fallback(
                            product=o.product,
                            side=o.side,
                            size=o.size,
                            bid=best_bid,
                            ask=best_ask,
                            now=now.to_pydatetime(),
                            fallback_to_market=self.cfg.execution.fallback_to_market,
                            client_order_id=o.client_order_id,
                        )

                    mode = str(quote.get("mode", "")).lower()
                    order_id = str(quote.get("order_id") or o.client_order_id)
                    submitted_size = float(quote.get("submitted_size", o.size) or o.size)

                    if mode == "maker_unfilled":
                        self.state.log_order_event(order_id, "maker_unfilled", {"quote": quote}, ts=int(now.timestamp()))
                        continue

                    if mode == "maker_limit":
                        self.state.log_order_event(order_id, "maker_limit_filled", {"quote": quote}, ts=int(now.timestamp()))
                        filled = True
                        fills.append(quote)
                        self.consecutive_order_failures = 0
                        continue

                    order_type = "market" if mode in {"market", "taker_market_fallback"} else o.order_type
                    order_price = float(quote.get("limit_price", o.price)) if o.price is not None else None

                    self.state.put_open_order(
                        order_id,
                        o.product,
                        o.side,
                        submitted_size,
                        order_type,
                        order_price,
                        int(now.timestamp()),
                        status="submitted",
                        filled_size=0.0,
                        metadata={"mode": mode or "unknown"},
                    )
                    self.state.log_order_event(order_id, "submitted", {"quote": quote}, ts=int(now.timestamp()))
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
        if hasattr(self.orchestrator, "runtime_state"):
            self.state.set_kv("orchestrator_runtime", self.orchestrator.runtime_state())
        else:
            self.state.set_kv("orchestrator_runtime", {})

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
            "fred": {
                "enabled": bool(self._last_fred_report.get("enabled", False)),
                "warnings": list(self._last_fred_report.get("warnings", []) or [])[:3],
            },
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
                "maker_first": bool(self.cfg.execution.maker_first),
                "fred_enabled": bool(self._last_fred_report.get("enabled", False)),
                "fred_warnings": list(self._last_fred_report.get("warnings", []) or [])[:3],
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

    def _wait_until_next_cycle(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return

        # Test/CLI finite-cycle mode keeps short sleeps for fast feedback.
        if self.cycles is not None:
            time.sleep(min(delay_seconds, 1.0))
            return

        # In continuous mode, poll reconcile while waiting so timeout-based
        # cancel/replace is enforced near configured order_timeout_s.
        timeout_s = max(1.0, float(self.cfg.execution.order_timeout_s))
        poll_interval = max(5.0, min(30.0, timeout_s / 4.0))

        deadline = time.time() + delay_seconds
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            sleep_for = min(poll_interval, remaining)
            if sleep_for > 0:
                time.sleep(sleep_for)
            try:
                self._reconcile()
            except Exception:
                logger.exception("reconcile polling failed during wait")

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
                self._wait_until_next_cycle(delay)
