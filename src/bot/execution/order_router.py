from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional, Tuple
import hashlib
import math
import time

from ..config import ExecutionConfig
from ..coinbase_client import ProductConstraints, RESTClientWrapper


@dataclass
class RoutedOrder:
    client_order_id: str
    side: str
    product: str
    size: float
    order_type: str
    price: Optional[float]
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: str = "submitted"


@dataclass
class SimulatedFill:
    side: str
    size: float
    price: float
    notional: float
    fee_rate: float
    is_taker: bool
    fee: float
    ts: datetime


class OrderRouter:
    def __init__(self, client: RESTClientWrapper, cfg: ExecutionConfig):
        self.client = client
        self.cfg = cfg
        self._constraints_cache: dict[str, ProductConstraints] = {}

    @staticmethod
    def _make_order_id(product: str, side: str, size: float, now: datetime) -> str:
        raw = f"{product}:{side}:{size:.8f}:{int(now.timestamp())}"
        return hashlib.md5(raw.encode()).hexdigest()[:32]

    def make_order_id(self, product: str, side: str, size: float, now: datetime) -> str:
        return self._make_order_id(product, side, size, now)

    @staticmethod
    def _quantize_down(value: float, increment: float) -> float:
        if increment <= 0:
            return value
        return math.floor(value / increment) * increment

    @staticmethod
    def _quantize_up(value: float, increment: float) -> float:
        if increment <= 0:
            return value
        return math.ceil(value / increment) * increment

    def _get_constraints(self, product: str) -> ProductConstraints:
        if product in self._constraints_cache:
            return self._constraints_cache[product]
        constraints = self.client.get_product_constraints(product)
        self._constraints_cache[product] = constraints
        return constraints

    def round_price(self, product: str, price: float, side: str = "BUY") -> float:
        px = max(0.0, float(price))
        if not self.cfg.enforce_product_constraints:
            return px
        c = self._get_constraints(product)
        # round buy prices down and sell prices up to avoid crossing unexpectedly
        if side.upper() == "BUY":
            return self._quantize_down(px, c.price_increment)
        return self._quantize_up(px, c.price_increment)

    def normalize_size(self, product: str, side: str, size: float, ref_price: float) -> Tuple[float, str]:
        size = max(0.0, float(size))
        if size <= 0:
            return 0.0, "size_non_positive"

        if not self.cfg.enforce_product_constraints:
            if size * max(ref_price, 0.0) < self.cfg.min_trade_notional_usd:
                return 0.0, "below_min_trade_notional_usd"
            return size, "ok"

        c = self._get_constraints(product)
        size = self._quantize_down(size, c.base_increment)
        if size <= 0:
            return 0.0, "below_increment"

        if size < c.base_min_size:
            size = self._quantize_up(c.base_min_size, c.base_increment)

        if c.base_max_size is not None and size > c.base_max_size:
            size = self._quantize_down(c.base_max_size, c.base_increment)

        min_notional = max(
            0.0,
            c.min_notional + self.cfg.min_notional_buffer_quote,
            c.quote_min_size,
            self.cfg.min_trade_notional_usd,
        )
        if ref_price > 0 and min_notional > 0 and size * ref_price < min_notional:
            size = self._quantize_up(min_notional / ref_price, c.base_increment)
            if c.base_max_size is not None and size > c.base_max_size:
                return 0.0, "above_max_size_for_min_notional"

        size = self._quantize_down(size, c.base_increment)
        if size <= 0:
            return 0.0, "size_zero_after_rounding"
        if size < c.base_min_size:
            return 0.0, "below_min_size"
        if ref_price > 0 and size * ref_price < min_notional:
            return 0.0, "below_min_notional"

        return size, "ok"

    @staticmethod
    def _order_status(row: dict[str, Any]) -> str:
        for k in ["status", "order_status", "state"]:
            v = row.get(k)
            if v is not None:
                return str(v).upper()
        return "UNKNOWN"

    @staticmethod
    def _is_terminal(status: str) -> bool:
        return status.upper() in {"FILLED", "CANCELLED", "CANCELED", "FAILED", "EXPIRED", "REJECTED"}

    @staticmethod
    def _is_filled(status: str) -> bool:
        return status.upper() in {"FILLED", "DONE", "COMPLETED"}

    def _poll_order(self, client_order_id: str, product: Optional[str] = None) -> Optional[dict[str, Any]]:
        rows = self.client.list_orders(product_id=product)
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("client_order_id") or row.get("clientOrderId") or "") == str(client_order_id):
                return row
        return None

    def _maker_limit_price(self, side: str, bid: float, ask: float, product: str) -> float:
        # Never cross the spread for post-only maker behavior.
        if side == "BUY":
            base = bid if bid > 0 else max(0.0, ask * (1.0 - self.cfg.limit_price_offset_bps / 10_000.0))
            px = min(base, ask) if ask > 0 else base
            return self.round_price(product, px, side="BUY")
        base = ask if ask > 0 else bid * (1.0 + self.cfg.limit_price_offset_bps / 10_000.0)
        px = max(base, bid) if bid > 0 else base
        return self.round_price(product, px, side="SELL")

    def _taker_fallback_allowed(self, expected_edge_bps: Optional[float] = None) -> bool:
        if not self.cfg.allow_taker_fallback:
            return False
        if not self.cfg.taker_fallback_only_if_edge_exceeds_cost:
            return True
        if expected_edge_bps is None:
            return False
        est_cost_bps = self.cfg.spread_bps / 2.0 + self.cfg.impact_bps + self.cfg.taker_bps
        return float(expected_edge_bps) > float(est_cost_bps)

    def place_limit_with_fallback(
        self,
        product: str,
        side: str,
        size: float,
        bid: float,
        ask: float,
        now: datetime,
        fallback_to_market: bool = True,
        client_order_id: Optional[str] = None,
    ) -> dict:
        mid = ((bid + ask) / 2.0) if bid > 0 and ask > 0 else max(bid, ask)
        norm_size, reason = self.normalize_size(product, side, size, ref_price=mid)
        if norm_size <= 0:
            raise ValueError(f"invalid_order_size: {reason}")

        client_order_id = client_order_id or self._make_order_id(product, side, norm_size, now)
        size_s = f"{norm_size:.8f}"

        order_price = self._maker_limit_price(side, bid, ask, product)

        try:
            resp = self.client.create_order(
                product_id=product,
                side=side,
                size=size_s,
                client_order_id=client_order_id,
                order_type="limit",
                limit_price=f"{order_price:.8f}",
                post_only=self.cfg.post_only,
            )
            return {
                "mode": "limit",
                "response": resp,
                "order_id": client_order_id,
                "submitted_size": norm_size,
                "limit_price": order_price,
            }
        except Exception:
            if not fallback_to_market:
                raise
            resp = self.client.create_order(
                product_id=product,
                side=side,
                size=size_s,
                client_order_id=client_order_id,
                order_type="market",
            )
            return {
                "mode": "market",
                "response": resp,
                "order_id": client_order_id,
                "submitted_size": norm_size,
            }

    def place_maker_first(
        self,
        product: str,
        side: str,
        size: float,
        now: datetime,
        *,
        expected_edge_bps: Optional[float] = None,
    ) -> dict:
        book = self.client.get_best_bid_ask(product)
        bid = float(book.bid)
        ask = float(book.ask)
        mid = ((bid + ask) / 2.0) if bid > 0 and ask > 0 else max(bid, ask)

        norm_size, reason = self.normalize_size(product, side, size, ref_price=mid)
        if norm_size <= 0:
            raise ValueError(f"invalid_order_size: {reason}")

        retries = max(1, int(self.cfg.maker_retries))
        timeout_s = max(1, int(self.cfg.maker_timeout_seconds))

        for attempt in range(retries):
            book = self.client.get_best_bid_ask(product)
            bid = float(book.bid)
            ask = float(book.ask)
            limit_price = self._maker_limit_price(side, bid, ask, product)

            client_order_id = self._make_order_id(product, side, norm_size, now + timedelta(seconds=attempt))
            resp = self.client.create_order(
                product_id=product,
                side=side,
                size=f"{norm_size:.8f}",
                client_order_id=client_order_id,
                order_type="limit",
                limit_price=f"{limit_price:.8f}",
                post_only=True,
            )

            deadline = time.time() + timeout_s
            while time.time() < deadline:
                row = self._poll_order(client_order_id, product=product)
                if row is None:
                    # if order disappears, assume terminal and break to retry or success path
                    break
                status = self._order_status(row)
                if self._is_filled(status):
                    return {
                        "mode": "maker_limit",
                        "order_id": client_order_id,
                        "limit_price": limit_price,
                        "submitted_size": norm_size,
                        "response": row,
                    }
                if self._is_terminal(status):
                    break
                time.sleep(1.0)

            # timeout / not filled
            try:
                self.client.cancel_order(client_order_id)
            except Exception:
                pass

        if self._taker_fallback_allowed(expected_edge_bps=expected_edge_bps):
            client_order_id = self._make_order_id(product, side, norm_size, now)
            resp = self.client.create_order(
                product_id=product,
                side=side,
                size=f"{norm_size:.8f}",
                client_order_id=client_order_id,
                order_type="market",
            )
            return {
                "mode": "taker_market_fallback",
                "order_id": client_order_id,
                "submitted_size": norm_size,
                "response": resp,
            }

        return {
            "mode": "maker_unfilled",
            "order_id": None,
            "submitted_size": norm_size,
            "response": None,
        }

    def target_to_order(
        self,
        product: str,
        current_fraction: float,
        target_fraction: float,
        equity_usd: float,
        price: float,
        latest_bid: float,
        latest_ask: float,
    ) -> List[RoutedOrder]:
        if target_fraction <= 0 and current_fraction <= 0:
            return []
        if abs(target_fraction - current_fraction) < 1e-9:
            return []

        side = "BUY" if target_fraction > current_fraction else "SELL"
        delta = abs(target_fraction - current_fraction)
        if equity_usd <= 0 or price <= 0:
            return []

        usd_notional = delta * equity_usd
        if usd_notional <= 1e-6:
            return []

        px = (latest_ask if side == "BUY" else latest_bid) or price
        raw_size = usd_notional / px
        size, reason = self.normalize_size(product, side, raw_size, ref_price=px)
        if size <= 0:
            return []

        now = datetime.now(tz=timezone.utc)
        order_id = self._make_order_id(product, side, size, now)
        return [
            RoutedOrder(
                client_order_id=order_id,
                side=side,
                product=product,
                size=float(size),
                order_type="limit",
                price=float(price),
                created_at=now,
                status="new",
            )
        ]

    def simulate_fill(self, side: str, fraction_delta: float, price: float, equity_usd: float, maker_fee_rate: float, taker_fee_rate: float) -> SimulatedFill:
        fee_rate = maker_fee_rate if fraction_delta < 0.5 else taker_fee_rate
        notional = abs(fraction_delta) * equity_usd
        fill_price = price
        fee = notional * fee_rate
        return SimulatedFill(
            side=side,
            size=notional / price,
            price=fill_price,
            notional=notional,
            fee_rate=fee_rate,
            is_taker=notional > 0 and fee_rate == taker_fee_rate,
            fee=fee,
            ts=datetime.now(tz=timezone.utc),
        )

    def is_filled_within_timeout(self, order_id: str, timeout_s: int = 60, product: Optional[str] = None) -> bool:
        deadline = time.time() + max(0, timeout_s)
        while time.time() <= deadline:
            row = self._poll_order(order_id, product=product)
            if row is None:
                return False
            status = self._order_status(row)
            if self._is_filled(status):
                return True
            if self._is_terminal(status):
                return False
            time.sleep(1.0)
        return False

    def cancel_order(self, client_order_id: str) -> bool:
        try:
            self.client.cancel_order(client_order_id)
            return True
        except Exception:
            return False
