from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib
import time
import uuid

from ..config import ExecutionConfig
from ..coinbase_client import RESTClientWrapper


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

    @staticmethod
    def _make_order_id(product: str, side: str, size: float, now: datetime) -> str:
        raw = f"{product}:{side}:{size:.8f}:{int(now.timestamp())}"
        return hashlib.md5(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _signed_size(size: float) -> float:
        return max(0.0, float(size))

    def place_limit_with_fallback(
        self,
        product: str,
        side: str,
        size: float,
        bid: float,
        ask: float,
        now: datetime,
        fallback_to_market: bool = True,
    ) -> dict:
        client_order_id = self._make_order_id(product, side, size, now)
        size_s = f"{size:.8f}"
        # limit just inside spread to reduce taker risk
        if side == "BUY":
            limit = max(0.00000001, ask * (1 - self.cfg.limit_price_offset_bps / 10000.0))
            order_price = max(limit, 0.0)
        else:
            order_price = bid * (1 + self.cfg.limit_price_offset_bps / 10000.0)

        try:
            resp = self.client.create_order(
                product_id=product,
                side=side,
                size=size_s,
                client_order_id=client_order_id,
                order_type="limit",
                limit_price=f"{order_price:.2f}",
                post_only=self.cfg.post_only,
            )
            return {"mode": "limit", "response": resp, "order_id": client_order_id}
        except Exception:
            if not fallback_to_market:
                raise
            # fallback to market
            resp = self.client.create_order(
                product_id=product,
                side=side,
                size=size_s,
                client_order_id=client_order_id,
                order_type="market",
            )
            return {"mode": "market", "response": resp, "order_id": client_order_id}

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
        size = usd_notional / ((latest_ask if side == "BUY" else latest_bid) or price)
        order_id = self._make_order_id(product, side, size, datetime.utcnow())
        return [
            RoutedOrder(
                client_order_id=order_id,
                side=side,
                product=product,
                size=float(size),
                order_type="limit",
                price=float(price),
                created_at=datetime.utcnow(),
            )
        ]

    def simulate_fill(self, side: str, fraction_delta: float, price: float, equity_usd: float, maker_fee_rate: float, taker_fee_rate: float) -> SimulatedFill:
        fee_rate = maker_fee_rate if fraction_delta < 0.5 else taker_fee_rate
        notional = abs(fraction_delta) * equity_usd
        fill_price = price
        fee = notional * fee_rate
        return SimulatedFill(side=side, size=notional / price, price=fill_price, notional=notional, fee_rate=fee_rate, is_taker=notional > 0 and fee_rate == taker_fee_rate, fee=fee, ts=datetime.utcnow())

    def is_filled_within_timeout(self, order_id: str, timeout_s: int = 60) -> bool:
        # In this scaffold, we emulate immediate fill behavior for market style and assume
        # limit orders are often still open in live. In production, poll REST order status.
        # Here keep deterministic: return True only when timeout<=0
        return timeout_s <= 0

    def cancel_order(self, client_order_id: str) -> None:
        try:
            self.client.cancel_order(client_order_id)
        except Exception:
            pass
