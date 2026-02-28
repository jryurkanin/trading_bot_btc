from __future__ import annotations

from datetime import datetime, timezone

from bot.coinbase_client import ProductConstraints
from bot.config import ExecutionConfig
from bot.execution.order_router import OrderRouter


class DummyClient:
    def __init__(self):
        self.orders = []

    def get_product_constraints(self, product_id: str) -> ProductConstraints:
        return ProductConstraints(
            product_id=product_id,
            base_increment=0.001,
            quote_increment=0.01,
            base_min_size=0.01,
            quote_min_size=1.0,
            price_increment=0.01,
            base_max_size=5.0,
            min_notional=10.0,
        )

    def create_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"ok": True, "order": kwargs}

    def cancel_order(self, client_order_id: str):
        return {"ok": True}


def test_normalize_size_enforces_min_size_and_notional():
    router = OrderRouter(DummyClient(), ExecutionConfig(enforce_product_constraints=True))

    # Too small raw size should be raised to meet min size and min notional.
    size, reason = router.normalize_size("BTC-USD", "BUY", size=0.005, ref_price=1000.0)
    assert reason == "ok"
    assert size >= 0.01
    assert size * 1000.0 >= 10.0


def test_target_to_order_uses_normalized_size():
    router = OrderRouter(DummyClient(), ExecutionConfig(enforce_product_constraints=True))
    orders = router.target_to_order(
        product="BTC-USD",
        current_fraction=0.0,
        target_fraction=0.001,
        equity_usd=10_000.0,
        price=1000.0,
        latest_bid=1000.0,
        latest_ask=1000.0,
    )
    assert len(orders) == 1
    assert orders[0].size >= 0.01


def test_place_limit_with_fallback_submits_adjusted_size():
    client = DummyClient()
    router = OrderRouter(client, ExecutionConfig(enforce_product_constraints=True, fallback_to_market=True))
    out = router.place_limit_with_fallback(
        product="BTC-USD",
        side="BUY",
        size=0.002,
        bid=999.0,
        ask=1001.0,
        now=datetime.now(tz=timezone.utc),
        fallback_to_market=True,
    )
    assert out["submitted_size"] >= 0.01
    assert len(client.orders) >= 1
