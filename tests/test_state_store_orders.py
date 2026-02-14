from __future__ import annotations

from bot.execution.state_store import BotStateStore


def test_open_order_lifecycle_fields(tmp_path):
    store = BotStateStore(tmp_path / "state.sqlite")
    store.put_open_order(
        order_id="abc123",
        product="BTC-USD",
        side="BUY",
        size=0.05,
        order_type="limit",
        price=50000.0,
        created_at_ts=123456,
        status="submitted",
        filled_size=0.0,
        replace_count=0,
        metadata={"test": True},
    )

    row = store.get_open_order("abc123")
    assert row is not None
    assert row["status"] == "submitted"

    store.update_open_order("abc123", status="PARTIALLY_FILLED", filled_size=0.01, replace_count=1)
    row2 = store.get_open_order("abc123")
    assert row2 is not None
    assert row2["status"] == "PARTIALLY_FILLED"
    assert float(row2["filled_size"]) == 0.01
    assert int(row2["replace_count"]) == 1
