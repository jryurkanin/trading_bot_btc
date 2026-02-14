from __future__ import annotations

from datetime import datetime, timezone

from bot.execution.state_store import BotStateStore
from bot.live.paper import PaperTrader


def test_paper_portfolio_equity_and_bounds(tmp_path):
    store = BotStateStore(tmp_path / "state.sqlite")
    trader = PaperTrader(store)

    trader.set_portfolio(usd=1_000.0, btc=0.1)
    eq = trader.get_portfolio().equity(10_000.0)
    assert eq == 2_000.0

    # Try to buy to full allocation; should not spend beyond available cash.
    fills = trader.execute_fraction(
        target_fraction=1.0,
        now=datetime.now(tz=timezone.utc),
        latest_close=10_000.0,
        latest_high=10_100.0,
        latest_low=9_900.0,
    )
    p = trader.get_portfolio()
    assert p.usd >= -1e-9
    assert p.btc >= 0.0

    # Try to sell to flat; should never create negative BTC.
    fills2 = trader.execute_fraction(
        target_fraction=0.0,
        now=datetime.now(tz=timezone.utc),
        latest_close=10_000.0,
        latest_high=10_100.0,
        latest_low=9_900.0,
    )
    p2 = trader.get_portfolio()
    assert p2.btc >= -1e-9
