from __future__ import annotations

from datetime import timedelta

import pandas as pd

from bot.config import DataConfig
from bot.data.candles import CandleQuery, CandleStore


class CountingClient:
    def __init__(self):
        self.calls = 0

    def get_product_candles(self, *args, **kwargs):
        self.calls += 1
        return []


def test_cache_coverage_uses_latest_closed_bar_without_refetch(tmp_path):
    cfg = DataConfig(cache_dir=tmp_path)
    store = CandleStore(cfg)

    ts = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="h", tz="UTC")
    base = 100.0
    cached = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [base + i for i in range(4)],
            "high": [base + i + 1 for i in range(4)],
            "low": [base + i - 1 for i in range(4)],
            "close": [base + i for i in range(4)],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    store._upsert("BTC-USD", "1h", cached)

    # End at 04:30 means latest closed bar is 03:00. Cache already has it.
    q = CandleQuery(
        product="BTC-USD",
        timeframe="1h",
        start=ts[0].to_pydatetime(),
        end=(ts[-1] + timedelta(hours=1, minutes=30)).to_pydatetime(),
        force_refresh=False,
    )

    client = CountingClient()
    out = store.get_candles(client, q)

    assert client.calls == 0
    assert len(out) == 4
    assert pd.to_datetime(out["timestamp"], utc=True).max() == ts[-1]
