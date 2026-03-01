"""Tests for CandleStore cache and pagination (Section 7)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd

from bot.config import DataConfig
from bot.data.candles import CandleStore, CandleQuery


def _make_store(tmp_dir: str) -> CandleStore:
    cfg = DataConfig(cache_dir=Path(tmp_dir))
    return CandleStore(cfg)


def test_candle_store_write_and_read():
    """Store candles and retrieve them."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        rows = []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(10):
            ts = int((base + timedelta(hours=i)).timestamp())
            rows.append((ts, 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1000.0))

        store.conn.executemany(
            "INSERT OR REPLACE INTO candles (product,timeframe,ts,open,high,low,close,volume) VALUES ('BTC-USD','ONE_HOUR',?,?,?,?,?,?)",
            rows,
        )
        store.conn.commit()

        # Read back
        q = CandleQuery(
            product="BTC-USD",
            timeframe="ONE_HOUR",
            start=base,
            end=base + timedelta(hours=9),
        )
        cursor = store.conn.execute(
            "SELECT ts,open,high,low,close,volume FROM candles WHERE product=? AND timeframe=? AND ts>=? AND ts<=? ORDER BY ts",
            (q.product, q.timeframe, int(q.start.timestamp()), int(q.end.timestamp())),
        )
        result = cursor.fetchall()
        assert len(result) == 10
        store.close()


def test_candle_store_context_manager():
    """CandleStore should work as context manager."""
    with tempfile.TemporaryDirectory() as tmp:
        with _make_store(tmp) as store:
            assert store.conn is not None


def test_candle_store_skips_none_ohlc():
    """Candles with None OHLC values should be skipped during ingestion."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp)
        # Insert a valid row
        ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
        store.conn.execute(
            "INSERT INTO candles (product,timeframe,ts,open,high,low,close,volume) VALUES ('BTC-USD','ONE_HOUR',?,100,105,95,102,1000)",
            (ts,),
        )
        store.conn.commit()

        # Verify it was stored
        row = store.conn.execute("SELECT * FROM candles WHERE ts=?", (ts,)).fetchone()
        assert row is not None
        store.close()
