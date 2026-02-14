from __future__ import annotations

from dataclasses import dataclass
from typing import List
from datetime import datetime, timedelta, timezone
import sqlite3
import logging

import pandas as pd

from ..config import DataConfig
from ..coinbase_client import RESTClientWrapper

logger = logging.getLogger("trading_bot.data.candles")


@dataclass
class CandleQuery:
    product: str
    timeframe: str
    start: datetime
    end: datetime
    force_refresh: bool = False


class CandleStore:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self.path = cfg.cache_dir / "candles.sqlite"
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                product TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ts INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY(product, timeframe, ts)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_candles_lookup ON candles(product,timeframe,ts)")
        self.conn.commit()

    def _fetch_cached(self, product: str, timeframe: str, start_ts: int, end_ts: int) -> pd.DataFrame:
        query = """
            SELECT ts, open, high, low, close, volume
            FROM candles
            WHERE product=? AND timeframe=? AND ts>=? AND ts<=?
            ORDER BY ts ASC
        """
        rows = self.conn.execute(query, (product, timeframe, start_ts, end_ts)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def _upsert(self, product: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        rows = []
        for _, row in df.iterrows():
            ts_obj = pd.Timestamp(row["timestamp"])
            if ts_obj.tzinfo is None:
                ts_obj = ts_obj.tz_localize(timezone.utc)
            else:
                ts_obj = ts_obj.tz_convert(timezone.utc)
            ts = int(ts_obj.timestamp())
            rows.append((product, timeframe, ts, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"])))
        self.conn.executemany(
            """
            INSERT INTO candles(product, timeframe, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(product,timeframe,ts) DO UPDATE SET
              open=excluded.open, high=excluded.high, low=excluded.low, close=excluded.close, volume=excluded.volume
            """,
            rows,
        )
        self.conn.commit()

    def get_candles(self, client: RESTClientWrapper, query: CandleQuery) -> pd.DataFrame:
        # normalize to UTC
        start_ts = pd.Timestamp(query.start)
        end_ts = pd.Timestamp(query.end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        start = start_ts.to_pydatetime()
        end = end_ts.to_pydatetime()

        if start > end:
            raise ValueError("start must be <= end")

        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        # first try cache
        cached = self._fetch_cached(query.product, query.timeframe, start_ts, end_ts)
        if not cached.empty and not query.force_refresh:
            # if cache has everything needed (contiguous enough) return
            if cached["timestamp"].min() <= pd.Timestamp(start).tz_convert("UTC") and cached["timestamp"].max() >= pd.Timestamp(end).tz_convert("UTC"):
                logger.info("Using cached %s candles for %s", query.timeframe, query.product)
                return cached

        # fetch from API with pagination
        all_frames: List[pd.DataFrame] = []
        cursor = start
        limit = min(query.timeframe == "1h" and self.cfg.hourly_limit or self.cfg.daily_limit, 350)
        batch_seconds = _tf_to_seconds(query.timeframe)
        while cursor < end:
            chunk_end = min(cursor + timedelta(seconds=limit * batch_seconds), end)
            logger.info("Fetching candles %s-%s", query.product, query.timeframe)
            raw = client.get_product_candles(query.product, cursor, chunk_end, timeframe=query.timeframe, limit=limit)
            if not raw:
                break

            rows = []
            for item in raw:
                if isinstance(item, (list, tuple)):
                    if len(item) >= 6:
                        ts, low, high, open_, close, volume = item[:6]
                        rows.append(
                            {
                                "timestamp": pd.to_datetime(int(ts), unit="s", utc=True),
                                "open": float(open_),
                                "high": float(high),
                                "low": float(low),
                                "close": float(close),
                                "volume": float(volume),
                            }
                        )
                    continue
                # Coinbase candle dict fields vary: start/time/open/high/low/close/volume
                ts = item.get("start") or item.get("time") or item.get("timestamp")
                if ts is None:
                    continue
                ts_int = int(ts)
                rows.append(
                    {
                        "timestamp": pd.to_datetime(ts_int, unit="s", utc=True),
                        "open": float(item.get("open")),
                        "high": float(item.get("high")),
                        "low": float(item.get("low")),
                        "close": float(item.get("close")),
                        "volume": float(item.get("volume", 0.0)),
                    }
                )
            if rows:
                df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                all_frames.append(df)
                last_ts = int(rows[-1]["timestamp"].timestamp())
            else:
                last_ts = int(cursor.timestamp())

            if len(rows) < limit:
                break
            cursor = datetime.fromtimestamp(last_ts, tz=timezone.utc) + timedelta(seconds=batch_seconds)

        remote = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if not remote.empty:
            self._upsert(query.product, query.timeframe, remote)

        out = remote.sort_values("timestamp")
        if cached.empty:
            return out
        merged = pd.concat([cached, out], ignore_index=True).drop_duplicates(subset=["timestamp"])
        return merged.sort_values("timestamp").reset_index(drop=True)


def _tf_to_seconds(tf: str) -> int:
    if tf == "1h":
        return 3600
    if tf == "1d":
        return 86400
    raise ValueError(f"unsupported timeframe: {tf}")


def align_closed_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # For safety drop any partial/current candle by requiring timestamp <= now-floor(tf)
    now = pd.Timestamp.now(tz="UTC").floor("1h" if timeframe == "1h" else "1d")
    return df[df["timestamp"] < now].copy()


def to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out.set_index("timestamp", drop=True).sort_index()
