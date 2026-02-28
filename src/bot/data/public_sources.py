from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import sqlite3
import json
import logging
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger("trading_bot.data.public")


@dataclass
class PublicDataCache:
    db_path: Path

    def __post_init__(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public_cache (
                source TEXT NOT NULL,
                key TEXT NOT NULL,
                ts INTEGER NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(source, key)
            )
            """
        )
        self.conn.commit()

    def load(self, source: str, key: str, ttl_minutes: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT ts, payload FROM public_cache WHERE source=? AND key=?",
            (source, key),
        ).fetchone()
        if not row:
            return None
        ts, payload = row
        age = datetime.now(tz=timezone.utc).timestamp() - float(ts)
        if age > ttl_minutes * 60:
            return None
        try:
            return json.loads(payload)
        except Exception:
            return None

    def save(self, source: str, key: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT INTO public_cache(source,key,ts,payload) VALUES(?,?,?,?) ON CONFLICT(source,key) DO UPDATE SET ts=excluded.ts,payload=excluded.payload",
            (source, key, int(datetime.now(tz=timezone.utc).timestamp()), json.dumps(payload)),
        )
        self.conn.commit()


class PublicDataFetcher:
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    BLOCKCHAIN_CHART_URL = "https://api.blockchain.info/charts/{chart}"
    BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"

    def __init__(self, cache_ttl_minutes: int = 60, cache_dir: Optional[Path] = None) -> None:
        self.cache_ttl_minutes = cache_ttl_minutes
        self.client = httpx.Client(timeout=15.0)
        cache_path = (cache_dir or Path(".trading_bot_cache")) / "public_sources.sqlite"
        self.cache = PublicDataCache(cache_path)

    def _get_json(self, source: str, key: str, url: str, params: Optional[dict] = None) -> dict:
        cached = self.cache.load(source, key, self.cache_ttl_minutes)
        if cached is not None:
            return cached
        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        self.cache.save(source, key, data)
        return data

    def fetch_fear_and_greed(self) -> pd.Series:
        data = self._get_json("fear_greed", "latest", self.FEAR_GREED_URL)
        values = data.get("data") or []
        if not values:
            return pd.Series(dtype=float)
        latest = values[0]
        return pd.Series(
            {
                "value": float(latest.get("value", 0.0)),
                "timestamp": pd.to_datetime(int(latest.get("timestamp", 0)), unit="s", utc=True),
                "classification": latest.get("value_classification", ""),
            }
        )

    def fetch_blockchain_chart(self, chart_name: str, timespan: str = "30days") -> pd.Series:
        params = {"timespan": timespan, "format": "json", "cors": True}
        raw = self._get_json("blockchain_chart", chart_name, self.BLOCKCHAIN_CHART_URL.format(chart=chart_name), params=params)
        values = raw.get("values", [])
        out = []
        for v in values:
            ts = pd.to_datetime(int(v["x"]), unit="s", utc=True)
            val = float(v["y"])
            out.append((ts, val))
        if not out:
            return pd.Series(dtype=float)
        idx, vals = zip(*out)
        return pd.Series(list(vals), index=pd.to_datetime(idx))

    def fetch_blockchain_stats(self) -> pd.Series:
        raw = self._get_json("blockchain_stats", "latest", self.BLOCKCHAIN_STATS_URL)
        return pd.Series(raw)

    def fetch_macro_context(self, timespan: str = "30days") -> Dict[str, pd.Series]:
        fg = self.fetch_fear_and_greed()
        stats = self.fetch_blockchain_stats()
        n_users = self.fetch_blockchain_chart("n-users", timespan=timespan)
        mkt_price = self.fetch_blockchain_chart("market-price", timespan=timespan)
        return {
            "fear_greed": fg,
            "blockchain_stats": stats,
            "blockchain_n_users": n_users,
            "blockchain_market_price": mkt_price,
        }

    def close(self) -> None:
        self.client.close()
