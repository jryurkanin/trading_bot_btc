from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import json
import sqlite3


@dataclass
class TradeState:
    last_signal_ts: int | None = None
    current_exposure: float = 0.0
    equity_peak: float = 1.0
    last_regime: str | None = None


class BotStateStore:
    """SQLite persistence for run state.

    This is deliberately conservative and intentionally schema-stable for later upgrades.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._init()

    def _init(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS open_orders (
                client_order_id TEXT PRIMARY KEY,
                product TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                order_type TEXT NOT NULL,
                price REAL,
                created_at INTEGER NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                ts INTEGER PRIMARY KEY,
                product TEXT NOT NULL,
                btc FLOAT NOT NULL,
                usd FLOAT NOT NULL,
                exposure FLOAT NOT NULL,
                comment TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                ts INTEGER PRIMARY KEY,
                product TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def set_kv(self, key: str, value: object) -> None:
        self.conn.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, json.dumps(value)))
        self.conn.commit()

    def get_kv(self, key: str, default=None):
        row = self.conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception:
            return row[0]

    def put_open_order(self, order_id: str, product: str, side: str, size: float, order_type: str, price: Optional[float], created_at_ts: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO open_orders(client_order_id,product,side,size,order_type,price,created_at) VALUES(?,?,?,?,?,?,?)",
            (order_id, product, side, size, order_type, price, created_at_ts),
        )
        self.conn.commit()

    def drop_open_order(self, order_id: str) -> None:
        self.conn.execute("DELETE FROM open_orders WHERE client_order_id=?", (order_id,))
        self.conn.commit()

    def list_open_orders(self):
        rows = self.conn.execute("SELECT client_order_id, product, side, size, order_type, price, created_at FROM open_orders").fetchall()
        return rows

    def store_position(self, ts: int, product: str, btc: float, usd: float, exposure: float, comment: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO positions(ts, product, btc, usd, exposure, comment) VALUES(?,?,?,?,?,?)",
            (ts, product, btc, usd, exposure, comment),
        )
        self.conn.commit()

    def latest_position(self, product: str):
        row = self.conn.execute(
            "SELECT ts, btc, usd, exposure, comment FROM positions WHERE product=? ORDER BY ts DESC LIMIT 1",
            (product,),
        ).fetchone()
        return row

    def log_decision(self, ts: int, product: str, payload: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO decisions(ts, product, payload) VALUES(?,?,?)",
            (ts, product, json.dumps(payload)),
        )
        self.conn.commit()

    def get_last_signal_ts(self) -> int | None:
        v = self.get_kv("last_signal_ts")
        return int(v) if isinstance(v, int) else None

    def set_last_signal_ts(self, ts: int) -> None:
        self.set_kv("last_signal_ts", ts)

    def get_last_regime(self) -> str | None:
        v = self.get_kv("last_regime")
        return str(v) if isinstance(v, str) else None

    def set_last_regime(self, regime: str) -> None:
        self.set_kv("last_regime", regime)

    def get_equity_peak(self) -> float:
        v = self.get_kv("equity_peak")
        return float(v) if isinstance(v, (int, float)) else 1.0

    def set_equity_peak(self, val: float) -> None:
        self.set_kv("equity_peak", float(val))

    def close(self):
        self.conn.close()
