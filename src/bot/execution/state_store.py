from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import json
import sqlite3
import time

from ..system_log import get_system_logger

logger = get_system_logger("execution.state_store")


@dataclass
class TradeState:
    last_signal_ts: int | None = None
    current_exposure: float = 0.0
    equity_peak: float = 1.0
    last_regime: str | None = None


class BotStateStore:
    """SQLite persistence for run state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        logger.info("state_store_open path=%s", self.path)
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
                created_at INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'submitted',
                filled_size REAL NOT NULL DEFAULT 0.0,
                last_update_ts INTEGER NOT NULL DEFAULT 0,
                replace_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS order_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                client_order_id TEXT NOT NULL,
                event TEXT NOT NULL,
                payload TEXT
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

        # Best-effort migration for old DBs missing newer columns.
        existing_cols = {r[1] for r in self.conn.execute("PRAGMA table_info(open_orders)").fetchall()}
        alter_statements = []
        if "status" not in existing_cols:
            alter_statements.append("ALTER TABLE open_orders ADD COLUMN status TEXT NOT NULL DEFAULT 'submitted'")
        if "filled_size" not in existing_cols:
            alter_statements.append("ALTER TABLE open_orders ADD COLUMN filled_size REAL NOT NULL DEFAULT 0.0")
        if "last_update_ts" not in existing_cols:
            alter_statements.append("ALTER TABLE open_orders ADD COLUMN last_update_ts INTEGER NOT NULL DEFAULT 0")
        if "replace_count" not in existing_cols:
            alter_statements.append("ALTER TABLE open_orders ADD COLUMN replace_count INTEGER NOT NULL DEFAULT 0")
        if "metadata" not in existing_cols:
            alter_statements.append("ALTER TABLE open_orders ADD COLUMN metadata TEXT")

        for stmt in alter_statements:
            self.conn.execute(stmt)
        if alter_statements:
            self.conn.commit()
            logger.info("state_store_migrations_applied path=%s count=%d statements=%s", self.path, len(alter_statements), alter_statements)
        else:
            logger.debug("state_store_schema_ok path=%s", self.path)

    def set_kv(self, key: str, value: object) -> None:
        self.conn.execute("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, json.dumps(value)))
        self.conn.commit()

    def get_kv(self, key: str, default=None):
        row = self.conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row[0])
        except Exception as exc:
            logger.warning("state_store_kv_json_decode_failed key=%s error=%s", key, exc)
            return row[0]

    def put_open_order(
        self,
        order_id: str,
        product: str,
        side: str,
        size: float,
        order_type: str,
        price: Optional[float],
        created_at_ts: int,
        status: str = "submitted",
        filled_size: float = 0.0,
        replace_count: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO open_orders(
                client_order_id,product,side,size,order_type,price,created_at,status,filled_size,last_update_ts,replace_count,metadata
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                order_id,
                product,
                side,
                size,
                order_type,
                price,
                created_at_ts,
                status,
                float(filled_size),
                int(created_at_ts),
                int(replace_count),
                json.dumps(metadata or {}),
            ),
        )
        self.conn.commit()

    def update_open_order(
        self,
        order_id: str,
        *,
        status: Optional[str] = None,
        filled_size: Optional[float] = None,
        replace_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
        ts: Optional[int] = None,
    ) -> None:
        updates: list[str] = []
        values: list[Any] = []

        if status is not None:
            updates.append("status=?")
            values.append(status)
        if filled_size is not None:
            updates.append("filled_size=?")
            values.append(float(filled_size))
        if replace_count is not None:
            updates.append("replace_count=?")
            values.append(int(replace_count))
        if metadata is not None:
            updates.append("metadata=?")
            values.append(json.dumps(metadata))

        updates.append("last_update_ts=?")
        values.append(int(ts if ts is not None else time.time()))

        values.append(order_id)
        self.conn.execute(f"UPDATE open_orders SET {', '.join(updates)} WHERE client_order_id=?", values)
        self.conn.commit()

    def log_order_event(self, client_order_id: str, event: str, payload: Optional[dict[str, Any]] = None, ts: Optional[int] = None) -> None:
        self.conn.execute(
            "INSERT INTO order_events(ts, client_order_id, event, payload) VALUES(?,?,?,?)",
            (int(ts if ts is not None else time.time()), client_order_id, event, json.dumps(payload or {})),
        )
        self.conn.commit()

    def drop_open_order(self, order_id: str) -> None:
        self.conn.execute("DELETE FROM open_orders WHERE client_order_id=?", (order_id,))
        self.conn.commit()

    def list_open_orders(self):
        rows = self.conn.execute(
            "SELECT client_order_id, product, side, size, order_type, price, created_at, status, filled_size, last_update_ts, replace_count, metadata FROM open_orders"
        ).fetchall()
        return [tuple(r) for r in rows]

    def list_open_orders_dict(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT client_order_id, product, side, size, order_type, price, created_at, status, filled_size, last_update_ts, replace_count, metadata FROM open_orders"
        ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except Exception as exc:
                logger.warning(
                    "state_store_open_orders_metadata_decode_failed order_id=%s error=%s",
                    item.get("client_order_id"),
                    exc,
                )
                item["metadata"] = {}
            out.append(item)
        return out

    def get_open_order(self, order_id: str) -> Optional[dict[str, Any]]:
        row = self.conn.execute(
            "SELECT client_order_id, product, side, size, order_type, price, created_at, status, filled_size, last_update_ts, replace_count, metadata FROM open_orders WHERE client_order_id=?",
            (order_id,),
        ).fetchone()
        if not row:
            return None
        out = dict(row)
        try:
            out["metadata"] = json.loads(out.get("metadata") or "{}")
        except Exception as exc:
            logger.warning("state_store_open_order_metadata_decode_failed order_id=%s error=%s", order_id, exc)
            out["metadata"] = {}
        return out

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
        logger.info("state_store_closed path=%s", self.path)
