from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import json
import logging
import time

import httpx
import pandas as pd

logger = logging.getLogger("trading_bot.data.fred")


@dataclass
class FredCacheStats:
    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    series_errors: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total <= 0:
            return 0.0
        return float(self.cache_hits) / float(total)


class FredClient:
    """Thin FRED v2 client with local file cache + retry/backoff."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.stlouisfed.org/fred",
        cache_dir: str | Path = ".trading_bot_cache/fred",
        timeout_seconds: float = 15.0,
        max_retries: int = 5,
        backoff_seconds: float = 0.5,
        cache_ttl_hours: float = 24.0,
        use_stale_cache_for_backtest: bool = True,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url).rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.backoff_seconds = max(0.1, float(backoff_seconds))
        self.cache_ttl_hours = max(0.0, float(cache_ttl_hours))
        self.use_stale_cache_for_backtest = bool(use_stale_cache_for_backtest)
        self.stats = FredCacheStats()

        self._client = httpx.Client(timeout=self.timeout_seconds)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(params: dict[str, Any]) -> str:
        # Stable deterministic hash over sorted JSON payload.
        payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _cache_paths(self, series_id: str, cache_key: str) -> tuple[Path, Path]:
        safe_series = "".join(c for c in str(series_id) if c.isalnum() or c in {"_", "-"})
        csv_path = self.cache_dir / f"{safe_series}_{cache_key}.csv"
        meta_path = self.cache_dir / f"{safe_series}_{cache_key}.meta.json"
        return csv_path, meta_path

    def _cache_is_fresh(self, meta: dict[str, Any]) -> bool:
        if self.use_stale_cache_for_backtest:
            return True
        fetched_at = meta.get("fetched_at")
        if not fetched_at:
            return False
        try:
            ts = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
        except Exception:
            return False
        age_hours = (datetime.now(tz=timezone.utc) - ts).total_seconds() / 3600.0
        return age_hours <= self.cache_ttl_hours

    def _load_cache(self, series_id: str, params: dict[str, Any]) -> Optional[pd.DataFrame]:
        key = self._cache_key(params)
        csv_path, meta_path = self._cache_paths(series_id, key)
        if not csv_path.exists() or not meta_path.exists():
            self.stats.cache_misses += 1
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            self.stats.cache_misses += 1
            return None

        if not self._cache_is_fresh(meta):
            self.stats.cache_misses += 1
            return None

        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True, errors="coerce")
                if "realtime_start" in df.columns:
                    df["realtime_start"] = pd.to_datetime(df["realtime_start"], utc=True, errors="coerce")
                if "realtime_end" in df.columns:
                    df["realtime_end"] = pd.to_datetime(df["realtime_end"], utc=True, errors="coerce")
                df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
                df = df.dropna(subset=["observation_date", "value"]).sort_values("observation_date")
            self.stats.cache_hits += 1
            return df
        except Exception:
            self.stats.cache_misses += 1
            return None

    def _save_cache(self, series_id: str, params: dict[str, Any], df: pd.DataFrame) -> None:
        key = self._cache_key(params)
        csv_path, meta_path = self._cache_paths(series_id, key)
        payload = {
            "series_id": series_id,
            "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
            "params": params,
            "rows": int(len(df)),
        }
        out = df.copy()
        if not out.empty:
            out["observation_date"] = pd.to_datetime(out["observation_date"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            if "realtime_start" in out.columns:
                out["realtime_start"] = pd.to_datetime(out["realtime_start"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            if "realtime_end" in out.columns:
                out["realtime_end"] = pd.to_datetime(out["realtime_end"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _fetch_series_observations_http(self, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/series/observations"
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._client.get(url, params=params)
                self.stats.requests += 1
                resp.raise_for_status()
                payload = resp.json() if resp.text else {}
                if isinstance(payload, dict) and payload.get("error_code"):
                    raise RuntimeError(f"FRED API error {payload.get('error_code')}: {payload.get('error_message')}")
                return payload if isinstance(payload, dict) else {}
            except Exception as exc:
                if attempt >= self.max_retries:
                    raise
                sleep_s = self.backoff_seconds * (2 ** (attempt - 1))
                sleep_s = min(sleep_s, 30.0)
                logger.warning("FRED request failed (%s), retrying in %.2fs [attempt %d/%d]", exc.__class__.__name__, sleep_s, attempt, self.max_retries)
                time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_series_observations(
        self,
        series_id: str,
        observation_start: str,
        observation_end: str,
        *,
        file_type: str = "json",
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        output_type: int = 1,
        vintage_dates: str | None = None,
        frequency: str | None = None,
        units: str = "lin",
        aggregation_method: str = "avg",
    ) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY is required when fred.enabled=true")

        params: dict[str, Any] = {
            "api_key": self.api_key,
            "series_id": str(series_id),
            "observation_start": str(observation_start),
            "observation_end": str(observation_end),
            "file_type": str(file_type),
            "output_type": int(output_type),
            "units": str(units),
            "aggregation_method": str(aggregation_method),
        }
        if realtime_start:
            params["realtime_start"] = str(realtime_start)
        if realtime_end:
            params["realtime_end"] = str(realtime_end)
        if vintage_dates:
            params["vintage_dates"] = str(vintage_dates)
        if frequency:
            params["frequency"] = str(frequency)

        cached = self._load_cache(series_id, params)
        if cached is not None:
            return cached

        payload = self._fetch_series_observations_http(params)
        rows = payload.get("observations") if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            rows = []

        out_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            date_raw = row.get("date")
            value_raw = row.get("value")
            if value_raw in {None, "", "."}:
                continue
            try:
                value = float(value_raw)
            except Exception:
                continue
            ts = pd.to_datetime(str(date_raw), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            out_rows.append(
                {
                    "observation_date": ts,
                    "value": value,
                    "realtime_start": pd.to_datetime(row.get("realtime_start"), utc=True, errors="coerce") if row.get("realtime_start") else pd.NaT,
                    "realtime_end": pd.to_datetime(row.get("realtime_end"), utc=True, errors="coerce") if row.get("realtime_end") else pd.NaT,
                }
            )

        out = pd.DataFrame(out_rows)
        if out.empty:
            out = pd.DataFrame(columns=["observation_date", "value", "realtime_start", "realtime_end"])
        else:
            out = out.sort_values("observation_date").drop_duplicates(subset=["observation_date"], keep="last")

        self._save_cache(series_id, params, out)
        return out

    def close(self) -> None:
        self._client.close()
