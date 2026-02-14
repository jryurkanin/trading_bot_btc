from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import time
from collections import deque
import logging
import json
import hmac
import hashlib
import base64
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .config import DataConfig, CoinbaseConfig

logger = logging.getLogger("trading_bot.coinbase")


class CoinbaseError(Exception):
    pass


class RateLimitError(CoinbaseError):
    pass


class TransientError(CoinbaseError):
    pass


@dataclass(frozen=True)
class BestBidAsk:
    bid: float
    ask: float
    time: datetime


@dataclass(frozen=True)
class FeeSummary:
    maker_fee_rate: float
    taker_fee_rate: float


@dataclass(frozen=True)
class Account:
    uuid: str
    currency: str
    balance: float
    available_balance: Optional[float] = None


class RESTClientWrapper:
    """Coinbase brokerage REST wrapper.

    Tries to use `coinbase-advanced-py` when available but keeps a robust
    httpx fallback so unit tests can run without a live SDK.
    """

    BASE_URL = "https://api.coinbase.com/api/v3/brokerage"
    SANDBOX_URL = "https://api-sandbox.coinbase.com/api/v3/brokerage"

    def __init__(self, config: CoinbaseConfig, data_cfg: Optional[DataConfig] = None) -> None:
        self.config = config
        self.data_cfg = data_cfg or DataConfig()
        self.base_url = self.SANDBOX_URL if config.use_sandbox else self.BASE_URL
        self._request_times: deque[float] = deque(maxlen=max(config.requests_per_minute, 1) * 2)
        self._sdk_client = self._init_sdk_client()
        self.http = httpx.AsyncClient if False else None
        self.sync_http = httpx.Client(
            timeout=config.request_timeout_s,
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "trading-bot-btc/1.0",
            },
        )
        if config.use_sandbox and config.sandbox_error_variant:
            self.sync_http.headers.update({"X-Sandbox": config.sandbox_error_variant})

    def _init_sdk_client(self):
        try:
            from coinbase.rest import RESTClient
            if self.config.api_key and self.config.api_secret:
                return RESTClient(api_key=self.config.api_key, api_secret=self.config.api_secret)
            return RESTClient()
        except Exception:
            return None

    @staticmethod
    def _sign_message(timestamp: str, method: str, path: str, body: str, secret: str) -> str:
        msg = f"{timestamp}{method}{path}{body}".encode()
        # Some Coinbase keys are base64 secrets; others are plain. Support both.
        try:
            key = base64.b64decode(secret)
            if not key:
                key = secret.encode()
        except Exception:
            key = secret.encode()
        return base64.b64encode(hmac.new(key, msg, hashlib.sha256).digest()).decode()

    def _request_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.config.use_sandbox:
            # Sandbox endpoint is static and can accept unauthenticated calls.
            return headers
        if not self.config.api_key or not self.config.api_secret:
            return headers
        ts = str(int(time.time()))
        signature = self._sign_message(ts, method, path, body, self.config.api_secret)
        headers.update(
            {
                "CB-ACCESS-KEY": self.config.api_key,
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": ts,
            }
        )
        if self.config.api_passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = self.config.api_passphrase
        return headers

    def _throttle(self):
        now = time.monotonic()
        # purge old entries
        while self._request_times and now - self._request_times[0] > 60:
            self._request_times.popleft()
        # simple budget rule
        if len(self._request_times) >= self.config.requests_per_minute:
            sleep_for = 60 - (now - self._request_times[0])
            if sleep_for > 0:
                time.sleep(max(0.1, sleep_for))
            now = time.monotonic()
            while self._request_times and now - self._request_times[0] > 60:
                self._request_times.popleft()
        self._request_times.append(time.monotonic())

    def _is_retryable_status(self, status: int) -> bool:
        return status == 429 or 500 <= status < 600

    @retry(
        retry=retry_if_exception_type((RateLimitError, TransientError, httpx.HTTPError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=0.5, max=12),
        reraise=True,
    )
    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, json_payload: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Any:
        self._throttle()
        params = params or {}
        url = path
        body = json.dumps(json_payload or {})
        headers = self._request_headers(method, path, body if method in {"POST", "PUT"} else "")

        # Use SDK path when possible for the common endpoints.
        if self._sdk_client is not None and not self.config.use_sandbox:
            try:
                # SDK surface is not fully stable across versions; fallback on HTTP on failure.
                logger.debug("Using SDK request for %s %s", method, path)
                return self._sdk_request(method, path, params, json_payload)
            except Exception:
                logger.exception("SDK request failed, falling back to raw HTTP")

        try:
            response = self.sync_http.request(method=method, url=url, params=params, json=json_payload, headers=headers, timeout=timeout or self.config.request_timeout_s)
        except httpx.HTTPError as exc:
            raise

        if self._is_retryable_status(response.status_code):
            if response.status_code == 429:
                raise RateLimitError(response.text[:512])
            raise TransientError(response.text[:512])

        if response.status_code >= 400:
            raise CoinbaseError(f"HTTP {response.status_code}: {response.text}")
        if not response.text:
            return {}
        return response.json()

    @staticmethod
    def _normalize_sdk_response(resp: Any) -> Any:
        if hasattr(resp, "to_dict"):
            try:
                return resp.to_dict()
            except Exception:
                pass
        if isinstance(resp, (dict, list)):
            return resp
        if hasattr(resp, "__dict__"):
            return dict(resp.__dict__)
        return resp

    def _sdk_request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, json_payload: Optional[Dict[str, Any]] = None) -> Any:
        # Minimal compatibility shim. Exact method names differ across sdk versions.
        payload = json_payload or {}
        params = params or {}
        # map generic paths
        if method == "GET" and path.startswith("/accounts"):
            if hasattr(self._sdk_client, "get_accounts"):
                return self._normalize_sdk_response(self._sdk_client.get_accounts())
            if hasattr(self._sdk_client, "get_brokerage_accounts"):
                return self._normalize_sdk_response(self._sdk_client.get_brokerage_accounts())
            raise AttributeError("SDK account endpoint missing")

        if method == "GET" and path.startswith("/products/") and path.endswith("/candles"):
            fn = getattr(self._sdk_client, "get_product_candles")
            return self._normalize_sdk_response(fn(**params))

        if method == "GET" and path == "/products/{product_id}/candles":
            fn = getattr(self._sdk_client, "get_product_candles")
            return self._normalize_sdk_response(fn(**params))

        if method == "GET" and path.startswith("/products/") and path.endswith("/book"):
            fn = getattr(self._sdk_client, "get_product_book")
            return self._normalize_sdk_response(fn(**params))

        if method == "POST" and path == "/orders":
            fn = getattr(self._sdk_client, "create_order")
            return self._normalize_sdk_response(fn(**payload))

        if method == "DELETE" and path == "/orders":
            fn = getattr(self._sdk_client, "cancel_orders")
            return self._normalize_sdk_response(fn(**payload))

        if method == "GET" and path == "/orders/historical/fills":
            fn = getattr(self._sdk_client, "get_fills")
            return self._normalize_sdk_response(fn(**params))

        if method == "GET" and path == "/orders":
            fn = getattr(self._sdk_client, "get_orders")
            return self._normalize_sdk_response(fn(**params))

        if method == "GET" and path == "/transactions/summary":
            fn = getattr(self._sdk_client, "get_product_transaction_summary")
            return self._normalize_sdk_response(fn(**params))

        raise NotImplementedError(f"SDK fallback for {method} {path}")

    def get_accounts(self) -> List[Account]:
        data = self._request("GET", "/accounts")
        if isinstance(data, list):
            accounts = data
        else:
            accounts = data.get("accounts") or data.get("response", {}).get("accounts", [])
        out: List[Account] = []
        for row in accounts:
            if not isinstance(row, dict):
                continue
            available_raw = row.get("available_balance")
            if isinstance(available_raw, dict):
                available_value = available_raw.get("value")
            elif isinstance(available_raw, (int, float, str)):
                available_value = available_raw
            else:
                available_value = None

            balance_raw = row.get("balance")
            if isinstance(balance_raw, dict):
                balance_value = balance_raw.get("value")
            elif isinstance(balance_raw, (int, float, str)):
                balance_value = balance_raw
            else:
                balance_value = 0.0

            out.append(
                Account(
                    uuid=str(row.get("uuid") or row.get("account_id") or ""),
                    currency=row.get("currency") or row.get("currency_symbol") or row.get("currency_code") or "",
                    balance=float(available_value if available_value is not None else balance_value or 0.0),
                    available_balance=float(available_value or 0.0),
                )
            )
        return out

    def get_product_candles(self, product_id: str, start: datetime, end: datetime, timeframe: str = "1h", limit: int = 350) -> List[Any]:
        # timeframe accepted by SDK: "1h","1d" -> we also pass granularity and start/end
        def _to_utc_iso(ts: datetime) -> str:
            if ts.tzinfo is not None:
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "product_id": product_id,
            "granularity": timeframe,
            "start": _to_utc_iso(start),
            "end": _to_utc_iso(end),
            "limit": min(limit, 350),
        }
        rows: List[Any] = []
        next_start = start
        # pagination: move forward by fetched rows
        while next_start < end:
            chunk_end = end
            req = self._request(
                "GET",
                "/products/{product_id}/candles".replace("{product_id}", product_id),
                params={**params, "start": _to_utc_iso(next_start), "end": _to_utc_iso(chunk_end)},
            )
            if isinstance(req, list):
                candles = req
            else:
                candles = req.get("candles") or req.get("response", {}).get("candles", [])
            if not candles:
                break
            # Keep deterministic ordering oldest->newest
            def _ts(v: object) -> int:
                if isinstance(v, (list, tuple)):
                    return int(v[0])
                if isinstance(v, dict):
                    return int(v.get("start") or v.get("time") or v.get("timestamp") or 0)
                return int(v)

            candles_sorted = sorted(candles, key=_ts)
            rows.extend(candles_sorted)
            if len(candles) < limit:
                break
            # move next_start to last candle time
            last = candles_sorted[-1]
            if isinstance(last, (list, tuple)):
                last_ts = int(last[0])
            else:
                last_ts = int(last.get("start") or last.get("time") or last.get("timestamp") or 0)
            if isinstance(last_ts, str):
                try:
                    last_ts = int(float(last_ts))
                except Exception:
                    last_ts = 0
            next_start = datetime.fromtimestamp(last_ts) + timedelta(seconds=self._granularity_seconds(timeframe))
        return rows

    def _granularity_seconds(self, tf: str) -> int:
        if tf in {"1h", "hour", "3600"}:
            return 3600
        if tf in {"1d", "day", "86400"}:
            return 86400
        raise ValueError(f"Unsupported granularity: {tf}")

    def get_product_book(self, product_id: str) -> BestBidAsk:
        data = self._request("GET", f"/products/{product_id}/book", params={"product_id": product_id})
        if not isinstance(data, dict):
            return BestBidAsk(bid=0.0, ask=0.0, time=datetime.utcnow())

        # Coinbase may return bids/asks with [price,size,...] or nested dicts.
        if "bids" not in data:
            book = data.get("data", {}).get("bids", [])
            asks = data.get("data", {}).get("asks", [])
        else:
            book = data.get("bids", [])
            asks = data.get("asks", [])

        def _top_px(levels: list, fallback: float = 0.0) -> float:
            if not levels:
                return fallback
            top = levels[0]
            if isinstance(top, (list, tuple)) and len(top) > 0:
                return float(top[0])
            if isinstance(top, dict):
                return float(top.get("price") or top.get("px") or fallback)
            return fallback

        bid = _top_px(book, 0.0)
        ask = _top_px(asks, bid)
        return BestBidAsk(bid=bid, ask=ask, time=datetime.utcnow())

    def create_order(
        self,
        product_id: str,
        side: str,
        size: str,
        client_order_id: str,
        order_type: str = "market",
        limit_price: Optional[str] = None,
        time_in_force: str = "GTC",
        post_only: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": size,
                }
                if order_type == "market"
                else {
                    "limit_limit_gtc": {
                        "base_size": size,
                        "limit_price": limit_price,
                        "post_only": post_only,
                    }
                }
            },
            "time_in_force": time_in_force,
        }
        return self._request("POST", "/orders", json_payload=payload)

    def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        payload = {"client_order_ids": [client_order_id]}
        return self._request("DELETE", "/orders", json_payload=payload)

    def list_orders(self, product_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id
        if status:
            params["status"] = status
        data = self._request("GET", "/orders", params=params)
        if isinstance(data, list):
            orders = data
        else:
            orders = data.get("orders") or data.get("response", {}).get("orders", [])
        return orders

    def list_fills(self, order_id: Optional[str] = None, product_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        params = {"limit": limit}
        if order_id:
            params["order_id"] = order_id
        if product_id:
            params["product_id"] = product_id
        data = self._request("GET", "/orders/historical/fills", params=params)
        if isinstance(data, list):
            return data
        return data.get("fills") or data.get("response", {}).get("fills", [])

    def get_transaction_summary(self, product_id: str) -> FeeSummary:
        data = self._request("GET", "/transactions/summary", params={"product_id": product_id})
        if isinstance(data, dict):
            payload = data.get("summary") or data.get("response", {}).get("summary", data)
            maker = float(payload.get("maker_fee_rate", payload.get("makerFeeRate", 0.0) or 0.0))
            taker = float(payload.get("taker_fee_rate", payload.get("takerFeeRate", 0.0) or 0.0))
        else:
            # unknown shape fallback: use configured defaults in callers
            maker = 0.0
            taker = 0.0
        return FeeSummary(maker_fee_rate=maker, taker_fee_rate=taker)


    def close(self) -> None:
        try:
            self.sync_http.close()
        except Exception:
            pass


class CoinbaseDataLoader:
    """Simple helper for strategy/backtest code using the client methods."""

    def __init__(self, client: RESTClientWrapper):
        self.client = client

    def fetch_fee_rates(self, product: str) -> FeeSummary:
        return self.client.get_transaction_summary(product)

    def fetch_balances(self, product: str, quote_currency: str) -> Dict[str, float]:
        accounts = self.client.get_accounts()
        out: Dict[str, float] = {}
        for a in accounts:
            if a.currency in {product.split("-")[0], product.split("-")[-1], quote_currency}:
                out[a.currency] = a.balance
        return out
