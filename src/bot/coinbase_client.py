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
from email.utils import parsedate_to_datetime
import math

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


class AuthError(CoinbaseError):
    pass


class ClockSkewError(TransientError):
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


@dataclass(frozen=True)
class ProductConstraints:
    product_id: str
    base_increment: float
    quote_increment: float
    base_min_size: float
    quote_min_size: float
    price_increment: float
    base_max_size: Optional[float]
    min_notional: float


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
        self._clock_skew_seconds: float = 0.0
        self._sdk_client = self._init_sdk_client()
        self._sdk_disabled: bool = False
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
        ts = str(int(time.time() + self._clock_skew_seconds))
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
        return status in {408, 409, 425, 429} or 500 <= status < 600

    def _update_clock_skew_from_headers(self, headers: Dict[str, Any]) -> None:
        date_value = headers.get("Date") if isinstance(headers, dict) else None
        if not date_value:
            return
        try:
            server_dt = parsedate_to_datetime(str(date_value)).astimezone(timezone.utc)
            local_dt = datetime.now(tz=timezone.utc)
            skew = (server_dt - local_dt).total_seconds()
            if abs(skew) <= self.config.max_clock_skew_s:
                self._clock_skew_seconds = skew
        except Exception:
            return

    def _classify_http_error(self, status_code: int, text: str, headers: Dict[str, Any]) -> Exception:
        msg = (text or "")[:512]
        if status_code == 429:
            return RateLimitError(msg)

        if self._is_retryable_status(status_code):
            return TransientError(msg)

        if status_code in {401, 403}:
            low = msg.lower()
            if any(k in low for k in ["timestamp", "clock", "skew", "ahead", "behind", "expired"]):
                self._update_clock_skew_from_headers(headers)
                return ClockSkewError(msg)
            return AuthError(msg)

        return CoinbaseError(f"HTTP {status_code}: {msg}")

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
        if self._sdk_client is not None and not self.config.use_sandbox and not self._sdk_disabled:
            try:
                # SDK surface is not fully stable across versions; fallback on HTTP on failure.
                logger.debug("Using SDK request for %s %s", method, path)
                return self._sdk_request(method, path, params, json_payload)
            except Exception as exc:
                # Disable SDK after first compatibility failure to avoid noisy repeats.
                self._sdk_disabled = True
                logger.warning(
                    "SDK request failed (%s); disabling SDK for this run and falling back to raw HTTP",
                    exc.__class__.__name__,
                )

        try:
            response = self.sync_http.request(method=method, url=url, params=params, json=json_payload, headers=headers, timeout=timeout or self.config.request_timeout_s)
        except httpx.HTTPError as exc:
            raise

        self._update_clock_skew_from_headers(dict(response.headers))

        if response.status_code >= 400:
            raise self._classify_http_error(response.status_code, response.text, dict(response.headers))
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

    @staticmethod
    def _sdk_public_candle_granularity(value: Any) -> Any:
        mapping = {
            "1h": "ONE_HOUR",
            "hour": "ONE_HOUR",
            "3600": "ONE_HOUR",
            "one_hour": "ONE_HOUR",
            "1d": "ONE_DAY",
            "day": "ONE_DAY",
            "86400": "ONE_DAY",
            "one_day": "ONE_DAY",
        }
        raw = str(value).strip()
        key = raw.lower()
        if raw in {"ONE_HOUR", "ONE_DAY"}:
            return raw
        return mapping.get(key, value)

    @staticmethod
    def _sdk_public_candle_ts(value: Any) -> Any:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, datetime):
            ts = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return int(ts.timestamp())
        if isinstance(value, str):
            try:
                ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                return int(ts.timestamp())
            except Exception:
                return value
        return value

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
            if hasattr(self._sdk_client, "get_product_candles"):
                fn = getattr(self._sdk_client, "get_product_candles")
                return self._normalize_sdk_response(fn(**params))
            if hasattr(self._sdk_client, "get_public_candles"):
                fn = getattr(self._sdk_client, "get_public_candles")
                sdk_params = dict(params)
                sdk_params["granularity"] = self._sdk_public_candle_granularity(sdk_params.get("granularity"))
                sdk_params["start"] = self._sdk_public_candle_ts(sdk_params.get("start"))
                sdk_params["end"] = self._sdk_public_candle_ts(sdk_params.get("end"))
                return self._normalize_sdk_response(fn(**sdk_params))
            raise AttributeError("SDK candle endpoint missing")

        if method == "GET" and path == "/products/{product_id}/candles":
            if hasattr(self._sdk_client, "get_product_candles"):
                fn = getattr(self._sdk_client, "get_product_candles")
                return self._normalize_sdk_response(fn(**params))
            if hasattr(self._sdk_client, "get_public_candles"):
                fn = getattr(self._sdk_client, "get_public_candles")
                sdk_params = dict(params)
                sdk_params["granularity"] = self._sdk_public_candle_granularity(sdk_params.get("granularity"))
                sdk_params["start"] = self._sdk_public_candle_ts(sdk_params.get("start"))
                sdk_params["end"] = self._sdk_public_candle_ts(sdk_params.get("end"))
                return self._normalize_sdk_response(fn(**sdk_params))
            raise AttributeError("SDK candle endpoint missing")

        if method == "GET" and path.startswith("/products/") and path.endswith("/book"):
            fn = getattr(self._sdk_client, "get_product_book")
            return self._normalize_sdk_response(fn(**params))

        if method == "GET" and path == "/best_bid_ask" and hasattr(self._sdk_client, "get_best_bid_ask"):
            fn = getattr(self._sdk_client, "get_best_bid_ask")
            return self._normalize_sdk_response(fn(**params))

        if method == "GET" and path.startswith("/products/") and "/candles" not in path and "/book" not in path:
            if hasattr(self._sdk_client, "get_product"):
                fn = getattr(self._sdk_client, "get_product")
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

            total_balance = float(balance_value or 0.0)
            available_balance = float(available_value) if available_value is not None else total_balance

            out.append(
                Account(
                    uuid=str(row.get("uuid") or row.get("account_id") or ""),
                    currency=row.get("currency") or row.get("currency_symbol") or row.get("currency_code") or "",
                    balance=total_balance,
                    available_balance=available_balance,
                )
            )
        return out

    def get_product_candles(self, product_id: str, start: datetime, end: datetime, timeframe: str = "1h", limit: int = 350) -> List[Any]:
        # timeframe accepted by SDK: "1h","1d" -> we also pass granularity and start/end
        def _to_utc_naive(ts: datetime) -> datetime:
            if ts.tzinfo is None:
                return ts
            return ts.astimezone(timezone.utc).replace(tzinfo=None)

        def _to_utc_iso(ts: datetime) -> str:
            return _to_utc_naive(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

        start = _to_utc_naive(start)
        end = _to_utc_naive(end)

        params = {
            "product_id": product_id,
            "granularity": timeframe,
            "start": _to_utc_iso(start),
            "end": _to_utc_iso(end),
            "limit": min(limit, 350),
        }

        public_granularity = {
            "1h": "ONE_HOUR",
            "1d": "ONE_DAY",
            "hour": "ONE_HOUR",
            "day": "ONE_DAY",
        }.get(timeframe, "ONE_HOUR")
        rows: List[Any] = []
        next_start = start
        prev_start: Optional[datetime] = None
        safety_iter = 0
        # pagination: move forward by fetched rows
        while next_start < end:
            safety_iter += 1
            if safety_iter > 10000:
                raise RuntimeError("candle pagination safety stop triggered")
            chunk_end = end
            req_params = {**params, "start": _to_utc_iso(next_start), "end": _to_utc_iso(chunk_end)}
            try:
                req = self._request(
                    "GET",
                    "/products/{product_id}/candles".replace("{product_id}", product_id),
                    params=req_params,
                )
                if isinstance(req, list):
                    candles = req
                else:
                    candles = req.get("candles") or req.get("response", {}).get("candles", [])
            except AuthError:
                # Public fallback endpoint for backtests when brokerage auth is unavailable.
                public_resp = self.sync_http.request(
                    method="GET",
                    url=f"/market/products/{product_id}/candles",
                    params={
                        "start": int(next_start.replace(tzinfo=timezone.utc).timestamp()),
                        "end": int(chunk_end.replace(tzinfo=timezone.utc).timestamp()),
                        "granularity": public_granularity,
                    },
                    headers={},
                    timeout=self.config.request_timeout_s,
                )
                if public_resp.status_code >= 400:
                    raise self._classify_http_error(public_resp.status_code, public_resp.text, dict(public_resp.headers))
                payload = public_resp.json() if public_resp.text else {}
                candles = payload.get("candles") if isinstance(payload, dict) else []
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
            # Keep pagination in UTC regardless of host local timezone.
            next_candidate = datetime.fromtimestamp(last_ts, tz=timezone.utc).replace(tzinfo=None) + timedelta(
                seconds=self._granularity_seconds(timeframe)
            )
            if prev_start is not None and next_candidate <= prev_start:
                # no forward progress from provider payload; avoid infinite loop
                break
            prev_start = next_start
            next_start = next_candidate
        return rows

    def _granularity_seconds(self, tf: str) -> int:
        if tf in {"1h", "hour", "3600"}:
            return 3600
        if tf in {"1d", "day", "86400"}:
            return 86400
        raise ValueError(f"Unsupported granularity: {tf}")

    def get_best_bid_ask(self, product_id: str) -> BestBidAsk:
        # Preferred endpoint for top-of-book snapshots.
        try:
            data = self._request("GET", "/best_bid_ask", params={"product_ids": product_id})
            books = []
            if isinstance(data, dict):
                books = data.get("pricebooks") or data.get("price_books") or data.get("books") or []
            if books:
                book = books[0]
                bids = book.get("bids") or []
                asks = book.get("asks") or []

                def _px(levels: list[dict[str, Any]]) -> float:
                    if not levels:
                        return 0.0
                    top = levels[0]
                    return self._as_float(top.get("price") or top.get("px"), 0.0)

                bid = _px(bids)
                ask = _px(asks)
                return BestBidAsk(bid=bid, ask=ask if ask > 0 else bid, time=datetime.utcnow())
        except Exception:
            pass

        # fallback to product book endpoint
        return self.get_product_book(product_id)

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

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def get_product(self, product_id: str) -> Dict[str, Any]:
        data = self._request("GET", f"/products/{product_id}", params={"product_id": product_id})
        if isinstance(data, dict):
            return data.get("product") or data.get("response", {}).get("product", data)
        return {}

    def get_product_constraints(self, product_id: str) -> ProductConstraints:
        data = self.get_product(product_id)
        base_increment = self._as_float(data.get("base_increment"), 1e-8)
        quote_increment = self._as_float(data.get("quote_increment"), 0.01)
        price_increment = self._as_float(data.get("price_increment"), quote_increment)
        base_min_size = self._as_float(data.get("base_min_size"), base_increment)
        quote_min_size = self._as_float(data.get("quote_min_size"), 0.0)
        base_max_size_raw = data.get("base_max_size")
        base_max_size = self._as_float(base_max_size_raw, 0.0) if base_max_size_raw is not None else None

        min_notional = max(
            self._as_float(data.get("min_market_funds"), 0.0),
            self._as_float(data.get("quote_min_size"), 0.0),
            self._as_float(data.get("min_order_size"), 0.0),
        )
        return ProductConstraints(
            product_id=product_id,
            base_increment=base_increment if base_increment > 0 else 1e-8,
            quote_increment=quote_increment if quote_increment > 0 else 0.01,
            base_min_size=base_min_size if base_min_size > 0 else base_increment,
            quote_min_size=max(0.0, quote_min_size),
            price_increment=price_increment if price_increment > 0 else quote_increment,
            base_max_size=base_max_size if base_max_size and base_max_size > 0 else None,
            min_notional=max(0.0, min_notional),
        )

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
