from __future__ import annotations

from bot.coinbase_client import RESTClientWrapper
from bot.config import CoinbaseConfig, DataConfig


class StubCoinbaseClient(RESTClientWrapper):
    def __init__(self, payload):
        super().__init__(CoinbaseConfig(), DataConfig())
        self._payload = payload

    def _request(self, method, path, params=None, json_payload=None, timeout=None):  # type: ignore[override]
        return self._payload


def test_get_accounts_uses_total_balance_and_available_balance_separately():
    client = StubCoinbaseClient(
        {
            "accounts": [
                {
                    "uuid": "acct-1",
                    "currency": "BTC",
                    "balance": {"value": "2.5"},
                    "available_balance": {"value": "1.2"},
                }
            ]
        }
    )
    try:
        accounts = client.get_accounts()
        assert len(accounts) == 1
        assert accounts[0].balance == 2.5
        assert accounts[0].available_balance == 1.2
    finally:
        client.close()


def test_get_accounts_falls_back_to_total_when_available_missing():
    client = StubCoinbaseClient(
        {
            "accounts": [
                {
                    "uuid": "acct-2",
                    "currency": "USD",
                    "balance": {"value": "1000.0"},
                }
            ]
        }
    )
    try:
        accounts = client.get_accounts()
        assert len(accounts) == 1
        assert accounts[0].balance == 1000.0
        assert accounts[0].available_balance == 1000.0
    finally:
        client.close()
