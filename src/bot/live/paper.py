from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

import pandas as pd

from ..execution.order_router import SimulatedFill
from ..execution.state_store import BotStateStore


@dataclass
class PaperPortfolio:
    usd: float
    btc: float = 0.0

    def equity(self, price: float) -> float:
        return self.usd + self.btc * price


class PaperTrader:
    """Paper broker for --paper mode. Deterministic execution + optional slippage."""

    def __init__(self, state: BotStateStore, maker_fee_bps: float = 10.0, taker_fee_bps: float = 25.0, slippage_bps: float = 5.0, spread_bps: float = 15.0):
        self.state = state
        self.maker_fee = maker_fee_bps / 10000.0
        self.taker_fee = taker_fee_bps / 10000.0
        self.slippage_bps = slippage_bps
        self.spread_bps = spread_bps
        self.portfolio = PaperPortfolio(usd=10_000.0, btc=0.0)

    def get_portfolio(self) -> PaperPortfolio:
        return self.portfolio

    def set_portfolio(self, usd: float, btc: float = 0.0) -> None:
        self.portfolio = PaperPortfolio(usd=float(usd), btc=float(btc))

    def _fill_price(self, side: str, close: float, high: float, low: float) -> float:
        if side == "BUY":
            return max(0.0, close * (1 + self.slippage_bps / 10000.0))
        return max(0.0, close * (1 - self.slippage_bps / 10000.0))

    def execute_fraction(self, target_fraction: float, now: datetime, latest_close: float, latest_high: float, latest_low: float) -> List[SimulatedFill]:
        equity = self.portfolio.equity(latest_close)
        current_fraction = max(0.0, min(1.0, self.portfolio.btc * latest_close / max(equity, 1e-12)))
        if abs(target_fraction - current_fraction) < 1e-9:
            return []

        side = "BUY" if target_fraction > current_fraction else "SELL"
        delta = abs(target_fraction - current_fraction)
        usd_notional = delta * equity
        px = self._fill_price(side, latest_close, latest_high, latest_low)

        fee_rate = self.taker_fee
        if side == "BUY":
            # Never spend beyond available cash including fees.
            max_notional = self.portfolio.usd / (1.0 + fee_rate)
            usd_notional = max(0.0, min(usd_notional, max_notional))
            qty = usd_notional / max(px, 1e-12)
            fee = usd_notional * fee_rate
            self.portfolio.usd -= (usd_notional + fee)
            self.portfolio.btc += qty
        else:
            # Never sell more BTC than held.
            max_qty = max(0.0, self.portfolio.btc)
            qty = min(usd_notional / max(px, 1e-12), max_qty)
            usd_notional = qty * px
            fee = usd_notional * fee_rate
            self.portfolio.btc -= qty
            self.portfolio.usd += max(0.0, usd_notional - fee)

        if qty <= 0 or usd_notional <= 0:
            return []

        fill = SimulatedFill(
            side=side,
            size=qty,
            price=px,
            notional=usd_notional,
            fee_rate=fee_rate,
            is_taker=True,
            fee=fee,
            ts=now,
        )

        self.state.log_decision(
            int(now.timestamp()),
            "BTC-USD",
            {
                "mode": "paper",
                "side": side,
                "size": qty,
                "price": px,
                "target_fraction": target_fraction,
                "current_fraction_before": current_fraction,
                "equity": equity,
                "fee": fee,
            },
        )
        return [fill]
