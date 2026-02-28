from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["BUY", "SELL"]


@dataclass
class CostModel:
    maker_fee_rate: float
    taker_fee_rate: float
    spread_bps: float = 0.0
    impact_bps: float = 0.0

    def fee_rate(self, is_maker: bool) -> float:
        return float(self.maker_fee_rate if is_maker else self.taker_fee_rate)

    def fee(self, notional: float, is_maker: bool) -> float:
        return float(max(0.0, notional) * self.fee_rate(is_maker))

    @staticmethod
    def slippage_cost(side: Side, trade_price: float, mark_price: float, qty: float) -> float:
        qty = max(0.0, float(qty))
        if qty <= 0:
            return 0.0
        if side == "BUY":
            return max(0.0, (float(trade_price) - float(mark_price)) * qty)
        return max(0.0, (float(mark_price) - float(trade_price)) * qty)

    @staticmethod
    def slippage_bps(side: Side, trade_price: float, mark_price: float) -> float:
        if mark_price <= 0:
            return 0.0
        if side == "BUY":
            return (float(trade_price) / float(mark_price) - 1.0) * 10_000.0
        return (float(mark_price) / float(trade_price) - 1.0) * 10_000.0 if trade_price > 0 else 0.0
