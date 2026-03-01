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
    funding_rate_annual: float = 0.0  # annualized funding/carry cost (e.g. 0.05 = 5%)

    def fee_rate(self, is_maker: bool) -> float:
        return float(self.maker_fee_rate if is_maker else self.taker_fee_rate)

    def fee(self, notional: float, is_maker: bool) -> float:
        return float(max(0.0, notional) * self.fee_rate(is_maker))

    def funding_cost_per_bar(self, position_value: float, bars_per_year: int = 8760) -> float:
        """Compute per-bar funding/carry cost for a given position value."""
        if self.funding_rate_annual == 0.0 or bars_per_year <= 0:
            return 0.0
        return abs(float(position_value)) * self.funding_rate_annual / bars_per_year

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
