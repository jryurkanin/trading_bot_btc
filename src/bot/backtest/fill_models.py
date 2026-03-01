from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional, Literal

import pandas as pd


Side = Literal["BUY", "SELL"]


@dataclass
class BacktestOrder:
    side: Side
    qty: float
    order_type: Literal["market", "limit"] = "market"
    limit_price: Optional[float] = None
    post_only: bool = False


@dataclass
class MarketState:
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_bps: float = 0.0
    impact_bps: float = 0.0
    max_fill_pct_of_volume: float = 0.10  # max 10% of bar volume


@dataclass
class Fill:
    filled: bool
    side: Side
    qty: float
    price: float
    mark_price: float
    is_maker: bool
    reason: str = ""


class FillModel(Protocol):
    name: str

    def fill(self, order: BacktestOrder, bar_t: pd.Series, bar_t1: pd.Series, market_state: MarketState) -> Fill:
        ...


def _safe(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _mark_from_bar(bar_t1: pd.Series) -> float:
    mark = _safe(bar_t1.get("open"), 0.0)
    if mark <= 0:
        mark = _safe(bar_t1.get("close"), 0.0)
    return mark


class NextBarOpenFillModel:
    name = "next_open"

    def __init__(self, slippage_bps: float = 1.0):
        self.slippage_bps = float(slippage_bps)

    def fill(self, order: BacktestOrder, bar_t: pd.Series, bar_t1: pd.Series, market_state: MarketState) -> Fill:
        mark = _mark_from_bar(bar_t1)
        if mark <= 0 or order.qty <= 0:
            return Fill(False, order.side, 0.0, 0.0, mark, is_maker=False, reason="invalid_mark_or_qty")

        if order.side == "BUY":
            px = mark * (1.0 + self.slippage_bps / 10_000.0)
        else:
            px = mark * (1.0 - self.slippage_bps / 10_000.0)
        return Fill(True, order.side, order.qty, px, mark, is_maker=False)


class BidAskBpsFillModel:
    name = "bid_ask"

    def __init__(self, spread_bps: float = 3.0, impact_bps: float = 2.0, partial_fill_enabled: bool = False):
        self.spread_bps = float(spread_bps)
        self.impact_bps = float(impact_bps)
        self.partial_fill_enabled = partial_fill_enabled

    def _bid_ask(self, bar_t1: pd.Series, market_state: MarketState) -> tuple[float, float, float]:
        mark = _mark_from_bar(bar_t1)
        spread_bps = market_state.spread_bps if market_state.spread_bps > 0 else self.spread_bps

        bid = market_state.bid if market_state.bid is not None and market_state.bid > 0 else mark * (1.0 - spread_bps / 20_000.0)
        ask = market_state.ask if market_state.ask is not None and market_state.ask > 0 else mark * (1.0 + spread_bps / 20_000.0)
        return bid, ask, mark

    def _volume_cap(self, order: BacktestOrder, bar_t1: pd.Series, market_state: MarketState, price: float) -> float:
        """Cap fill quantity to a fraction of bar volume."""
        bar_vol = _safe(bar_t1.get("volume"), 0.0)
        if bar_vol <= 0 or price <= 0:
            return order.qty
        max_qty = bar_vol * market_state.max_fill_pct_of_volume
        return min(order.qty, max(0.0, max_qty))

    @staticmethod
    def _partial_fill_ratio(limit_price: float, low: float, high: float, qty: float) -> float:
        """Estimate partial fill based on how far price traded through the limit.

        Assumes fill probability is proportional to how deep into the bar
        range the limit price sits.  If the order is near the edge (barely
        touched), only a small fraction fills.
        """
        bar_range = high - low
        if bar_range <= 0:
            return qty
        # How far through the range did price travel past the limit?
        depth = min(abs(limit_price - low), abs(high - limit_price)) / bar_range
        # Partial fill ratio: at least 10% when touched, up to 100% when deeply through
        ratio = max(0.1, min(1.0, depth * 2.0))
        return max(0.0, qty * ratio)

    @staticmethod
    def _sqrt_impact_bps(qty: float, bar_volume: float, base_impact_bps: float) -> float:
        """Square-root market impact model: impact ~ base * sqrt(qty / volume)."""
        if bar_volume <= 0 or qty <= 0:
            return base_impact_bps
        participation = qty / bar_volume
        return base_impact_bps * max(1.0, (participation / 0.01) ** 0.5)

    def fill(self, order: BacktestOrder, bar_t: pd.Series, bar_t1: pd.Series, market_state: MarketState) -> Fill:
        if order.qty <= 0:
            return Fill(False, order.side, 0.0, 0.0, 0.0, is_maker=False, reason="non_positive_qty")

        bid, ask, mark = self._bid_ask(bar_t1, market_state)
        base_impact = market_state.impact_bps if market_state.impact_bps > 0 else self.impact_bps

        high = _safe(bar_t1.get("high"), mark)
        low = _safe(bar_t1.get("low"), mark)

        # Limit fill-on-touch: require price to trade through limit, not just touch
        if order.order_type == "limit" and order.limit_price is not None:
            lp = float(order.limit_price)
            filled_qty = self._volume_cap(order, bar_t1, market_state, lp)
            if filled_qty <= 0:
                return Fill(False, order.side, 0.0, 0.0, mark, is_maker=True, reason="volume_cap")
            if order.side == "BUY":
                if low < lp:  # strict inequality: must trade through
                    if self.partial_fill_enabled:
                        filled_qty = self._partial_fill_ratio(lp, low, high, filled_qty)
                    return Fill(True, order.side, filled_qty, lp, mark, is_maker=True)
                return Fill(False, order.side, 0.0, 0.0, mark, is_maker=True, reason="limit_not_touched")
            if high > lp:  # strict inequality
                if self.partial_fill_enabled:
                    filled_qty = self._partial_fill_ratio(lp, low, high, filled_qty)
                return Fill(True, order.side, filled_qty, lp, mark, is_maker=True)
            return Fill(False, order.side, 0.0, 0.0, mark, is_maker=True, reason="limit_not_touched")

        # Marketable taker with sqrt market impact
        bar_vol = _safe(bar_t1.get("volume"), 0.0)
        filled_qty = self._volume_cap(order, bar_t1, market_state, mark)
        if filled_qty <= 0:
            return Fill(False, order.side, 0.0, 0.0, mark, is_maker=False, reason="volume_cap")
        impact = self._sqrt_impact_bps(filled_qty, bar_vol, base_impact)
        if order.side == "BUY":
            px = ask * (1.0 + impact / 10_000.0)
        else:
            px = bid * (1.0 - impact / 10_000.0)
        return Fill(True, order.side, filled_qty, px, mark, is_maker=False)


class WorstCaseBarFillModel:
    name = "worst_case_bar"

    def fill(self, order: BacktestOrder, bar_t: pd.Series, bar_t1: pd.Series, market_state: MarketState) -> Fill:
        if order.qty <= 0:
            return Fill(False, order.side, 0.0, 0.0, 0.0, is_maker=False, reason="non_positive_qty")

        mark = _mark_from_bar(bar_t1)
        if order.side == "BUY":
            px = _safe(bar_t1.get("high"), mark)
        else:
            px = _safe(bar_t1.get("low"), mark)
        return Fill(True, order.side, order.qty, px, mark, is_maker=False)


def make_fill_model(name: str, slippage_bps: float = 1.0, spread_bps: float = 3.0, impact_bps: float = 2.0, partial_fill_enabled: bool = False) -> FillModel:
    key = (name or "").lower()
    if key in {"next_open", "nextbaropen", "next_bar_open"}:
        return NextBarOpenFillModel(slippage_bps=slippage_bps)
    if key in {"bid_ask", "bidask", "bid_ask_bps"}:
        return BidAskBpsFillModel(spread_bps=spread_bps, impact_bps=impact_bps, partial_fill_enabled=partial_fill_enabled)
    if key in {"worst_case", "worst_case_bar", "worstcase"}:
        return WorstCaseBarFillModel()
    raise ValueError(f"Unknown fill model: {name}")
