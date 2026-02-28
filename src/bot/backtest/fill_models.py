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

    def __init__(self, spread_bps: float = 3.0, impact_bps: float = 2.0):
        self.spread_bps = float(spread_bps)
        self.impact_bps = float(impact_bps)

    def _bid_ask(self, bar_t1: pd.Series, market_state: MarketState) -> tuple[float, float, float]:
        mark = _mark_from_bar(bar_t1)
        spread_bps = market_state.spread_bps if market_state.spread_bps > 0 else self.spread_bps

        bid = market_state.bid if market_state.bid is not None and market_state.bid > 0 else mark * (1.0 - spread_bps / 20_000.0)
        ask = market_state.ask if market_state.ask is not None and market_state.ask > 0 else mark * (1.0 + spread_bps / 20_000.0)
        return bid, ask, mark

    def fill(self, order: BacktestOrder, bar_t: pd.Series, bar_t1: pd.Series, market_state: MarketState) -> Fill:
        if order.qty <= 0:
            return Fill(False, order.side, 0.0, 0.0, 0.0, is_maker=False, reason="non_positive_qty")

        bid, ask, mark = self._bid_ask(bar_t1, market_state)
        impact = market_state.impact_bps if market_state.impact_bps > 0 else self.impact_bps

        high = _safe(bar_t1.get("high"), mark)
        low = _safe(bar_t1.get("low"), mark)

        # Limit fill-on-touch, conservative price at limit.
        if order.order_type == "limit" and order.limit_price is not None:
            lp = float(order.limit_price)
            if order.side == "BUY":
                if low <= lp:
                    return Fill(True, order.side, order.qty, lp, mark, is_maker=True)
                return Fill(False, order.side, 0.0, 0.0, mark, is_maker=True, reason="limit_not_touched")
            if high >= lp:
                return Fill(True, order.side, order.qty, lp, mark, is_maker=True)
            return Fill(False, order.side, 0.0, 0.0, mark, is_maker=True, reason="limit_not_touched")

        # Marketable taker
        if order.side == "BUY":
            px = ask * (1.0 + impact / 10_000.0)
        else:
            px = bid * (1.0 - impact / 10_000.0)
        return Fill(True, order.side, order.qty, px, mark, is_maker=False)


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


def make_fill_model(name: str, slippage_bps: float = 1.0, spread_bps: float = 3.0, impact_bps: float = 2.0) -> FillModel:
    key = (name or "").lower()
    if key in {"next_open", "nextbaropen", "next_bar_open"}:
        return NextBarOpenFillModel(slippage_bps=slippage_bps)
    if key in {"bid_ask", "bidask", "bid_ask_bps"}:
        return BidAskBpsFillModel(spread_bps=spread_bps, impact_bps=impact_bps)
    if key in {"worst_case", "worst_case_bar", "worstcase"}:
        return WorstCaseBarFillModel()
    raise ValueError(f"Unknown fill model: {name}")
