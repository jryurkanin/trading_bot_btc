from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd


@dataclass
class OrderIntent:
    product: str
    target_fraction: float
    side: str  # BUY/SELL
    size_fraction: float
    order_type: str = "limit"
    price: Optional[float] = None
    client_order_id: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyDecision:
    timestamp: pd.Timestamp
    target_fraction: float
    regime: str
    sub_regime: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    desired_orders: List[OrderIntent] = field(default_factory=list)


class Strategy(Protocol):
    name: str

    def compute_target_position(self, context: Dict[str, Any]) -> StrategyDecision: ...

    def desired_orders(
        self,
        current_fraction: float,
        target_fraction: float,
        price: float,
        product: str,
        now: pd.Timestamp,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[OrderIntent]:
        ...


class BaseStrategy:
    name = "base"

    def compute_target_position(self, context: Dict[str, Any]) -> StrategyDecision:
        raise NotImplementedError

    def desired_orders(
        self,
        current_fraction: float,
        target_fraction: float,
        price: float,
        product: str,
        now: pd.Timestamp,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[OrderIntent]:
        context = context or {}
        if abs(target_fraction - current_fraction) < 1e-9:
            return []
        side = "BUY" if target_fraction > current_fraction else "SELL"
        return [
            OrderIntent(
                product=product,
                target_fraction=target_fraction,
                size_fraction=target_fraction - current_fraction,
                side=side,
                order_type=context.get("order_type", "limit"),
                metadata={"from_strategy": self.name},
            )
        ]
