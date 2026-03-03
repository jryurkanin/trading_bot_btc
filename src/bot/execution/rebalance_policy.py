from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


PolicyName = Literal["signal_change_only", "band", "always"]


@dataclass
class RebalanceState:
    last_trade_bucket: Optional[float] = None


class RebalancePolicy:
    def __init__(
        self,
        policy: PolicyName = "signal_change_only",
        min_trade_notional_usd: float = 50.0,
        min_exposure_delta: float = 0.05,
        target_quantization_step: float = 0.25,
    ) -> None:
        self.policy = policy
        self.min_trade_notional_usd = float(min_trade_notional_usd)
        self.min_exposure_delta = float(min_exposure_delta)
        self.target_quantization_step = float(target_quantization_step)
        self.state = RebalanceState()

    def snapshot(self) -> dict:
        """Serialize rebalance state for persistence."""
        return {
            "last_trade_bucket": self.state.last_trade_bucket,
        }

    def restore(self, payload: dict | None) -> None:
        """Restore rebalance state from persistence."""
        if not isinstance(payload, dict):
            return
        if isinstance(payload.get("last_trade_bucket"), (int, float)):
            self.state.last_trade_bucket = float(payload["last_trade_bucket"])

    def quantize_target(self, target: float) -> float:
        t = min(1.0, max(0.0, float(target)))
        step = self.target_quantization_step
        if step <= 0:
            return t
        q = round(t / step) * step
        return min(1.0, max(0.0, q))

    def should_rebalance(self, target: float, current: float, equity_usd: float, now: pd.Timestamp) -> tuple[bool, float, str]:
        target_q = self.quantize_target(target)
        delta = target_q - float(current)
        abs_delta = abs(delta)
        notional = abs_delta * max(0.0, float(equity_usd))

        if notional < self.min_trade_notional_usd:
            return False, target_q, "below_min_notional"

        if self.policy == "always":
            return abs_delta > 1e-12, target_q, "always"

        if self.policy == "band":
            if abs_delta < self.min_exposure_delta:
                return False, target_q, "inside_band"
            return True, target_q, "band_break"

        # signal_change_only
        if self.state.last_trade_bucket is not None and target_q == self.state.last_trade_bucket:
            return False, target_q, "no_signal_bucket_change"
        if abs_delta < self.min_exposure_delta:
            return False, target_q, "delta_too_small"
        return True, target_q, "signal_bucket_change"

    def on_trade(self, now: pd.Timestamp, traded_target_bucket: float) -> None:
        self.state.last_trade_bucket = float(traded_target_bucket)
