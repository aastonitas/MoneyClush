"""Base strategy interface for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from moneyclush.data.models import MarketState, OutcomeSide, Position
from moneyclush.pricing.fair_value import FairValueResult


class SignalAction(str, Enum):
    BUY_UP = "buy_up"
    BUY_DOWN = "buy_down"
    SELL_UP = "sell_up"
    SELL_DOWN = "sell_down"
    HOLD = "hold"


@dataclass
class TradeSignal:
    action: SignalAction
    side: OutcomeSide
    target_price: float
    target_size: float
    edge: float
    confidence: float
    reason: str


class Strategy(ABC):
    """Base class all strategies must implement."""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        state: MarketState,
        fair_value: FairValueResult,
        position: Position,
    ) -> Optional[TradeSignal]:
        """Evaluate current market state and return a trade signal or None."""
        ...

    @abstractmethod
    def should_exit(
        self,
        state: MarketState,
        fair_value: FairValueResult,
        position: Position,
    ) -> Optional[TradeSignal]:
        """Check if the current position should be exited."""
        ...
