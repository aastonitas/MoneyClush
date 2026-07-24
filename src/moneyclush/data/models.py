from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OutcomeSide(str, Enum):
    UP = "Up"
    DOWN = "Down"


class OrderBookLevel(BaseModel):
    price: float = Field(ge=0.0, le=1.0)
    size: float = Field(ge=0.0)


class OrderBookSnapshot(BaseModel):
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    timestamp_ms: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def total_bid_volume(self) -> float:
        return sum(level.size for level in self.bids)

    def total_ask_volume(self) -> float:
        return sum(level.size for level in self.asks)

    def executable_cost(self, side: str, quantity: float) -> Optional[float]:
        """VWAP cost to fill `quantity` shares on the given side."""
        levels = self.asks if side == "buy" else self.bids
        remaining = quantity
        total_cost = 0.0
        for level in levels:
            fill = min(remaining, level.size)
            total_cost += fill * level.price
            remaining -= fill
            if remaining <= 0:
                break
        if remaining > 0:
            return None
        return total_cost / quantity


class Trade(BaseModel):
    side: OutcomeSide
    price: float
    size: float
    timestamp_ms: int
    is_taker_buy: bool = True


class MarketInfo(BaseModel):
    condition_id: str
    token_id_up: str
    token_id_down: str
    question: str = ""
    duration_minutes: int = 5
    opening_price_btc: Optional[float] = None
    open_timestamp_ms: int = 0
    close_timestamp_ms: int = 0


class MarketState(BaseModel):
    """Full snapshot of a single BTC Up/Down market at a point in time."""

    info: MarketInfo
    book_up: OrderBookSnapshot = Field(default_factory=OrderBookSnapshot)
    book_down: OrderBookSnapshot = Field(default_factory=OrderBookSnapshot)
    recent_trades: list[Trade] = Field(default_factory=list)
    btc_price: float = 0.0
    btc_price_timestamp_ms: int = 0
    timestamp_ms: int = 0

    @property
    def seconds_remaining(self) -> float:
        if self.info.close_timestamp_ms <= 0:
            return 0.0
        remaining = (self.info.close_timestamp_ms - self.timestamp_ms) / 1000.0
        return max(remaining, 0.0)

    @property
    def btc_above_open(self) -> Optional[bool]:
        if self.info.opening_price_btc is None:
            return None
        return self.btc_price > self.info.opening_price_btc

    @property
    def btc_distance_from_open_pct(self) -> Optional[float]:
        if self.info.opening_price_btc is None or self.info.opening_price_btc == 0:
            return None
        return (
            (self.btc_price - self.info.opening_price_btc)
            / self.info.opening_price_btc
        )


class Position(BaseModel):
    market_condition_id: str
    shares_up: float = 0.0
    shares_down: float = 0.0
    avg_cost_up: float = 0.0
    avg_cost_down: float = 0.0

    @property
    def paired_shares(self) -> float:
        return min(self.shares_up, self.shares_down)

    @property
    def directional_side(self) -> Optional[OutcomeSide]:
        if self.shares_up > self.shares_down:
            return OutcomeSide.UP
        elif self.shares_down > self.shares_up:
            return OutcomeSide.DOWN
        return None

    @property
    def directional_exposure(self) -> float:
        return abs(self.shares_up - self.shares_down)

    @property
    def paired_cost(self) -> float:
        """Cost per paired share. If > 1.0, the pair loses money."""
        if self.paired_shares == 0:
            return 0.0
        return self.avg_cost_up + self.avg_cost_down

    @property
    def total_invested(self) -> float:
        return (self.shares_up * self.avg_cost_up) + (
            self.shares_down * self.avg_cost_down
        )

    def add_fill(self, side: OutcomeSide, price: float, size: float) -> None:
        if side == OutcomeSide.UP:
            total_cost = self.shares_up * self.avg_cost_up + size * price
            self.shares_up += size
            self.avg_cost_up = total_cost / self.shares_up if self.shares_up > 0 else 0
        else:
            total_cost = self.shares_down * self.avg_cost_down + size * price
            self.shares_down += size
            self.avg_cost_down = (
                total_cost / self.shares_down if self.shares_down > 0 else 0
            )

    def pnl_if_resolves(self, winner: OutcomeSide) -> float:
        """PnL assuming resolution pays $1 per winning share."""
        payout = self.shares_up if winner == OutcomeSide.UP else self.shares_down
        return payout - self.total_invested
