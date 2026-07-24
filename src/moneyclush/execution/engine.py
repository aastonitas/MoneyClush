"""Execution engine: order management, inventory-adjusted pricing, split orders.

Converts trade signals into actual orders, managing fills, cancellations,
and the reservation price adjustment based on current inventory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from moneyclush.data.models import MarketState, OrderBookSnapshot, OutcomeSide, Position
from moneyclush.strategies.base import SignalAction, TradeSignal

log = structlog.get_logger()


@dataclass
class OrderPlan:
    side: OutcomeSide
    action: str
    limit_price: float
    size: float
    order_type: str = "GTC"


@dataclass
class ExecutionResult:
    filled: bool
    fill_price: float
    fill_size: float
    order_id: str = ""
    slippage: float = 0.0


def reservation_price(
    fair_value: float,
    inventory_imbalance: float,
    risk_aversion: float = 0.5,
    volatility: float = 0.01,
    time_remaining_frac: float = 1.0,
) -> float:
    """Inventory-adjusted reservation price.

    Shifts the price the bot is willing to pay based on how much
    uncovered inventory it already holds. More inventory on one side
    → lower willingness to buy more of that side.
    """
    adjustment = inventory_imbalance * risk_aversion * (volatility**2) * time_remaining_frac
    return fair_value - adjustment


@dataclass
class ExecutionEngine:
    """Converts trade signals into executable order plans."""

    split_count: int = 3
    max_slippage: float = 0.02
    risk_aversion: float = 0.5

    def plan_orders(
        self,
        signal: TradeSignal,
        state: MarketState,
        position: Position,
    ) -> list[OrderPlan]:
        """Split a trade signal into multiple limit orders."""
        if signal.action == SignalAction.HOLD:
            return []

        side = signal.side
        book = state.book_up if side == OutcomeSide.UP else state.book_down

        inv_imbalance = (position.shares_up - position.shares_down) / max(
            position.shares_up + position.shares_down, 1
        )
        if side == OutcomeSide.DOWN:
            inv_imbalance = -inv_imbalance

        time_frac = state.seconds_remaining / max(
            state.info.duration_minutes * 60, 1
        )

        res_price = reservation_price(
            fair_value=signal.target_price,
            inventory_imbalance=inv_imbalance,
            risk_aversion=self.risk_aversion,
            volatility=0.01,
            time_remaining_frac=time_frac,
        )

        total_size = signal.target_size
        chunk = total_size / self.split_count
        orders: list[OrderPlan] = []

        for i in range(self.split_count):
            price_offset = i * 0.005
            limit = round(min(res_price + price_offset, signal.target_price + self.max_slippage), 4)
            limit = max(0.01, min(0.99, limit))

            orders.append(
                OrderPlan(
                    side=side,
                    action="BUY" if signal.action in (SignalAction.BUY_UP, SignalAction.BUY_DOWN) else "SELL",
                    limit_price=limit,
                    size=round(chunk, 2),
                    order_type="GTC" if time_frac > 0.2 else "GTD",
                )
            )

        return orders

    def simulate_fill(
        self,
        order: OrderPlan,
        book: OrderBookSnapshot,
    ) -> ExecutionResult:
        """Simulate order execution against current order book (for paper/backtest)."""
        levels = book.asks if order.action == "BUY" else book.bids
        remaining = order.size
        total_cost = 0.0
        filled_size = 0.0

        for level in levels:
            if order.action == "BUY" and level.price > order.limit_price:
                break
            if order.action == "SELL" and level.price < order.limit_price:
                break

            fill = min(remaining, level.size)
            total_cost += fill * level.price
            filled_size += fill
            remaining -= fill
            if remaining <= 0:
                break

        if filled_size == 0:
            return ExecutionResult(filled=False, fill_price=0, fill_size=0)

        avg_price = total_cost / filled_size
        slippage = abs(avg_price - order.limit_price)

        return ExecutionResult(
            filled=True,
            fill_price=avg_price,
            fill_size=filled_size,
            order_id=f"sim-{int(time.time() * 1000)}",
            slippage=slippage,
        )
