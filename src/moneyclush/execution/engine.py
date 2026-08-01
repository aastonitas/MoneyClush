"""Execution engine: order management, inventory-adjusted pricing, split orders.

Converts trade signals into actual orders, managing fills, cancellations,
and the reservation price adjustment based on current inventory.

Status: planning only, nothing here submits a live order. `plan_orders` and
`simulate_fill` are paper-mode building blocks for a *taker* strategy
(TemporalArbitrageStrategy buys at the current ask on whichever leg is
cheap). A two-sided maker engine — resting a bid on Up and a bid on Down at
the same time, sized by model confidence, the way the dashboard's "PAR
MAKER" reading in the TRADING tab describes — does not exist yet. Gap to
close before that can run live, in order:

1. Real CLOB auth. `PolymarketClient._auth_headers` sends POLY_API_KEY/
   SECRET/PASSPHRASE headers, which is not how Polymarket's CLOB
   authenticates. Orders must be EIP-712-signed with the trading wallet's
   private key, then exchanged for L2 API credentials (see py-clob-client).
   `place_limit_order`'s POST to /order will 401 against the real API as
   written today.
2. A maker/ladder strategy. `TemporalArbitrageStrategy` only ever buys at
   the ask (crosses the spread). A two-sided engine needs its own class:
   place a resting bid on both legs, keep bid_up + bid_down under the same
   max_pair_cost gate, and skew size toward the side the model favours
   (reservation_price() below already has the inventory-skew math for this).
3. Order lifecycle. No cancel/replace loop exists — a resting bid must be
   pulled and re-quoted as the book moves, and partial fills on one leg
   need to be tracked so the other leg's target size adjusts.
4. Wiring into poll_loop (dashboard/server.py). The poll loop only reads
   market state today; a live engine needs its own tick that reads
   STATE["markets"], calls the maker strategy, and posts orders through an
   authenticated PolymarketClient.
5. Guardrails before any of this touches real funds: a balance check before
   every order, a max-exposure-per-window cap, and a kill switch. Paper
   mode (no api_key -> `place_limit_order` returns a SIMULATED stub) should
   stay the default until each of the above is verified independently.
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
