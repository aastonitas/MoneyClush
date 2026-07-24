"""Order book signals: imbalance, spread pressure, executable depth."""

from __future__ import annotations

from moneyclush.data.models import OrderBookSnapshot


def order_book_imbalance(book: OrderBookSnapshot) -> float:
    """Bid/ask volume imbalance in [-1, 1]. Positive = more bid pressure."""
    total_bid = book.total_bid_volume()
    total_ask = book.total_ask_volume()
    total = total_bid + total_ask
    if total == 0:
        return 0.0
    return (total_bid - total_ask) / total


def spread_ratio(book: OrderBookSnapshot) -> float:
    """Spread as fraction of midpoint. Lower = tighter market."""
    if book.best_bid is None or book.best_ask is None:
        return 1.0
    mid = (book.best_bid + book.best_ask) / 2
    if mid == 0:
        return 1.0
    return (book.best_ask - book.best_bid) / mid


def executable_depth(book: OrderBookSnapshot, side: str, max_price: float) -> float:
    """Total shares available up to `max_price` on the given side."""
    levels = book.asks if side == "buy" else book.bids
    total = 0.0
    for level in levels:
        if side == "buy" and level.price > max_price:
            break
        if side == "sell" and level.price < max_price:
            break
        total += level.size
    return total


def combined_pair_cost(
    book_up: OrderBookSnapshot,
    book_down: OrderBookSnapshot,
    quantity: float,
) -> float | None:
    """VWAP cost to acquire `quantity` shares of both Up and Down.

    Returns the cost per pair, or None if insufficient liquidity.
    """
    cost_up = book_up.executable_cost("buy", quantity)
    cost_down = book_down.executable_cost("buy", quantity)
    if cost_up is None or cost_down is None:
        return None
    return cost_up + cost_down
