"""Price momentum and trade flow signals."""

from __future__ import annotations

from moneyclush.data.models import MarketState, Trade


def price_momentum(
    current_price: float,
    opening_price: float,
) -> float:
    """Directional momentum as percentage distance from opening price.

    Positive = BTC above open (favors Up), negative = below (favors Down).
    """
    if opening_price == 0:
        return 0.0
    return (current_price - opening_price) / opening_price


def trade_flow_imbalance(trades: list[Trade], window_ms: int = 30_000) -> float:
    """Net buy pressure from recent trades in [-1, 1].

    Looks at trades within `window_ms` of the most recent trade.
    """
    if not trades:
        return 0.0

    cutoff = trades[-1].timestamp_ms - window_ms
    recent = [t for t in trades if t.timestamp_ms >= cutoff]
    if not recent:
        return 0.0

    buy_vol = sum(t.size for t in recent if t.is_taker_buy)
    sell_vol = sum(t.size for t in recent if not t.is_taker_buy)
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return (buy_vol - sell_vol) / total


def volatility_estimate(
    prices: list[float],
) -> float:
    """Simple realized volatility from a list of sequential prices."""
    if len(prices) < 2:
        return 0.0
    returns = [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] != 0
    ]
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return variance**0.5


def time_pressure(seconds_remaining: float, window_seconds: float = 300) -> float:
    """Urgency signal in [0, 1]. Closer to 1 = less time remaining."""
    if window_seconds <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - seconds_remaining / window_seconds))
