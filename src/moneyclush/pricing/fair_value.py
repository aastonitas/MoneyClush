"""Fair value engine for BTC Up/Down outcomes.

The dominant term is not sentiment or order book pressure — it is the
probability that a driftless random walk ends above its starting point
given where it is now and how much time is left.

    P(Up) = Phi( d / sigma_remaining )

where d is the return from the window's opening price to the current
price, and sigma_remaining is the volatility scaled to the time left in
the window. As time runs out sigma_remaining shrinks, so the same price
distance implies a far more decisive probability.

This matches observed Polymarket pricing closely. A logistic/Bayesian
model over raw momentum does not: it ignores time remaining entirely and
prices a 5 bps move the same with 4 minutes left as with 10 seconds left.

Order book imbalance and trade flow are applied as small adjustments on
top, never as primary drivers — visible liquidity is weak evidence about
where a price will close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from moneyclush.data.models import MarketState
from moneyclush.signals.momentum import trade_flow_imbalance
from moneyclush.signals.order_book import order_book_imbalance


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class FairValueResult:
    posterior_up: float
    posterior_down: float
    best_ask_up: float | None
    best_ask_down: float | None
    gross_edge_up: float
    gross_edge_down: float
    net_edge_up: float
    net_edge_down: float
    # diagnostics
    distance_bps: float = 0.0
    sigma_remaining_bps: float = 0.0
    z_score: float = 0.0
    base_probability: float = 0.5
    valid: bool = True


def brownian_probability(
    distance: float,
    sigma_per_minute: float,
    seconds_remaining: float,
    tracking_error: float = 0.0001,
) -> float:
    """P(price ends above opening) for a driftless random walk.

    `distance` is the current return relative to the opening price.

    `tracking_error` is the irreducible uncertainty between our price
    source and the one the market actually resolves against (Chainlink).
    Measured at ~1 bp: across 285 resolved windows, OKX called the winner
    correctly 95.4% of the time, and every miss came from a window that
    closed within 0.86 bps of its open.

    Without this floor the model drives to 0 or 1 as time runs out, which
    is a claim of certainty the data does not support. The two sources of
    uncertainty are independent, so their variances add.
    """
    sigma_remaining = sigma_per_minute * math.sqrt(
        max(seconds_remaining, 0.0) / 60.0
    )
    sigma_effective = math.sqrt(sigma_remaining**2 + tracking_error**2)

    if sigma_effective <= 0:
        return 1.0 if distance >= 0 else 0.0

    return _phi(distance / sigma_effective)


def calculate_executable_edge(
    fair_value: float,
    expected_fill_price: float,
    trading_fee: float = 0.012,
    expected_slippage: float = 0.006,
    safety_buffer: float = 0.010,
) -> tuple[float, float]:
    """Returns (gross_edge, net_edge) after deducting all costs."""
    gross = fair_value - expected_fill_price
    net = gross - trading_fee - expected_slippage - safety_buffer
    return gross, net


@dataclass
class FairValueEngine:
    """Prices Up/Down outcomes from time-scaled distance to the opening price."""

    trading_fee: float = 0.012
    expected_slippage: float = 0.006
    safety_buffer: float = 0.010

    # Realized BTC 1-minute volatility, as a return (2.6 bps measured over
    # 300 minutes of OKX candles). Update via calibrate_volatility().
    sigma_per_minute: float = 0.00026

    # Weight of book/flow adjustments, in probability points per unit of
    # imbalance. Deliberately small: these signals are weak evidence.
    book_adjustment: float = 0.03
    flow_adjustment: float = 0.02

    def evaluate(self, state: MarketState) -> FairValueResult:
        opening = state.info.opening_price_btc
        best_ask_up = state.book_up.best_ask
        best_ask_down = state.book_down.best_ask

        # Without a true opening price the market cannot be priced at all.
        if not opening or opening <= 0 or state.btc_price <= 0:
            return FairValueResult(
                posterior_up=0.5,
                posterior_down=0.5,
                best_ask_up=best_ask_up,
                best_ask_down=best_ask_down,
                gross_edge_up=0.0,
                gross_edge_down=0.0,
                net_edge_up=-1.0,
                net_edge_down=-1.0,
                valid=False,
            )

        distance = (state.btc_price - opening) / opening
        seconds_left = state.seconds_remaining

        base = brownian_probability(distance, self.sigma_per_minute, seconds_left)

        # Weak secondary signals, scaled down as the outcome becomes certain
        # (near resolution, book noise should not move the price).
        uncertainty = 4.0 * base * (1.0 - base)  # peaks at 1.0 when base == 0.5
        imb = order_book_imbalance(state.book_up)
        flow = trade_flow_imbalance(state.recent_trades)
        adjustment = uncertainty * (
            imb * self.book_adjustment + flow * self.flow_adjustment
        )

        posterior_up = max(0.005, min(0.995, base + adjustment))
        posterior_down = 1.0 - posterior_up

        fill_up = best_ask_up if best_ask_up is not None else 1.0
        fill_down = best_ask_down if best_ask_down is not None else 1.0

        gross_up, net_up = calculate_executable_edge(
            posterior_up, fill_up,
            self.trading_fee, self.expected_slippage, self.safety_buffer,
        )
        gross_down, net_down = calculate_executable_edge(
            posterior_down, fill_down,
            self.trading_fee, self.expected_slippage, self.safety_buffer,
        )

        sigma_rem = self.sigma_per_minute * math.sqrt(max(seconds_left, 0) / 60.0)

        return FairValueResult(
            posterior_up=posterior_up,
            posterior_down=posterior_down,
            best_ask_up=best_ask_up,
            best_ask_down=best_ask_down,
            gross_edge_up=gross_up,
            gross_edge_down=gross_down,
            net_edge_up=net_up,
            net_edge_down=net_down,
            distance_bps=distance * 10000,
            sigma_remaining_bps=sigma_rem * 10000,
            z_score=(distance / sigma_rem) if sigma_rem > 0 else 0.0,
            base_probability=base,
            valid=True,
        )

    def calibrate_volatility(self, minute_closes: list[float]) -> float:
        """Set sigma_per_minute from a series of 1-minute closing prices."""
        if len(minute_closes) < 30:
            return self.sigma_per_minute
        rets = [
            (minute_closes[i] - minute_closes[i - 1]) / minute_closes[i - 1]
            for i in range(1, len(minute_closes))
            if minute_closes[i - 1] > 0
        ]
        if len(rets) < 20:
            return self.sigma_per_minute
        mean = sum(rets) / len(rets)
        variance = sum((r - mean) ** 2 for r in rets) / len(rets)
        self.sigma_per_minute = max(variance**0.5, 1e-6)
        return self.sigma_per_minute
