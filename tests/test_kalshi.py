"""Tests for the Kalshi ladder scan and its fee arithmetic.

The failure mode these guard against is not a crash — it is the panel
reporting an arbitrage that is not one. Two ways that happens: counting
fees wrong so a 2c credit looks tradeable when Kalshi takes 4c of it, and
treating an unquoted strike (0 bid / 100 ask) as a real market at a 50c
midpoint. Both produce confident numbers with nothing behind them, which
is exactly the class of bug that made the first BTC backtest look
profitable.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush.data.kalshi import (  # noqa: E402
    KalshiMarket,
    LadderGroup,
    ladder_arbitrages,
    round_trip_cost,
    taker_fee,
)


def _market(label, strike, bid, ask, ticker=None):
    return KalshiMarket(
        ticker=ticker or f"KXBTCD-26JUL3017-T{strike}",
        event_ticker="KXBTCD-26JUL3017",
        series="KXBTCD",
        title="Bitcoin price",
        subtitle=label,
        yes_bid=bid,
        yes_ask=ask,
        volume=100.0,
        open_interest=100.0,
        liquidity=100.0,
        close_time=datetime.now(timezone.utc) + timedelta(minutes=30),
        status="active",
        strike=strike,
    )


def _group(markets):
    return LadderGroup(
        event_ticker="KXBTCD-26JUL3017",
        series="KXBTCD",
        title="Bitcoin price",
        close_time=datetime.now(timezone.utc) + timedelta(minutes=30),
        markets=markets,
    )


# ------------------------------------------------------------------ fees

def test_fee_peaks_at_the_money():
    """Kalshi's fee is largest at 50c, which is where trading happens."""
    assert taker_fee(0.50) >= taker_fee(0.10)
    assert taker_fee(0.50) >= taker_fee(0.90)


def test_fee_rounds_up_to_the_cent():
    """0.07 x 0.5 x 0.5 = 1.75c, and Kalshi does not round that down."""
    assert taker_fee(0.50) == 0.02


def test_fee_is_zero_outside_a_valid_price():
    assert taker_fee(0.0) == 0.0
    assert taker_fee(1.0) == 0.0


def test_round_trip_charges_both_legs():
    assert round_trip_cost(0.50) == taker_fee(0.50) + taker_fee(0.50)


# --------------------------------------------------------------- laddering

def test_consistent_ladder_yields_nothing():
    """A ladder in the right order is not an opportunity.

    Lower strike quoted higher than the upper strike is the normal,
    arbitrage-free state, and the scan must report zero rather than
    inventing something from the spread.
    """
    group = _group([
        _market("$63,000 or above", 63000.0, 0.80, 0.83),
        _market("$64,000 or above", 64000.0, 0.45, 0.50),
    ])
    assert ladder_arbitrages(group) == []


def test_monotonicity_violation_is_found():
    """Buying the low strike cheaper than the high strike sells is free money.

    Here the $63k rung can be bought at 40c while the $64k rung — which
    cannot be more likely — is bid at 55c. Long the low, short the high:
    the payoff is 1 inside the band and 0 outside, never negative, and it
    comes with 15c of credit up front.
    """
    group = _group([
        _market("$63,000 or above", 63000.0, 0.35, 0.40),
        _market("$64,000 or above", 64000.0, 0.55, 0.60),
    ])

    found = ladder_arbitrages(group)
    assert len(found) == 1

    arb = found[0]
    assert arb.low.strike == 63000.0
    assert arb.high.strike == 64000.0
    assert abs(arb.credit - 0.15) < 1e-9
    # Both legs are charged at their own price, not at the net.
    assert abs(arb.fees - (taker_fee(0.40) + taker_fee(0.55))) < 1e-9
    assert abs(arb.net - (arb.credit - arb.fees)) < 1e-9


def test_fees_can_erase_a_thin_violation():
    """A 1c dislocation is not tradeable once Kalshi takes its cut.

    This is the whole reason the fee model exists: the same 1c gap on
    Polymarket would be profit, and here it is not.
    """
    group = _group([
        _market("$63,000 or above", 63000.0, 0.45, 0.50),
        _market("$64,000 or above", 64000.0, 0.51, 0.56),
    ])
    assert ladder_arbitrages(group) == []


def test_unquoted_strikes_are_not_markets():
    """A 0 bid / 100 ask rung has no price, and no midpoint worth using.

    Kalshi lists every strike whether or not anyone makes a market on it.
    Reading those as 50c markets would put a fabricated number into the
    distribution and could pair with a real rung to fake an arbitrage.
    """
    empty = _market("$70,000 or above", 70000.0, 0.0, 1.0)
    assert not empty.tradeable

    group = _group([_market("$63,000 or above", 63000.0, 0.80, 0.83), empty])
    assert ladder_arbitrages(group) == []
    assert [m for m, _ in group.implied_distribution()] == [group.markets[0]]


def test_crossed_quote_is_rejected():
    """A book whose ask sits below its bid is stale, not an opportunity."""
    assert not _market("$63,000 or above", 63000.0, 0.60, 0.55).tradeable


def test_distribution_differences_the_ladder():
    """Consecutive digitals differ by the probability of the band between.

    This is what a ladder gives you that a lone Up/Down binary cannot: the
    market's whole view of where the asset lands, not just one threshold.
    """
    group = _group([
        _market("$63,000 or above", 63000.0, 0.78, 0.82),   # mid 0.80
        _market("$64,000 or above", 64000.0, 0.48, 0.52),   # mid 0.50
        _market("$65,000 or above", 65000.0, 0.18, 0.22),   # mid 0.20
    ])

    bands = dict(
        (m.strike, round(p, 4)) for m, p in group.implied_distribution()
    )
    assert bands[63000.0] == 0.30      # between 63k and 64k
    assert bands[64000.0] == 0.30      # between 64k and 65k
    assert bands[65000.0] == 0.20      # above 65k, the open tail
