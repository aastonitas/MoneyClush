"""Tests for what the trading desk recommends, and what it refuses to.

The failure this guards against is the one that makes signal bots
worthless: ranking by the size of the number instead of by whether the
number means anything. A 56c model-versus-market gap looks enormous next
to a 2c arbitrage, and a desk that sorts on edge alone puts the useless
one on top. The BTC backtest already measured that gap over 299 windows
and found the model behind the market, so surfacing it as a buy is not a
cosmetic mistake — it is rebuilding the thing that was disproved.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush.signals.trading_desk import (  # noqa: E402
    Confidence,
    build_desk,
    from_model_edges,
)

MARKETS = [
    {"asset": "BTC", "duration": "5m", "slug": "btc-5m-1", "seconds_remaining": 67},
    {"asset": "BTC", "duration": "15m", "slug": "btc-15m-1", "seconds_remaining": 600},
]

# The model is wildly more confident than the market here. This is a real
# shape seen live: fair 81c against a 22c ask.
BIG_MODEL_EDGE = [{
    "market": "BTC 5m", "slug": "btc-5m-1", "fair": 0.812, "ask": 0.22,
    "edge": 0.5636, "side": "UP", "signal": "EDGE", "z": 0.93,
}]

SMALL_ARB = [{
    "market": "BTC 5m", "pair_cost": 0.96, "profit_per_pair": 0.022, "size": 25,
}]


def test_riskless_arb_outranks_a_much_larger_model_edge():
    """2c of arithmetic beats 56c of disagreement, and must sort that way."""
    desk = build_desk(SMALL_ARB, MARKETS, [], BIG_MODEL_EDGE)

    assert desk.signals[0].confidence is Confidence.ALTA
    assert desk.signals[0].edge == 0.022
    assert desk.signals[-1].confidence is Confidence.NINGUNA
    assert desk.signals[-1].edge == 0.5636


def test_model_edges_are_never_tradeable_or_staked():
    """The disproved signal gets no money and no green light, ever."""
    signals = from_model_edges(BIG_MODEL_EDGE, MARKETS)

    assert len(signals) == 1
    assert not signals[0].tradeable
    assert signals[0].stake == 0.0
    assert "NO OPERAR" in signals[0].action
    assert "0.202" in signals[0].evidence      # cites the measurement


def test_statistical_edges_are_capped_at_a_dollar():
    """The favourite bias is a sample being collected, not a position."""
    opportunities = [{
        "strategy": "favourite_fade", "venue": "polymarket", "category": "deportes",
        "kind": "statistical", "label": "X vs Y", "edge": 0.03,
        "rationale": "r", "evidence": "e", "url": "",
    }]
    desk = build_desk([], MARKETS, opportunities, [])

    assert desk.signals[0].confidence is Confidence.BAJA
    assert desk.signals[0].stake == 1.0


def test_structural_category_edges_keep_their_tier():
    opportunities = [{
        "strategy": "kalshi_ladder_arb", "venue": "kalshi", "category": "crypto",
        "kind": "structural", "label": "63k/64k", "edge": 0.012,
        "rationale": "r", "evidence": "e", "url": "",
    }]
    desk = build_desk([], MARKETS, opportunities, [])

    assert desk.signals[0].confidence is Confidence.ALTA
    assert desk.actionable == 1


def test_a_model_only_scan_reports_no_trade():
    """Signals present but none worth taking is still "do not trade"."""
    desk = build_desk([], MARKETS, [], BIG_MODEL_EDGE)

    assert desk.actionable == 0
    assert "modelo" in desk.no_trade_reason.lower()


def test_empty_scan_distinguishes_quiet_from_blind():
    """No data and no opportunity are different states and must read so."""
    blind = build_desk([], [], [], [])
    assert "no se ha podido mirar" in blind.no_trade_reason

    quiet = build_desk([], MARKETS, [], [])
    assert "eficiente" in quiet.no_trade_reason


def test_expiring_signal_sorts_ahead_of_a_fatter_one():
    """A signal you cannot reach in time is worth nothing, so urgency wins."""
    roomy = [{
        "strategy": "kalshi_ladder_arb", "venue": "kalshi", "category": "crypto",
        "kind": "structural", "label": "fat", "edge": 0.05,
        "rationale": "r", "evidence": "e", "url": "",
    }]
    desk = build_desk(SMALL_ARB, MARKETS, roomy, [])

    # The 2.2c arb expires in 67s; the 5c ladder has no clock on it.
    assert desk.signals[0].source == "arbitraje_btc"
    assert desk.signals[1].edge == 0.05


def test_filters_explain_themselves_when_they_hide_everything():
    desk = build_desk(SMALL_ARB, MARKETS, [], [], timeframe="15m")

    assert desk.signals == []
    assert "filtro" in desk.no_trade_reason


def test_expired_signals_are_dropped_before_ranking():
    dead = [{"asset": "BTC", "duration": "5m", "slug": "btc-5m-1",
             "seconds_remaining": -5}]
    desk = build_desk([], dead, [], [{
        "market": "BTC 5m", "slug": "btc-5m-1", "fair": 0.8, "ask": 0.2,
        "edge": 0.6, "side": "UP", "signal": "EDGE", "z": 0.9,
    }])

    assert desk.signals == []
    assert desk.scanned == 0
