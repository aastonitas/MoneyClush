"""Paper positions on trending events, with the option to sell early.

Every other track in this project buys once and waits for the result,
because a BTC 5-minute window resolves in five minutes and a football
match resolves the same evening. Trending markets do not work like that:
an election or a Fed decision can sit open for months, so "hold to
resolution" is not a strategy anybody would actually run — the capital is
locked the whole time and the only number that ever moves is the price.

So this track marks each position to market and lets it be closed at
whatever the outcome trades at right now. That turns a single question
("was the favourite right?") into the two that matter here:

    hold      does a 75c favourite really win 75% of the time?
    sell      does its *price* drift up after entry, win or lose?

Those can disagree. A favourite that drifts 78c -> 85c and then loses
pays a profit to whoever sold and a full loss to whoever held, and only
by recording the exit price separately can the two be told apart.

Entries are restricted to strong favourites because that is the band the
backtest flagged as overpriced — this is the live test of that finding on
a different asset class.

The arithmetic throughout: a `stake` at `entry` buys `stake / entry`
shares, each worth $1 if the outcome happens and nothing if it does not.
Marking at price `p` values those shares at `p` each, so resolution is
just the same formula with `p` pinned to 1 or 0.
"""

from __future__ import annotations

# The band the backtest found favourites to be overpriced in. Below this a
# position is not testing the hypothesis, it is just a coin flip.
MIN_ENTRY_PROB = 0.75

# A side trading this high has effectively resolved; buying it is paying
# ~$1 for $1 and would pad the hit rate without risking anything.
MAX_ENTRY_PROB = 0.97


def shares(stake: float, entry_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return stake / entry_price


def position_pnl(entry_price: float, stake: float, mark_price: float) -> float:
    """Profit if the position were closed at `mark_price`.

    `mark_price` is 1.0 for an outcome that happened and 0.0 for one that
    did not, so settlement needs no separate branch.
    """
    return shares(stake, entry_price) * mark_price - stake


def entry_rejection(price: float | None) -> str | None:
    """Why this price cannot be entered, or None when it can."""
    if price is None:
        return "sin precio"
    if price < MIN_ENTRY_PROB:
        return (
            f"cotiza a {price * 100:.0f}¢ — este simulador solo entra a "
            f"favoritos de {MIN_ENTRY_PROB * 100:.0f}¢ o más"
        )
    if price > MAX_ENTRY_PROB:
        return (
            f"cotiza a {price * 100:.0f}¢ — prácticamente resuelto, no queda "
            "nada que apostar"
        )
    return None


def summarise(closed: list[dict], open_positions: list[dict]) -> dict:
    """Realised performance, split by how each position was closed.

    Held and sold positions are counted separately on purpose: mixing them
    answers neither of the two questions this track exists to ask.
    """
    held = [b for b in closed if b.get("exit_reason") == "resolved"]
    sold = [b for b in closed if b.get("exit_reason") == "manual"]

    def block(bets: list[dict]) -> dict:
        pnl = sum(b.get("pnl") or 0.0 for b in bets)
        staked = sum(b.get("stake") or 0.0 for b in bets)
        wins = sum(1 for b in bets if (b.get("pnl") or 0.0) > 0)
        return {
            "n": len(bets),
            "wins": wins,
            "hit_rate": round(wins / len(bets), 4) if bets else None,
            "pnl": round(pnl, 4),
            "roi": round(pnl / staked, 4) if staked > 0 else None,
        }

    unrealised = sum(
        position_pnl(
            b["entry_price"],
            b.get("stake") or 1.0,
            b.get("last_price") if b.get("last_price") is not None else b["entry_price"],
        )
        for b in open_positions
    )

    # What the entry prices claimed would happen, against what did. The gap
    # is the same favourite-bias measurement the rest of the project runs,
    # restricted to positions actually carried to resolution.
    expected = (
        round(sum(b["entry_price"] for b in held) / len(held), 4) if held else None
    )

    return {
        "held": block(held),
        "sold": block(sold),
        "expected_hit_rate": expected,
        "open": len(open_positions),
        "open_stake": round(sum(b.get("stake") or 0.0 for b in open_positions), 2),
        "unrealised": round(unrealised, 4),
        "realised": round(sum(b.get("pnl") or 0.0 for b in closed), 4),
        "min_entry": MIN_ENTRY_PROB,
    }


def find_settlement(event: dict, outcome: str) -> float | None:
    """Final price of `outcome` once its market has settled, else None.

    A trending event carries many sub-markets and only the one that was
    backed matters. Price alone is not enough to call it: an outcome can
    trade at 99c because the market is confident, which is a forecast, not
    a result — so the market has to be flagged closed or resolved first.
    """
    wanted = outcome.strip().lower()
    for market in event.get("markets") or []:
        label = (market.get("groupItemTitle") or market.get("question") or "").strip()
        if label.lower() != wanted:
            continue

        uma = (market.get("umaResolutionStatus") or "").strip().lower()
        if not market.get("closed") and uma not in {"proposed", "resolved", "settled"}:
            return None

        raw = market.get("outcomePrices")
        if isinstance(raw, str):
            import json

            try:
                raw = json.loads(raw)
            except ValueError:
                return None
        if not raw:
            return None
        try:
            return float(raw[0])
        except (TypeError, ValueError):
            return None
    return None
