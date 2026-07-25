"""Today's sports matches from Polymarket, priced as 1X2 markets.

A football match on Polymarket is not one three-way market — it is three
separate binary markets (home / draw / away) that happen to be mutually
exclusive and collectively exhaustive. That structure is the same one the
BTC Up/Down detector already exploits, one outcome wider: buying all of
them guarantees exactly $1 back, so whenever they cost less than $1 in
total the difference is riskless.

Two prices per outcome matter and they are not the same number:

    last traded   what the site shows; what someone already paid
    best ask      what it would cost you right now

The overround is computed from asks, because paying the ask is what
actually happens. Using last-traded prices makes books look far more
arbitrageable than they are — Tijuana vs Leon showed 100.5c on last
trades and 102c on asks, and only the second one is real.

Events whose slug ends in `-more-markets` carry spreads and totals rather
than the match result, so they are skipped here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import httpx
import structlog

log = structlog.get_logger()

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Cost of crossing the spread on every leg of a multi-outcome basket.
# Wider than the two-leg BTC figure because a 1X2 basket needs three fills.
BASKET_COST = 0.025


@dataclass
class Outcome:
    """One side of a match: a team, or the draw."""

    label: str
    kind: str                      # "home" | "away" | "draw"
    last_price: float | None
    best_bid: float | None
    best_ask: float | None
    liquidity: float
    token_id_yes: str | None

    @property
    def implied(self) -> float | None:
        """Probability implied by the last traded price."""
        return self.last_price


@dataclass
class Match:
    """A single fixture with its result market."""

    event_id: str
    title: str
    slug: str
    sport: str
    league: str
    shape: str                     # "1x2" (separate markets) | "moneyline" (one market)
    start_time: datetime | None
    end_time: datetime | None
    outcomes: list[Outcome] = field(default_factory=list)
    volume_24h: float = 0.0
    liquidity: float = 0.0

    @property
    def is_three_way(self) -> bool:
        return any(o.kind == "draw" for o in self.outcomes)

    @property
    def overround(self) -> float | None:
        """Total cost to buy every outcome at the ask.

        Above 1.0 is the bookmaker's margin. Below 1.0 is free money.
        """
        asks = [o.best_ask for o in self.outcomes]
        if not asks or any(a is None for a in asks):
            return None
        return sum(asks)

    @property
    def arb_profit(self) -> float | None:
        """Profit per $1 basket after costs; None when there is none.

        Moneyline fixtures are excluded outright: both sides come from one
        market as complementary tokens, so their asks sum to $1 plus the
        spread by construction and can never be arbitraged.
        """
        if self.shape != "1x2":
            return None
        total = self.overround
        if total is None:
            return None
        profit = 1.0 - total - BASKET_COST
        return profit if profit > 0 else None

    def normalised(self) -> list[tuple[Outcome, float]]:
        """Outcomes with the margin stripped out, so probabilities sum to 1.

        Raw last-traded prices carry the overround, so a 40/30/40 book is
        not three probabilities — it is three prices summing to 110%.
        """
        priced = [(o, o.last_price) for o in self.outcomes if o.last_price is not None]
        total = sum(p for _, p in priced)
        if total <= 0:
            return [(o, 0.0) for o in self.outcomes]
        return [(o, p / total) for o, p in priced]

    @property
    def status(self) -> str:
        now = datetime.now(timezone.utc)
        if self.start_time and now < self.start_time:
            return "upcoming"
        if self.end_time and now > self.end_time:
            return "ended"
        return "live"

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"

    @property
    def day_bucket(self) -> str:
        """Which calendar day the fixture kicks off on, in UTC.

        Grouping by date rather than by hours-from-now keeps "tomorrow"
        meaning tomorrow: a match at 23:00 today and one at 01:00 tonight
        are two hours apart but belong in different buckets.
        """
        if self.status == "live":
            return "live"
        if not self.start_time:
            return "later"

        today = datetime.now(timezone.utc).date()
        delta = (self.start_time.date() - today).days
        if delta <= 0:
            return "today"
        if delta == 1:
            return "tomorrow"
        return "later"

    @property
    def minutes_to_start(self) -> float | None:
        if not self.start_time:
            return None
        return (self.start_time - datetime.now(timezone.utc)).total_seconds() / 60


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _loads(value, default):
    """Gamma returns some list fields as JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value if value is not None else default


def _team_names(event: dict) -> list[str]:
    """Home side first, away side second."""
    teams = _loads(event.get("teams"), [])
    named = [t for t in teams if isinstance(t, dict) and t.get("name")]
    if len(named) == 2:
        home = [t["name"] for t in named if t.get("ordering") == "home"]
        away = [t["name"] for t in named if t.get("ordering") == "away"]
        if home and away:
            return [home[0], away[0]]
        return [t["name"] for t in named]

    title = event.get("title", "")
    if " vs. " in title:
        # Strip the "Valorant: " prefix and " (BO3) - League" suffix.
        core = title.split(":", 1)[-1].split(" - ")[0].strip()
        parts = [s.strip() for s in core.split(" vs. ", 1)]
        if len(parts) == 2:
            return [parts[0], re.sub(r"\s*\(BO\d+\)\s*$", "", parts[1]).strip()]
    return []


def _outcomes_from_1x2(markets: list[dict], teams: list[str]) -> list[Outcome]:
    """Soccer shape: one Yes/No market per result, grouped by team name.

    Because the three markets are independent, their asks can sum to less
    than $1 — that is the arbitrage this module looks for.
    """
    found: list[Outcome] = []
    for m in markets:
        if m.get("closed") or not m.get("active"):
            continue
        if [str(o).lower() for o in _loads(m.get("outcomes"), [])] != ["yes", "no"]:
            continue

        group = (m.get("groupItemTitle") or "").strip()
        if group.lower().startswith("draw"):
            kind, label = "draw", "Empate"
        elif teams and group == teams[0]:
            kind, label = "home", group
        elif len(teams) > 1 and group == teams[1]:
            kind, label = "away", group
        else:
            continue  # a prop or side market, not the match result

        prices = _loads(m.get("outcomePrices"), [])
        tokens = _loads(m.get("clobTokenIds"), [])
        found.append(
            Outcome(
                label=label,
                kind=kind,
                last_price=_as_float(prices[0]) if prices else None,
                best_bid=_as_float(m.get("bestBid")),
                best_ask=_as_float(m.get("bestAsk")),
                liquidity=_as_float(m.get("liquidityNum") or m.get("liquidity")) or 0.0,
                token_id_yes=tokens[0] if tokens else None,
            )
        )
    return found


def _outcomes_from_moneyline(markets: list[dict], teams: list[str]) -> list[Outcome]:
    """Esports shape: a single `Match Winner` market holding both sides.

    The two tokens are complements of one another, so the second side's
    ask is `1 - bid` of the first. Their sum is therefore always $1 plus
    the spread — no arbitrage is possible within one market, only across
    separate ones.
    """
    for m in markets:
        if m.get("closed") or not m.get("active"):
            continue
        if (m.get("groupItemTitle") or "").strip().lower() != "match winner":
            continue

        names = [str(o) for o in _loads(m.get("outcomes"), [])]
        if len(names) != 2:
            continue

        prices = _loads(m.get("outcomePrices"), [])
        tokens = _loads(m.get("clobTokenIds"), [])
        bid, ask = _as_float(m.get("bestBid")), _as_float(m.get("bestAsk"))
        liq = _as_float(m.get("liquidityNum") or m.get("liquidity")) or 0.0

        return [
            Outcome(
                label=names[0], kind="home",
                last_price=_as_float(prices[0]) if prices else None,
                best_bid=bid, best_ask=ask, liquidity=liq,
                token_id_yes=tokens[0] if tokens else None,
            ),
            Outcome(
                label=names[1], kind="away",
                last_price=_as_float(prices[1]) if len(prices) > 1 else None,
                best_bid=(1 - ask) if ask is not None else None,
                best_ask=(1 - bid) if bid is not None else None,
                liquidity=liq,
                token_id_yes=tokens[1] if len(tokens) > 1 else None,
            ),
        ]
    return []


def _build_match(event: dict) -> Match | None:
    markets = event.get("markets") or []
    teams = _team_names(event)

    # Without two identifiable sides this is a prop, not a fixture.
    if len(teams) != 2:
        return None

    outcomes = _outcomes_from_1x2(markets, teams)
    shape = "1x2"
    if len(outcomes) < 2:
        outcomes = _outcomes_from_moneyline(markets, teams)
        shape = "moneyline"
    if len(outcomes) < 2:
        return None

    order = {"home": 0, "draw": 1, "away": 2}
    outcomes.sort(key=lambda o: order.get(o.kind, 9))

    tags = [t.get("label", "") for t in (event.get("tags") or [])]
    league = next(
        (
            t for t in tags
            if t not in ("Sports", "Games", "Recurring", "Hide From New", "Esports")
        ),
        event.get("seriesSlug") or "",
    )

    sport = event.get("sport")
    sport_name = sport.get("sport") if isinstance(sport, dict) else (sport or "")

    return Match(
        event_id=str(event.get("id")),
        title=event.get("title", ""),
        slug=event.get("slug", ""),
        sport=sport_name or ("soccer" if "Soccer" in tags else "sports"),
        league=league,
        shape=shape,
        start_time=_parse_dt(event.get("startTime") or event.get("eventDate")),
        end_time=_parse_dt(event.get("endDate")),
        outcomes=outcomes,
        volume_24h=_as_float(event.get("volume24hr")) or 0.0,
        liquidity=_as_float(event.get("liquidity")) or 0.0,
    )


async def fetch_todays_matches(
    client: httpx.AsyncClient,
    hours_ahead: int = 52,
    min_liquidity: float = 1000.0,
    limit: int = 500,
) -> list[Match]:
    """Sports fixtures resolving within `hours_ahead`, richest book first."""
    horizon = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)

    resp = await client.get(
        GAMMA_EVENTS,
        params={
            "closed": "false",
            "active": "true",
            "limit": limit,
            "end_date_max": horizon.isoformat(),
            "order": "volume24hr",
            "ascending": "false",
        },
        headers=UA_HEADERS,
    )
    resp.raise_for_status()

    matches: list[Match] = []
    for event in resp.json():
        # Spreads and totals live on their own event; skip them.
        if event.get("slug", "").endswith("-more-markets"):
            continue

        tags = {t.get("label", "") for t in (event.get("tags") or [])}
        if "Sports" not in tags and "Esports" not in tags:
            continue

        match = _build_match(event)
        if not match or match.liquidity < min_liquidity:
            continue
        if match.status == "ended":
            continue
        matches.append(match)

    # Kick-off order: what is starting soonest is what matters.
    matches.sort(
        key=lambda m: (
            m.start_time or datetime.max.replace(tzinfo=timezone.utc),
            -m.volume_24h,
        )
    )
    return matches


def to_rows(matches: list[Match]) -> list[dict]:
    """Flatten matches for the dashboard."""
    rows = []
    for m in matches:
        normalised = {id(o): p for o, p in m.normalised()}
        rows.append(
            {
                "id": m.event_id,
                "title": m.title,
                "slug": m.slug,
                "url": m.url,
                "league": m.league,
                "sport": m.sport,
                "shape": m.shape,
                "status": m.status,
                "day_bucket": m.day_bucket,
                "minutes_to_start": (
                    round(m.minutes_to_start) if m.minutes_to_start is not None else None
                ),
                "start_time": m.start_time.isoformat() if m.start_time else None,
                "volume_24h": round(m.volume_24h),
                "liquidity": round(m.liquidity),
                "three_way": m.is_three_way,
                "overround": round(m.overround, 4) if m.overround else None,
                "arb_profit": round(m.arb_profit, 4) if m.arb_profit else None,
                "outcomes": [
                    {
                        "label": o.label,
                        "kind": o.kind,
                        "last": o.last_price,
                        "bid": o.best_bid,
                        "ask": o.best_ask,
                        "prob": round(normalised.get(id(o), 0.0), 4),
                        "liquidity": round(o.liquidity),
                    }
                    for o in m.outcomes
                ],
            }
        )
    return rows
