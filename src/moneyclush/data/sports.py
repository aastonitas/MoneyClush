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

import asyncio
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

# Gamma labels each fixture with a league code (`mex`, `atp`, `cs2`). Rolling
# them up into disciplines is what makes a "solo tenis" filter possible.
DISCIPLINE_BY_TAG = {
    "soccer": "futbol",
    "tennis": "tenis",
    "baseball": "beisbol",
    "basketball": "basket",
    "cricket": "cricket",
    "ufc": "mma",
    "american football": "futbol-americano",
}
ESPORT_CODES = {"cs2", "lol", "val", "dota2", "mlbb", "hok", "rl", "ow2"}
DISCIPLINE_BY_CODE = {
    "mlb": "beisbol", "kbo": "beisbol", "npb": "beisbol",
    "atp": "tenis", "wta": "tenis", "itf": "tenis", "atp-doubles": "tenis",
    "ufc": "mma",
    "wnba": "basket", "nba": "basket",
    "cfl": "futbol-americano", "nfl": "futbol-americano",
    "nhl": "hockey",
}

DISCIPLINE_LABELS = {
    "futbol": "FÚTBOL",
    "esports": "ESPORTS",
    "tenis": "TENIS",
    "beisbol": "BÉISBOL",
    "mma": "UFC/MMA",
    "basket": "BASKET",
    "cricket": "CRICKET",
    "hockey": "HOCKEY",
    "futbol-americano": "F. AMERICANO",
    "otros": "OTROS",
}


def classify_discipline(sport_code: str, tags: list[str]) -> str:
    """Roll a league code and its tags up into a broad discipline."""
    code = (sport_code or "").lower()
    if code in ESPORT_CODES:
        return "esports"
    if code in DISCIPLINE_BY_CODE:
        return DISCIPLINE_BY_CODE[code]

    lowered = {t.lower() for t in tags}
    if "esports" in lowered:
        return "esports"
    if "ufc" in lowered or "mma" in lowered or "boxing" in lowered:
        return "mma"
    for tag, discipline in DISCIPLINE_BY_TAG.items():
        if tag in lowered:
            return discipline
    return "otros"


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
    discipline: str
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

    def fair_probabilities(self) -> list[tuple[Outcome, float]]:
        """De-vigged probabilities from mid prices, not from last trades.

        Mid is what the book says right now. Last-traded is what somebody
        paid at some point in the past, and on a quiet fixture that can be
        hours stale — stale enough to name the wrong favourite. Since the
        favourite-backing ledger picks a side from this number, using the
        last trade means occasionally betting against the current market.

        Mid also sits half a spread below the ask, which is what makes it
        the right baseline for calibration: comparing a realised hit rate
        against the ask charges the spread to the market's forecast.

        Falls back to last-traded normalisation when any outcome is
        missing a quote, rather than mixing the two inside one book.
        """
        mids: list[tuple[Outcome, float]] = []
        for o in self.outcomes:
            if o.best_bid is None or o.best_ask is None:
                return self.normalised()
            mids.append((o, (o.best_bid + o.best_ask) / 2.0))
        total = sum(m for _, m in mids)
        if total <= 0:
            return self.normalised()
        return [(o, m / total) for o, m in mids]

    @property
    def decided(self) -> bool:
        """Whether the market has already settled the result in practice.

        The declared window is not a reliable end signal: Polymarket gives
        esports fixtures a six-hour `endDate`, so a BO3 that finished two
        hours ago still looks live by the clock. The price does not lie —
        once a side trades at 97c+ the outcome is no longer in question.
        """
        top = max((p for _, p in self.fair_probabilities()), default=0.0)
        return top >= 0.97

    @property
    def status(self) -> str:
        now = datetime.now(timezone.utc)
        if self.start_time and now < self.start_time:
            return "upcoming"
        if self.end_time and now > self.end_time:
            return "ended"
        # Started, clock still running, but the market has called it.
        if self.decided:
            return "decided"
        return "live"

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"

    def favourite(self) -> tuple[Outcome, float] | None:
        """The outcome the market rates most likely, with its de-vigged
        probability. This is what the favourite-backing simulation bets on."""
        ranked = self.fair_probabilities()
        if not ranked:
            return None
        return max(ranked, key=lambda pair: pair[1])

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
    """Single market holding both sides as complementary tokens.

    The title of that market is not consistent across sports — esports
    call it `Match Winner`, a UFC card names the fight itself, and MLB
    leaves it null. What *is* consistent is that its two outcomes are the
    two competitors, so that is what identifies it.

    The tokens are complements, so the second side's ask is `1 - bid` of
    the first. Their sum is always $1 plus the spread — no arbitrage is
    possible within one market, only across separate ones.
    """
    wanted = {t.strip().lower() for t in teams}

    for m in markets:
        if m.get("closed") or not m.get("active"):
            continue

        names = [str(o) for o in _loads(m.get("outcomes"), [])]
        if len(names) != 2:
            continue
        if {n.strip().lower() for n in names} != wanted:
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
    # "Soccer" and "baseball" name the sport, not the competition. Skipping
    # them is what lets "Chinese Super League" or "OFB Cup" show instead.
    generic = {
        "Sports", "Games", "Recurring", "Hide From New", "Esports",
        "Soccer", "baseball", "basketball", "Tennis", "Cricket",
        "american football", "hockey",
    }
    league = next(
        (t for t in tags if t not in generic),
        next((t for t in tags if t not in {"Sports", "Games", "Recurring",
                                           "Hide From New"}), ""),
    )

    sport = event.get("sport")
    sport_name = sport.get("sport") if isinstance(sport, dict) else (sport or "")

    return Match(
        event_id=str(event.get("id")),
        title=event.get("title", ""),
        slug=event.get("slug", ""),
        sport=sport_name or ("soccer" if "Soccer" in tags else "sports"),
        league=league,
        discipline=classify_discipline(sport_name, tags),
        shape=shape,
        start_time=_parse_dt(event.get("startTime") or event.get("eventDate")),
        end_time=_parse_dt(event.get("endDate")),
        outcomes=outcomes,
        volume_24h=_as_float(event.get("volume24hr")) or 0.0,
        liquidity=_as_float(event.get("liquidity")) or 0.0,
    )


async def _fetch_page(
    client: httpx.AsyncClient, offset: int, size: int, horizon: str
) -> list[dict]:
    resp = await client.get(
        GAMMA_EVENTS,
        params={
            "closed": "false",
            "active": "true",
            "limit": size,
            "offset": offset,
            "end_date_max": horizon,
            "order": "volume24hr",
            "ascending": "false",
        },
        headers=UA_HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_todays_matches(
    client: httpx.AsyncClient,
    hours_ahead: int = 52,
    min_liquidity: float = 1000.0,
    pages: int = 8,
    page_size: int = 100,
) -> list[Match]:
    """Sports fixtures resolving within `hours_ahead`, richest book first.

    Results are ordered by volume across the whole of Polymarket, so a
    single page is mostly politics and crypto with a handful of fixtures
    at the top. Reading several pages is what surfaces the leagues that
    trade quietly — Guatemalan football, Korean baseball, ITF tennis.
    """
    horizon = (datetime.now(timezone.utc) + timedelta(hours=hours_ahead)).isoformat()

    pages_data = await asyncio.gather(
        *(_fetch_page(client, i * page_size, page_size, horizon) for i in range(pages)),
        return_exceptions=True,
    )

    matches: list[Match] = []
    seen: set[str] = set()
    for page in pages_data:
        if isinstance(page, BaseException):
            continue
        for event in page:
            # Spreads and totals live on their own event; skip them.
            if event.get("slug", "").endswith("-more-markets"):
                continue

            event_id = str(event.get("id"))
            if event_id in seen:
                continue

            tags = {t.get("label", "") for t in (event.get("tags") or [])}
            if "Sports" not in tags and "Esports" not in tags:
                continue

            match = _build_match(event)
            if not match or match.liquidity < min_liquidity:
                continue
            if match.status == "ended":
                continue
            seen.add(event_id)
            matches.append(match)

    # Anything the market has already called goes to the back regardless of
    # kick-off time; among the rest, soonest first.
    matches.sort(
        key=lambda m: (
            m.decided,
            m.start_time or datetime.max.replace(tzinfo=timezone.utc),
            -m.volume_24h,
        )
    )
    return matches


def to_rows(matches: list[Match]) -> list[dict]:
    """Flatten matches for the dashboard."""
    rows = []
    for m in matches:
        # The probability on screen is the same one the ledger bets on.
        normalised = {id(o): p for o, p in m.fair_probabilities()}
        rows.append(
            {
                "id": m.event_id,
                "title": m.title,
                "slug": m.slug,
                "url": m.url,
                "league": m.league,
                "sport": m.sport,
                "discipline": m.discipline,
                "discipline_label": DISCIPLINE_LABELS.get(m.discipline, "OTROS"),
                "shape": m.shape,
                "status": m.status,
                "decided": m.decided,
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
