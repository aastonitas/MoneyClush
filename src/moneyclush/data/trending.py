"""Polymarket's highest-volume markets, across every category.

Unlike the sports module this does not assume a 1X2 or moneyline shape —
a trending event can be a two-way prop, a 128-candidate election market,
or anything between. So each event is reduced to its top outcomes by
probability rather than modelled as a fixed structure.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Tags too generic to describe what an event is actually about.
NOISE_TAGS = {"Recurring", "Hide From New", "Games", "All"}

# Polymarket runs a genuine sideshow: how many times Elon Musk will tweet
# this week, which words a CEO will utter on an earnings call, what MrBeast
# will say in his next video. These tags are how the API labels them.
QUIRKY_TAGS = {
    "Tweet Markets",
    "Mentions",
    "Earnings Calls",
    "Pop Culture",
    "Celebrities",
    "Memes",
}


@dataclass
class TrendingOutcome:
    label: str
    price: float


@dataclass
class TrendingEvent:
    event_id: str
    title: str
    slug: str
    category: str
    volume_24h: float
    liquidity: float
    end_time: datetime | None
    quirky: bool = False
    outcomes: list[TrendingOutcome] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.slug}"

    @property
    def days_left(self) -> float | None:
        if not self.end_time:
            return None
        return (self.end_time - datetime.now(timezone.utc)).total_seconds() / 86400

    @property
    def horizon(self) -> str:
        """Soon means resolvable within a week; the rest is a long wait."""
        days = self.days_left
        if days is None:
            return "far"
        return "soon" if days <= 7 else "far"


def _loads(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value if value is not None else default


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _top_outcomes(markets: list[dict], limit: int = 4) -> list[TrendingOutcome]:
    """The highest-priced side of each sub-market, i.e. the current favourites."""
    candidates: list[TrendingOutcome] = []
    for m in markets:
        if m.get("closed") or not m.get("active"):
            continue
        label = m.get("groupItemTitle") or m.get("question") or ""
        prices = _loads(m.get("outcomePrices"), [])
        price = _as_float(prices[0]) if prices else None
        if label and price is not None:
            candidates.append(TrendingOutcome(label=label.strip(), price=price))

    candidates.sort(key=lambda o: o.price, reverse=True)
    return candidates[:limit]


def _build_event(event: dict) -> TrendingEvent | None:
    markets = event.get("markets") or []
    outcomes = _top_outcomes(markets)
    if not outcomes:
        return None

    tags = [t.get("label", "") for t in (event.get("tags") or [])]
    quirky = bool(set(tags) & QUIRKY_TAGS)
    # The quirky label is more descriptive than the generic one it sits next
    # to, so prefer it when naming the category.
    category = next(
        (t for t in tags if t in QUIRKY_TAGS),
        next((t for t in tags if t not in NOISE_TAGS), "General"),
    )

    return TrendingEvent(
        event_id=str(event.get("id")),
        title=event.get("title", ""),
        slug=event.get("slug", ""),
        category=category,
        volume_24h=_as_float(event.get("volume24hr")) or 0.0,
        liquidity=_as_float(event.get("liquidity")) or 0.0,
        end_time=_parse_dt(event.get("endDate")),
        quirky=quirky,
        outcomes=outcomes,
    )


async def _fetch_page(client: httpx.AsyncClient, offset: int, size: int) -> list[dict]:
    resp = await client.get(
        GAMMA_EVENTS,
        params={
            "closed": "false",
            "active": "true",
            "limit": size,
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false",
        },
        headers=UA_HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


async def fetch_trending(
    client: httpx.AsyncClient,
    limit: int = 24,
    min_volume: float = 50_000.0,
    quirky_min_volume: float = 2_000.0,
    pages: int = 4,
    page_size: int = 100,
) -> list[TrendingEvent]:
    """Top events by 24h volume, plus the oddities further down the list.

    The quirky markets are genuinely small — a market on what MrBeast will
    say next trades a fraction of what a Fed decision does — so scanning
    only the richest page would never surface them. Several pages are
    pulled and the curiosities admitted on a much lower volume bar.
    """
    pages_data = await asyncio.gather(
        *(_fetch_page(client, i * page_size, page_size) for i in range(pages)),
        return_exceptions=True,
    )

    seen_ids: set[str] = set()
    mainstream: list[TrendingEvent] = []
    quirky: list[TrendingEvent] = []

    for page in pages_data:
        if isinstance(page, BaseException):
            continue
        for raw in page:
            if raw.get("slug", "").endswith("-more-markets"):
                continue
            event_id = str(raw.get("id"))
            if event_id in seen_ids:
                continue

            built = _build_event(raw)
            if not built:
                continue
            seen_ids.add(event_id)

            if built.quirky and built.volume_24h >= quirky_min_volume:
                quirky.append(built)
            elif built.volume_24h >= min_volume:
                mainstream.append(built)

    mainstream.sort(key=lambda e: e.volume_24h, reverse=True)
    quirky.sort(key=lambda e: e.volume_24h, reverse=True)
    return mainstream[:limit] + quirky[:limit]


def to_rows(events: list[TrendingEvent]) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [
        {
            "id": e.event_id,
            "title": e.title,
            "slug": e.slug,
            "url": e.url,
            "category": e.category,
            "quirky": e.quirky,
            "horizon": e.horizon,
            "volume_24h": round(e.volume_24h),
            "liquidity": round(e.liquidity),
            "ends_in_days": (
                round((e.end_time - now).total_seconds() / 86400, 1)
                if e.end_time and e.end_time > now
                else None
            ),
            "outcomes": [
                {"label": o.label, "prob": round(o.price, 4)} for o in e.outcomes
            ],
        }
        for e in events
    ]
