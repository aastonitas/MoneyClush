"""Polymarket's highest-volume markets, across every category.

Unlike the sports module this does not assume a 1X2 or moneyline shape —
a trending event can be a two-way prop, a 128-candidate election market,
or anything between. So each event is reduced to its top outcomes by
probability rather than modelled as a fixed structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Tags too generic to describe what an event is actually about.
NOISE_TAGS = {"Recurring", "Hide From New", "Games", "All"}


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
    outcomes: list[TrendingOutcome] = field(default_factory=list)


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
    category = next((t for t in tags if t not in NOISE_TAGS), "General")

    return TrendingEvent(
        event_id=str(event.get("id")),
        title=event.get("title", ""),
        slug=event.get("slug", ""),
        category=category,
        volume_24h=_as_float(event.get("volume24hr")) or 0.0,
        liquidity=_as_float(event.get("liquidity")) or 0.0,
        end_time=_parse_dt(event.get("endDate")),
        outcomes=outcomes,
    )


async def fetch_trending(
    client: httpx.AsyncClient,
    limit: int = 15,
    min_volume: float = 50_000.0,
    fetch_n: int = 60,
) -> list[TrendingEvent]:
    """Top events by 24h volume, richest first, regardless of category."""
    resp = await client.get(
        GAMMA_EVENTS,
        params={
            "closed": "false",
            "active": "true",
            "limit": fetch_n,
            "order": "volume24hr",
            "ascending": "false",
        },
        headers=UA_HEADERS,
    )
    resp.raise_for_status()

    events: list[TrendingEvent] = []
    for raw in resp.json():
        if raw.get("slug", "").endswith("-more-markets"):
            continue
        built = _build_event(raw)
        if built and built.volume_24h >= min_volume:
            events.append(built)
        if len(events) >= limit:
            break

    return events


def to_rows(events: list[TrendingEvent]) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [
        {
            "id": e.event_id,
            "title": e.title,
            "slug": e.slug,
            "category": e.category,
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
