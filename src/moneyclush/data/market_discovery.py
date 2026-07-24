"""Real market discovery via Polymarket Gamma API.

Up/Down markets follow slug pattern: {asset}-updown-{duration}-{window_epoch}
where window_epoch is the unix timestamp of the window start, aligned to
the duration boundary (5m -> multiples of 300s, 15m -> 900s).

No authentication required for market discovery or order books.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

GAMMA_URL = "https://gamma-api.polymarket.com"
UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

ASSETS = {
    "BTC": "btc",
    "ETH": "eth",
    "SOL": "sol",
    "XRP": "xrp",
}

DURATIONS = {
    "5m": 300,
    "15m": 900,
}


@dataclass
class DiscoveredMarket:
    asset: str
    duration: str
    slug: str
    title: str
    condition_id: str
    token_id_up: str
    token_id_down: str
    outcome_price_up: float
    outcome_price_down: float
    liquidity: float
    window_start_epoch: int
    window_end_epoch: int

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.window_end_epoch - time.time())


def current_window_slug(asset: str, duration: str, offset_windows: int = 0) -> str:
    """Compute the slug for the currently active (or offset) window."""
    mod = DURATIONS[duration]
    now = int(time.time())
    window = (now // mod + offset_windows) * mod
    return f"{ASSETS[asset]}-updown-{duration}-{window}", window


async def fetch_market(
    client: httpx.AsyncClient, asset: str, duration: str, offset_windows: int = 0
) -> DiscoveredMarket | None:
    """Fetch one Up/Down market for the given asset/duration window."""
    slug, window = current_window_slug(asset, duration, offset_windows)
    mod = DURATIONS[duration]

    resp = await client.get(
        f"{GAMMA_URL}/events", params={"slug": slug}, headers=UA_HEADERS
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not data:
        return None

    ev = data[0]
    m = ev["markets"][0]
    try:
        token_ids = json.loads(m["clobTokenIds"])
        prices = json.loads(m.get("outcomePrices", '["0.5","0.5"]'))
    except (KeyError, json.JSONDecodeError):
        return None

    return DiscoveredMarket(
        asset=asset,
        duration=duration,
        slug=slug,
        title=ev.get("title", slug),
        condition_id=m.get("conditionId", ""),
        token_id_up=token_ids[0],
        token_id_down=token_ids[1],
        outcome_price_up=float(prices[0]),
        outcome_price_down=float(prices[1]),
        liquidity=float(ev.get("liquidity", 0)),
        window_start_epoch=window,
        window_end_epoch=window + mod,
    )


async def discover_active_markets(
    client: httpx.AsyncClient,
    assets: list[str] | None = None,
    durations: list[str] | None = None,
) -> list[DiscoveredMarket]:
    """Discover all currently active Up/Down markets.

    If the current window market isn't found (created late), tries the next.
    """
    assets = assets or ["BTC", "ETH", "SOL", "XRP"]
    durations = durations or ["5m", "15m"]
    found: list[DiscoveredMarket] = []

    for asset in assets:
        for duration in durations:
            market = await fetch_market(client, asset, duration, 0)
            if market is None or market.seconds_remaining < 5:
                market = await fetch_market(client, asset, duration, 1)
            if market is not None:
                found.append(market)

    return found
