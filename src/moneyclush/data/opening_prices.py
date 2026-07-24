"""Real window opening prices from exchange candles.

CRITICAL: the opening price is the single most important input to the
fair value model. Using an approximation (e.g. the first spot price seen
after discovering a market mid-window) produces systematically wrong
fair values and fabricates large false edges.

Polymarket resolves these markets against Chainlink data streams. OKX 1m
candles are a close proxy: the candle whose timestamp equals the window
start carries that minute's open, which is the window's opening price.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"

INSTRUMENTS = {
    "BTC": "BTC-USDT",
    "ETH": "ETH-USDT",
    "SOL": "SOL-USDT",
    "XRP": "XRP-USDT",
}


@dataclass
class OpeningPriceCache:
    """Fetches and caches real opening prices per (asset, window_start)."""

    _cache: dict[tuple[str, int], float] = field(default_factory=dict)
    _misses: set[tuple[str, int]] = field(default_factory=set)

    async def get(
        self,
        client: httpx.AsyncClient,
        asset: str,
        window_start_epoch: int,
    ) -> float | None:
        """Return the real opening price for a window, or None if unavailable.

        Never guesses: returning None is correct behaviour when the true
        opening cannot be established, and callers must skip fair value
        rather than trade on an approximation.
        """
        key = (asset, window_start_epoch)
        if key in self._cache:
            return self._cache[key]

        inst = INSTRUMENTS.get(asset)
        if inst is None:
            return None

        # How far back is the window start? Fetch enough 1m candles to reach it.
        age_minutes = int((time.time() - window_start_epoch) / 60) + 3
        limit = max(5, min(300, age_minutes))

        try:
            resp = await client.get(
                OKX_CANDLES_URL,
                params={"instId": inst, "bar": "1m", "limit": str(limit)},
            )
            resp.raise_for_status()
            candles = resp.json().get("data", [])
        except Exception:
            return None

        for c in candles:
            if int(c[0]) // 1000 == window_start_epoch:
                opening = float(c[1])
                self._cache[key] = opening
                self._misses.discard(key)
                self._prune()
                return opening

        self._misses.add(key)
        return None

    def _prune(self) -> None:
        """Drop cache entries older than 2 hours."""
        cutoff = time.time() - 7200
        for key in [k for k in self._cache if k[1] < cutoff]:
            self._cache.pop(key, None)
        for key in [k for k in self._misses if k[1] < cutoff]:
            self._misses.discard(key)
