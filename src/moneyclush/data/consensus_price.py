"""Multi-source consensus spot price.

Why this exists: in a 5-minute window the entire signal is the distance
between the current price and the window's opening price, measured in
basis points. Typical BTC volatility over the remaining time is only a
few bps, while the price dispersion between individual exchanges is
routinely ~10 bps. A single-exchange feed therefore injects an error
several times larger than the signal it is meant to measure, which shows
up as large phantom "edges" against a market that is actually correct.

Polymarket resolves these markets against Chainlink, which itself
aggregates many venues. Taking the median across several exchanges is a
far closer proxy to that aggregate than any single venue, and the spread
between sources is a direct, reportable measure of how much to trust the
resulting fair value.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field

import httpx

SYMBOLS = {
    "BTC": {"okx": "BTC-USDT", "coinbase": "BTC-USD", "kraken": "XBTUSD", "bybit": "BTCUSDT"},
    "ETH": {"okx": "ETH-USDT", "coinbase": "ETH-USD", "kraken": "ETHUSD", "bybit": "ETHUSDT"},
    "SOL": {"okx": "SOL-USDT", "coinbase": "SOL-USD", "kraken": "SOLUSD", "bybit": "SOLUSDT"},
    "XRP": {"okx": "XRP-USDT", "coinbase": "XRP-USD", "kraken": "XRPUSD", "bybit": "XRPUSDT"},
}


@dataclass
class ConsensusPrice:
    asset: str
    median: float
    sources: dict[str, float]
    dispersion_bps: float
    timestamp_ms: int

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def reliable(self) -> bool:
        """At least 3 sources agreeing within a usable band."""
        return self.source_count >= 3 and self.dispersion_bps < 25.0


async def _okx(client: httpx.AsyncClient, sym: str) -> float | None:
    try:
        r = await client.get(
            "https://www.okx.com/api/v5/market/ticker", params={"instId": sym}
        )
        return float(r.json()["data"][0]["last"])
    except Exception:
        return None


async def _coinbase(client: httpx.AsyncClient, sym: str) -> float | None:
    try:
        r = await client.get(f"https://api.coinbase.com/v2/prices/{sym}/spot")
        return float(r.json()["data"]["amount"])
    except Exception:
        return None


async def _kraken(client: httpx.AsyncClient, sym: str) -> float | None:
    try:
        r = await client.get(
            "https://api.kraken.com/0/public/Ticker", params={"pair": sym}
        )
        result = r.json()["result"]
        first = next(iter(result.values()))
        return float(first["c"][0])
    except Exception:
        return None


async def _bybit(client: httpx.AsyncClient, sym: str) -> float | None:
    try:
        r = await client.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "spot", "symbol": sym},
        )
        return float(r.json()["result"]["list"][0]["lastPrice"])
    except Exception:
        return None


async def fetch_consensus(
    client: httpx.AsyncClient, asset: str
) -> ConsensusPrice | None:
    """Fetch the same asset from several venues concurrently and take the median.

    Concurrency matters: sequential requests spread over a second would
    themselves introduce price drift into the dispersion measurement.
    """
    syms = SYMBOLS.get(asset)
    if syms is None:
        return None

    results = await asyncio.gather(
        _okx(client, syms["okx"]),
        _coinbase(client, syms["coinbase"]),
        _kraken(client, syms["kraken"]),
        _bybit(client, syms["bybit"]),
        return_exceptions=True,
    )

    sources: dict[str, float] = {}
    for name, value in zip(("okx", "coinbase", "kraken", "bybit"), results):
        if isinstance(value, float) and value > 0:
            sources[name] = value

    if not sources:
        return None

    values = list(sources.values())
    median = statistics.median(values)
    dispersion = (
        (max(values) - min(values)) / median * 10000 if len(values) > 1 else 0.0
    )

    return ConsensusPrice(
        asset=asset,
        median=median,
        sources=sources,
        dispersion_bps=dispersion,
        timestamp_ms=int(time.time() * 1000),
    )


@dataclass
class ConsensusFeed:
    """Keeps the latest consensus price per asset."""

    latest: dict[str, ConsensusPrice] = field(default_factory=dict)

    async def refresh(
        self, client: httpx.AsyncClient, assets: list[str]
    ) -> dict[str, ConsensusPrice]:
        results = await asyncio.gather(
            *(fetch_consensus(client, a) for a in assets), return_exceptions=True
        )
        for asset, res in zip(assets, results):
            if isinstance(res, ConsensusPrice):
                self.latest[asset] = res
        return self.latest

    def price(self, asset: str) -> float:
        cp = self.latest.get(asset)
        return cp.median if cp else 0.0

    def dispersion_bps(self, asset: str) -> float:
        cp = self.latest.get(asset)
        return cp.dispersion_bps if cp else float("inf")
