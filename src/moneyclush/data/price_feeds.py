"""External crypto price feeds via OKX WebSocket and REST fallback.

Binance is geo-blocked (HTTP 451) from this location, so OKX is the
primary source with Coinbase as REST fallback.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx
import structlog
import websockets

log = structlog.get_logger()

OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OKX_REST_URL = "https://www.okx.com/api/v5/market/ticker"
COINBASE_URL = "https://api.coinbase.com/v2/prices"


@dataclass
class PriceTick:
    symbol: str
    price: float
    timestamp_ms: int


@dataclass
class PriceFeed:
    """Streams real-time spot prices from OKX public WebSocket."""

    ws_url: str = OKX_WS_URL
    inst_id: str = "BTC-USDT"
    _latest: PriceTick | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)

    @property
    def latest(self) -> PriceTick | None:
        return self._latest

    @property
    def price(self) -> float:
        return self._latest.price if self._latest else 0.0

    @property
    def age_ms(self) -> float:
        if self._latest is None:
            return float("inf")
        return time.time() * 1000 - self._latest.timestamp_ms

    async def start(self) -> None:
        self._running = True
        log.info("price_feed.starting", url=self.ws_url, inst=self.inst_id)
        subscribe_msg = json.dumps({
            "op": "subscribe",
            "args": [{"channel": "tickers", "instId": self.inst_id}],
        })
        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    await ws.send(subscribe_msg)
                    log.info("price_feed.connected")
                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        data = msg.get("data")
                        if not data:
                            continue
                        tick = data[0]
                        self._latest = PriceTick(
                            symbol=self.inst_id,
                            price=float(tick["last"]),
                            timestamp_ms=int(tick["ts"]),
                        )
            except (websockets.ConnectionClosed, OSError) as exc:
                log.warning("price_feed.disconnected", error=str(exc))
                if self._running:
                    await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    async def fetch_once(self) -> PriceTick:
        """REST fallback — OKX first, then Coinbase."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(OKX_REST_URL, params={"instId": self.inst_id})
                resp.raise_for_status()
                tick_data = resp.json()["data"][0]
                tick = PriceTick(
                    symbol=self.inst_id,
                    price=float(tick_data["last"]),
                    timestamp_ms=int(tick_data["ts"]),
                )
            except Exception:
                base = self.inst_id.split("-")[0]
                resp = await client.get(f"{COINBASE_URL}/{base}-USD/spot")
                resp.raise_for_status()
                tick = PriceTick(
                    symbol=self.inst_id,
                    price=float(resp.json()["data"]["amount"]),
                    timestamp_ms=int(time.time() * 1000),
                )
            self._latest = tick
            return tick
