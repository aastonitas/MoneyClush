"""Client for the Polymarket CLOB API.

Handles market discovery, order book snapshots, order placement,
and position tracking for BTC Up/Down markets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog

from moneyclush.data.models import (
    MarketInfo,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    OutcomeSide,
    Trade,
)

log = structlog.get_logger()

BTC_UP_DOWN_SLUG = "btc-up-or-down"


@dataclass
class PolymarketClient:
    """Thin wrapper around the Polymarket CLOB REST API."""

    base_url: str = "https://clob.polymarket.com"
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    _http: httpx.AsyncClient = field(default=None, init=False)

    async def __aenter__(self) -> PolymarketClient:
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10.0,
            headers=self._auth_headers(),
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._http:
            await self._http.aclose()

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["POLY_API_KEY"] = self.api_key
            headers["POLY_API_SECRET"] = self.api_secret
            headers["POLY_PASSPHRASE"] = self.api_passphrase
        return headers

    async def get_markets(
        self, slug_contains: str = BTC_UP_DOWN_SLUG
    ) -> list[dict[str, Any]]:
        resp = await self._http.get("/markets", params={"next_cursor": "MA=="})
        resp.raise_for_status()
        data = resp.json()
        markets = data if isinstance(data, list) else data.get("data", [])
        return [
            m
            for m in markets
            if slug_contains.lower() in m.get("question", "").lower()
            or slug_contains.lower() in m.get("slug", "").lower()
        ]

    async def get_order_book(
        self, token_id: str
    ) -> OrderBookSnapshot:
        resp = await self._http.get("/book", params={"token_id": token_id})
        resp.raise_for_status()
        data = resp.json()

        def parse_levels(raw: list[dict]) -> list[OrderBookLevel]:
            return [
                OrderBookLevel(price=float(lv["price"]), size=float(lv["size"]))
                for lv in raw
            ]

        return OrderBookSnapshot(
            bids=parse_levels(data.get("bids", [])),
            asks=parse_levels(data.get("asks", [])),
            timestamp_ms=int(time.time() * 1000),
        )

    async def get_recent_trades(
        self, token_id: str, limit: int = 50
    ) -> list[Trade]:
        resp = await self._http.get(
            "/trades",
            params={"token_id": token_id, "limit": limit},
        )
        resp.raise_for_status()
        raw_trades = resp.json()
        if isinstance(raw_trades, dict):
            raw_trades = raw_trades.get("data", [])
        trades: list[Trade] = []
        for t in raw_trades:
            trades.append(
                Trade(
                    side=OutcomeSide.UP,
                    price=float(t.get("price", 0)),
                    size=float(t.get("size", 0)),
                    timestamp_ms=int(t.get("timestamp", 0)),
                    is_taker_buy=t.get("side", "BUY") == "BUY",
                )
            )
        return trades

    async def build_market_state(
        self,
        market_info: MarketInfo,
        btc_price: float,
        btc_price_ts: int,
    ) -> MarketState:
        book_up, book_down, trades = await self._fetch_books_and_trades(market_info)
        return MarketState(
            info=market_info,
            book_up=book_up,
            book_down=book_down,
            recent_trades=trades,
            btc_price=btc_price,
            btc_price_timestamp_ms=btc_price_ts,
            timestamp_ms=int(time.time() * 1000),
        )

    async def _fetch_books_and_trades(
        self, info: MarketInfo
    ) -> tuple[OrderBookSnapshot, OrderBookSnapshot, list[Trade]]:
        book_up = await self.get_order_book(info.token_id_up)
        book_down = await self.get_order_book(info.token_id_down)
        trades = await self.get_recent_trades(info.token_id_up)
        return book_up, book_down, trades

    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> Optional[dict[str, Any]]:
        """Place a limit order. Returns the order response or None if paper mode."""
        log.info(
            "order.place",
            token_id=token_id[:16],
            side=side,
            price=price,
            size=size,
            order_type=order_type,
        )
        if not self.api_key:
            log.info("order.paper_mode", msg="No API key — simulating order")
            return {
                "id": "paper-" + str(int(time.time() * 1000)),
                "status": "SIMULATED",
                "price": price,
                "size": size,
            }

        payload = {
            "tokenID": token_id,
            "side": side,
            "price": str(price),
            "size": str(size),
            "type": order_type,
        }
        resp = await self._http.post("/order", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def cancel_order(self, order_id: str) -> bool:
        if not self.api_key:
            log.info("order.cancel_paper", order_id=order_id)
            return True
        resp = await self._http.delete(f"/order/{order_id}")
        return resp.status_code == 200
