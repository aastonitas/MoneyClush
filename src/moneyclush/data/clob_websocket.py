"""Live CLOB order books over WebSocket.

REST polling gives a snapshot every few seconds. In a five-minute market
that is a very long time to be looking at a stale book: the quotes worth
taking are precisely the ones that are about to be cancelled, and they
are gone long before the next poll lands.

The `market` channel pushes two event types:
  - `book`         full snapshot, sent on subscribe and after big changes
  - `price_change` one or more level updates, each carrying the resulting
                   best_bid / best_ask for that asset

Because `price_change` already reports top of book, best bid/ask stay
current without replaying the whole book. Depth is still maintained from
the level updates for sizing decisions.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field

import structlog
import websockets

from moneyclush.data.models import OrderBookLevel, OrderBookSnapshot

log = structlog.get_logger()

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class LiveBook:
    """Current state of one outcome token's order book."""

    asset_id: str
    bids: dict[float, float] = field(default_factory=dict)   # price -> size
    asks: dict[float, float] = field(default_factory=dict)
    best_bid: float | None = None
    best_ask: float | None = None
    last_update_ms: int = 0
    updates: int = 0

    def apply_snapshot(self, bids: list[dict], asks: list[dict]) -> None:
        self.bids = {round(float(l["price"]), 4): float(l["size"]) for l in bids
                     if float(l["size"]) > 0}
        self.asks = {round(float(l["price"]), 4): float(l["size"]) for l in asks
                     if float(l["size"]) > 0}
        self.best_bid = max(self.bids) if self.bids else None
        self.best_ask = min(self.asks) if self.asks else None
        self.last_update_ms = int(time.time() * 1000)
        self.updates += 1

    def apply_change(self, change: dict) -> None:
        price = round(float(change["price"]), 4)
        size = float(change["size"])
        side = change.get("side", "").upper()

        book = self.bids if side == "BUY" else self.asks
        if size > 0:
            book[price] = size
        else:
            book.pop(price, None)

        # The event reports the resulting top of book directly.
        bb, ba = change.get("best_bid"), change.get("best_ask")
        self.best_bid = float(bb) if bb not in (None, "") else (
            max(self.bids) if self.bids else None
        )
        self.best_ask = float(ba) if ba not in (None, "") else (
            min(self.asks) if self.asks else None
        )
        self.last_update_ms = int(time.time() * 1000)
        self.updates += 1

    @property
    def age_ms(self) -> float:
        if not self.last_update_ms:
            return float("inf")
        return time.time() * 1000 - self.last_update_ms

    def to_snapshot(self) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            bids=[
                OrderBookLevel(price=p, size=s)
                for p, s in sorted(self.bids.items(), reverse=True)
            ],
            asks=[
                OrderBookLevel(price=p, size=s)
                for p, s in sorted(self.asks.items())
            ],
            timestamp_ms=self.last_update_ms,
        )


@dataclass
class ClobWebSocket:
    """Keeps live books for a changing set of outcome tokens.

    Subscriptions are fixed for the life of a connection, so changing the
    token set (every five minutes, as windows roll over) reconnects.
    """

    books: dict[str, LiveBook] = field(default_factory=dict)
    _tokens: set[str] = field(default_factory=set)
    _task: asyncio.Task | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _resubscribe: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    connected: bool = field(default=False, init=False)
    messages_received: int = field(default=0, init=False)
    connected_since_ms: int = field(default=0, init=False)

    def book(self, token_id: str) -> LiveBook | None:
        return self.books.get(token_id)

    def snapshot(self, token_id: str) -> OrderBookSnapshot:
        lb = self.books.get(token_id)
        return lb.to_snapshot() if lb else OrderBookSnapshot()

    def has_fresh(self, token_id: str, max_age_ms: float = 30_000) -> bool:
        lb = self.books.get(token_id)
        return lb is not None and lb.best_ask is not None and lb.age_ms < max_age_ms

    @property
    def updates_per_second(self) -> float:
        if not self.connected_since_ms:
            return 0.0
        elapsed = (time.time() * 1000 - self.connected_since_ms) / 1000
        return self.messages_received / elapsed if elapsed > 0 else 0.0

    def set_tokens(self, tokens: list[str]) -> None:
        """Replace the subscription set, reconnecting if it changed."""
        new = set(t for t in tokens if t)
        if new == self._tokens:
            return
        self._tokens = new
        for tid in list(self.books):
            if tid not in new:
                self.books.pop(tid, None)
        self._resubscribe.set()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._running = True
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        self._resubscribe.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while self._running:
            if not self._tokens:
                await asyncio.sleep(1)
                continue

            tokens = sorted(self._tokens)
            self._resubscribe.clear()
            try:
                async with websockets.connect(
                    WS_URL, open_timeout=10, ping_interval=15, ping_timeout=10
                ) as ws:
                    await ws.send(
                        json.dumps({"assets_ids": tokens, "type": "market"})
                    )
                    self.connected = True
                    self.connected_since_ms = int(time.time() * 1000)
                    self.messages_received = 0
                    log.info("clob_ws.connected", tokens=len(tokens))

                    resub = asyncio.create_task(self._resubscribe.wait())
                    try:
                        while self._running and not self._resubscribe.is_set():
                            recv = asyncio.create_task(ws.recv())
                            done, _ = await asyncio.wait(
                                {recv, resub},
                                return_when=asyncio.FIRST_COMPLETED,
                                timeout=30,
                            )
                            if recv in done:
                                self._handle(recv.result())
                            else:
                                recv.cancel()
                                if resub in done:
                                    break
                    finally:
                        resub.cancel()

            except Exception as exc:
                log.warning("clob_ws.disconnected", error=str(exc)[:80])
            finally:
                self.connected = False

            if self._running and not self._resubscribe.is_set():
                await asyncio.sleep(2)

    def _handle(self, raw: str | bytes) -> None:
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return

        self.messages_received += 1
        events = payload if isinstance(payload, list) else [payload]

        for ev in events:
            if not isinstance(ev, dict):
                continue
            etype = ev.get("event_type")

            if etype == "book":
                asset = ev.get("asset_id")
                if not asset:
                    continue
                self.books.setdefault(asset, LiveBook(asset_id=asset)).apply_snapshot(
                    ev.get("bids", []), ev.get("asks", [])
                )

            elif etype == "price_change":
                for change in ev.get("price_changes", []) or []:
                    asset = change.get("asset_id")
                    if not asset:
                        continue
                    self.books.setdefault(
                        asset, LiveBook(asset_id=asset)
                    ).apply_change(change)
