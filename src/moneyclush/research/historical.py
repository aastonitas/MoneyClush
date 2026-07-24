"""Fetch and store real historical data for backtesting.

Three sources are combined per window:
  - Gamma API      -> the resolution (which side actually won)
  - CLOB history   -> what the market charged for Up, minute by minute
  - OKX 1m candles -> the underlying BTC price, including the window's open

Window slugs are deterministic (`btc-updown-5m-<epoch>` aligned to 300s),
so past windows can be addressed directly without a search endpoint.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

GAMMA_URL = "https://gamma-api.polymarket.com/events"
CLOB_HISTORY_URL = "https://clob.polymarket.com/prices-history"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


@dataclass
class PricePoint:
    t: int      # unix seconds
    p: float    # market price of the Up outcome


@dataclass
class HistoricalWindow:
    slug: str
    asset: str
    duration_min: int
    window_start: int
    window_end: int
    winner: str                       # "Up" | "Down"
    volume: float
    up_prices: list[PricePoint] = field(default_factory=list)
    btc_open: float | None = None     # price at window start
    btc_close: float | None = None    # price at window end
    btc_path: list[list[float]] = field(default_factory=list)  # [[ts, close], ...]

    def to_json(self) -> dict:
        d = asdict(self)
        d["up_prices"] = [asdict(p) for p in self.up_prices]
        return d

    @classmethod
    def from_json(cls, d: dict) -> HistoricalWindow:
        pts = [PricePoint(**p) for p in d.pop("up_prices", [])]
        return cls(up_prices=pts, **d)

    @property
    def realized_up(self) -> int:
        return 1 if self.winner == "Up" else 0


async def _get_json(client: httpx.AsyncClient, url: str, params: dict) -> dict | list:
    resp = await client.get(url, params=params, headers=UA)
    resp.raise_for_status()
    return resp.json()


async def fetch_okx_candles(
    client: httpx.AsyncClient, inst: str, minutes: int
) -> dict[int, tuple[float, float]]:
    """Return {minute_ts: (open, close)} covering the last `minutes` minutes.

    OKX caps a single response at 300 candles, so older data is paged in
    with the `after` cursor.
    """
    out: dict[int, tuple[float, float]] = {}
    cursor: str | None = None

    while len(out) < minutes:
        params = {"instId": inst, "bar": "1m", "limit": "300"}
        if cursor:
            params["after"] = cursor
        try:
            data = (await _get_json(client, OKX_CANDLES_URL, params)).get("data", [])
        except Exception:
            break
        if not data:
            break

        for c in data:
            ts_ms = int(c[0])
            out[ts_ms // 1000] = (float(c[1]), float(c[4]))
        cursor = data[-1][0]
        await asyncio.sleep(0.12)

    return out


async def fetch_window(
    client: httpx.AsyncClient,
    window_start: int,
    asset: str = "btc",
    duration_min: int = 5,
) -> HistoricalWindow | None:
    """Fetch resolution + market price path for one past window."""
    slug = f"{asset}-updown-{duration_min}m-{window_start}"
    try:
        events = await _get_json(client, GAMMA_URL, {"slug": slug})
    except Exception:
        return None
    if not events:
        return None

    ev = events[0]
    m = ev["markets"][0]
    if not m.get("closed"):
        return None

    try:
        prices = json.loads(m["outcomePrices"])
        token_ids = json.loads(m["clobTokenIds"])
    except (KeyError, json.JSONDecodeError):
        return None

    # outcomePrices is [Up, Down]; the winner settles at "1"
    winner = "Up" if str(prices[0]).startswith("1") else "Down"
    window_end = window_start + duration_min * 60

    win = HistoricalWindow(
        slug=slug,
        asset=asset.upper(),
        duration_min=duration_min,
        window_start=window_start,
        window_end=window_end,
        winner=winner,
        volume=float(ev.get("volume") or 0),
    )

    try:
        hist = await _get_json(
            client,
            CLOB_HISTORY_URL,
            {
                "market": token_ids[0],
                "startTs": window_start,
                "endTs": window_end,
                "fidelity": "1",
            },
        )
        win.up_prices = [
            PricePoint(t=int(p["t"]), p=float(p["p"]))
            for p in hist.get("history", [])
            if window_start <= int(p["t"]) <= window_end
        ]
    except Exception:
        pass

    return win


def attach_btc_path(
    win: HistoricalWindow, candles: dict[int, tuple[float, float]]
) -> None:
    """Fill in the BTC open/close and intra-window price path from candles."""
    open_candle = candles.get(win.window_start)
    if open_candle:
        win.btc_open = open_candle[0]

    close_candle = candles.get(win.window_end)
    if close_candle:
        win.btc_close = close_candle[0]
    else:
        last = candles.get(win.window_end - 60)
        if last:
            win.btc_close = last[1]

    # A candle stamped `ts` covers [ts, ts+60), so its CLOSE is the price at
    # ts+60 — information that does not exist yet at ts. Only the OPEN is
    # causally available at ts. Using the close here leaks 60 seconds of the
    # future into every forecast and inflates backtest results enormously.
    path = []
    ts = win.window_start
    while ts < win.window_end:
        c = candles.get(ts)
        if c:
            path.append([ts, c[0]])
        ts += 60
    win.btc_path = path


async def fetch_history(
    num_windows: int = 300,
    asset: str = "btc",
    duration_min: int = 5,
    out_path: Path | None = None,
    progress: bool = True,
) -> list[HistoricalWindow]:
    """Download the last `num_windows` resolved windows with full context."""
    step = duration_min * 60
    now = int(time.time())
    current = now // step * step
    # skip the in-flight window and the one that may not have settled yet
    starts = [current - (i + 2) * step for i in range(num_windows)]
    oldest = min(starts)
    minutes_needed = int((now - oldest) / 60) + 10

    results: list[HistoricalWindow] = []

    async with httpx.AsyncClient(timeout=20.0) as client:
        if progress:
            print(f"Descargando {minutes_needed} velas OKX de 1m...")
        candles = await fetch_okx_candles(client, "BTC-USDT", minutes_needed)
        if progress:
            print(f"  {len(candles)} velas obtenidas")
            print(f"Descargando {num_windows} ventanas de Polymarket...")

        for i, ws in enumerate(starts):
            win = await fetch_window(client, ws, asset, duration_min)
            if win is not None:
                attach_btc_path(win, candles)
                results.append(win)
            if progress and (i + 1) % 25 == 0:
                print(f"  {i+1}/{num_windows} · {len(results)} válidas")
            await asyncio.sleep(0.08)

    results.sort(key=lambda w: w.window_start)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for w in results:
                f.write(json.dumps(w.to_json()) + "\n")
        if progress:
            print(f"\nGuardado en {out_path} ({len(results)} ventanas)")

    return results


def load_history(path: Path) -> list[HistoricalWindow]:
    windows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                windows.append(HistoricalWindow.from_json(json.loads(line)))
    return windows
