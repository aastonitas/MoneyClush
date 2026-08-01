"""MoneyClush Terminal Dashboard — FastAPI backend.

Polls Polymarket (Gamma + CLOB) and OKX/Coinbase in the background,
computes fair values and edges, runs paper trading with the temporal
arbitrage strategy, generates alerts + advisor tips, and persists
resolution metrics to data/paper_metrics.jsonl for validation.

Run:  python dashboard/server.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
import structlog
import uvicorn
from dataclasses import asdict
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

log = structlog.get_logger()

from moneyclush.data.clob_websocket import ClobWebSocket
from moneyclush.data.consensus_price import ConsensusFeed
from moneyclush.data.market_discovery import UA_HEADERS, discover_active_markets
from moneyclush.data.kalshi import fetch_crypto_ladders
from moneyclush.data.kalshi import to_rows as kalshi_to_rows
from moneyclush.data.news import fetch_headlines
from moneyclush.data.news import to_rows as news_to_rows
from moneyclush.data.predictions import PredictionLedger
from moneyclush.data import store
from moneyclush.data.standings import fetch_standings
from moneyclush.data.standings import to_rows as standings_to_rows
from moneyclush.data.sports import fetch_todays_matches, to_rows
from moneyclush.data.trending import fetch_trending
from moneyclush.data.trending import to_rows as trending_to_rows
from moneyclush.data.opening_prices import OpeningPriceCache
from moneyclush import notify as push
from moneyclush import trend_sim
from moneyclush.data.models import (
    MarketInfo,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    OutcomeSide,
    Position,
)
from moneyclush.pricing.fair_value import FairValueEngine, brownian_probability
from moneyclush.signals.order_book import combined_pair_cost, order_book_imbalance
from moneyclush.signals.trading_desk import build_desk
from moneyclush.strategies.categories import (
    FavouriteFade,
    LadderArb,
    SportsBasketArb,
)
from moneyclush.strategies.temporal_arbitrage import TemporalArbitrageStrategy
from moneyclush.execution.guardrails import KillSwitch, SafetyLimits

CLOB_URL = "https://clob.polymarket.com"
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
OKX_URL = "https://www.okx.com/api/v5/market/ticker"
COINBASE_URL = "https://api.coinbase.com/v2/prices"

# Polymarket's CLOB /balance-allowance endpoint reports 0 for both balance
# and allowance on this account even though the wallet verifiably holds
# pUSD and has max-uint256 allowance to the V2 exchange on-chain (see
# execution/engine.py item 0) — so the real balance is read straight off
# Polygon instead of trusting that endpoint.
PUSD_CONTRACT = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
PUSD_DECIMALS = 6
POLYGON_RPCS = [
    "https://polygon.drpc.org",
    "https://1rpc.io/matic",
    "https://polygon-bor-rpc.publicnode.com",
]
BALANCE_TTL = 30.0

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT / "data"
METRICS_FILE = DATA_DIR / "paper_metrics.jsonl"

app = FastAPI(title="MoneyClush Terminal")

# ---------------------------------------------------------------- shared state
STATE: dict = {
    "updated_ms": 0,
    "btc_price": 0.0,
    "eth_price": 0.0,
    "sol_price": 0.0,
    "xrp_price": 0.0,
    "markets": [],
    "scanner": [],
    "pnl_curve": [],
    "paper_pnl": 0.0,
    "best_pair_seen": None,
    "strategy_gate": {},
    # Real pUSD balance, read on-chain (see PUSD_CONTRACT above) — not the
    # paper PnL. None until the first successful read.
    "polymarket_balance": None,
    "polymarket_balance_curve": [],
    # Second paper track. The arbitrage strategy correctly refuses to trade
    # a pair that costs more than $1, so its curve is flat by design. This
    # one backs the market's favourite on every BTC window instead: it
    # settles every 5-15 minutes, so it actually moves, and it tests the
    # favourite-bias hypothesis at far higher frequency than sports can.
    "fav_curve": [],
    "fav_pnl": 0.0,
    "fav_stats": {"resolved": 0, "wins": 0, "staked": 0.0, "expected": 0.0},
    "paper_trades": [],
    "positions": {},
    "alerts": [],
    "advisor": [],
    "arb_events": [],
    "stats": {
        "resolved": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
        "fills": 0, "arbs_seen": 0,
    },
    "status": "starting",
    "poll_count": 0,
    "errors": 0,
    # Failures inside the per-market pricing block. Kept separate from
    # `errors` because a bug there used to be indistinguishable from a
    # quiet market: both produce an empty scanner and no fills.
    "pricing_errors": 0,
    "last_pricing_error": None,
    "btc_source": "okx",
    "storage": {"state": "unknown", "reason": "sin comprobar todavía"},
}

fv_engine = FairValueEngine()
strategy = TemporalArbitrageStrategy(block_size=25)

# Nothing places a live order yet (see execution/engine.py roadmap items
# 2-4) — this is the choke point that will gate it once that exists.
# Built and wired now so the emergency stop is proven to work before it is
# ever needed, not the day it is.
kill_switch = KillSwitch()
safety_limits = SafetyLimits()

paper_positions: dict[str, Position] = {}
# condition_id -> {slug, asset, duration, opening, window_end}
paper_meta: dict[str, dict] = {}

# Favourite-backing track: one $1 bet per BTC window, settled at window close.
fav_positions: dict[str, dict] = {}
fav_settled: set[str] = set()
FAV_MIN_PRICE = 0.50
FAV_MAX_PRICE = 0.92
opening_cache = OpeningPriceCache()        # real opening prices from OKX candles
consensus_feed = ConsensusFeed()           # cross-exchange dispersion monitor
clob_ws = ClobWebSocket()                  # live order books, ~300 updates/s

# Price distance is computed from OKX for both the opening candle and the
# current tick, so the USDT/USD basis cancels out. Cross-exchange dispersion
# is not used as the price — it is used to decide how much to trust it.
MAX_DISPERSION_BPS = 20.0

# Fees plus expected slippage on both legs of a paired trade.
ARB_COST = 0.018

# Fixtures move on the scale of hours; refetching every poll would be waste.
SPORTS_TTL = 40.0
SPORTS_CACHE: dict = {"rows": [], "fetched_at": 0.0, "error": None}

# Headlines and trending markets both move slower than the 3s poll loop.
NEWS_TTL = 300.0
NEWS_CACHE: dict[str, dict] = {
    cat: {"rows": [], "fetched_at": 0.0, "error": None}
    for cat in ("crypto", "sports", "general")
}
TRENDING_TTL = 120.0
TRENDING_CACHE: dict = {"rows": [], "fetched_at": 0.0, "error": None}

# Kalshi's hourly ladders reprice on the scale of seconds like any book,
# but the strike list itself only changes once an hour, and the panel is
# read far less often than the BTC scanner.
KALSHI_TTL = 45.0
KALSHI_CACHE: dict = {"rows": [], "fetched_at": 0.0, "error": None}

# Which spot price to compare each Kalshi series against. The ladder asks
# "will X be above this strike at the close", so the model needs the same
# asset's spot to compute a distance.
KALSHI_SPOT_KEY = {
    "KXBTCD": "btc_price", "KXBTC15M": "btc_price",
    "KXETHD": "eth_price", "KXETH15M": "eth_price",
    "KXSOLD": "sol_price", "KXXRPD": "xrp_price",
}

# Favourite-backing ledger. Only fixtures the market rates at 55%+ are
# picked: below that there is no meaningful favourite to back.
PREDICTIONS = PredictionLedger(path=DATA_DIR / "predictions.jsonl")
PREDICTION_MIN_PROB = 0.55
PREDICTION_INTERVAL = 120.0
PREDICTION_STAKE = 1.0

# Outside this band the market has effectively already resolved, so a pick
# there is not a forecast — it is either a certainty or a dead ticket.
MIN_PICK_PRICE = 0.02
MAX_PICK_PRICE = 0.97

# League tables refresh once a day upstream; hourly here is already generous.
STANDINGS_TTL = 1800.0
STANDINGS_CACHE: dict = {"rows": [], "fetched_at": 0.0, "error": None}
alerted_edges: set[str] = set()            # dedupe: slug+side alerted once per window
alerted_arbs: set[str] = set()              # dedupe: push once per window, not once per 3s poll
alerted_favourites: set[str] = set()        # dedupe: one push per sports fixture

# Below this, a sports favourite is not remarkable enough to interrupt
# someone for — 70-90% is exactly the band the BTC backtest found the
# market itself gets wrong most often.
FAV_PUSH_THRESHOLD = 0.75


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _favourite_mid(book_up, book_down, side: OutcomeSide) -> float | None:
    """The market's own probability for the backed side, de-vigged.

    Up and Down are complementary tokens, so their mids already sum to
    about $1; normalising by the total removes whatever is left over and
    makes the number a probability rather than a price.

    Returns None when either book is one-sided, in which case there is no
    mid to speak of and the calibration falls back to the ask.
    """
    mids = []
    for book in (book_up, book_down):
        if book.best_bid is None or book.best_ask is None:
            return None
        mids.append((book.best_bid + book.best_ask) / 2.0)
    total = mids[0] + mids[1]
    if total <= 0:
        return None
    return (mids[0] if side == OutcomeSide.UP else mids[1]) / total


def add_alert(level: str, text: str) -> None:
    """level: edge | fill | resolve | warn | info"""
    STATE["alerts"].insert(0, {
        "ts": int(time.time() * 1000),
        "level": level,
        "text": text,
    })
    STATE["alerts"] = STATE["alerts"][:50]


def notify_push(title: str, body: str, tag: str | None = None, url: str | None = None) -> None:
    """Fire a Web Push to every subscribed browser, without blocking the poll.

    Reserved for the handful of things worth interrupting someone for:
    risk-free arbitrage, a net BTC edge, storage falling back to
    ephemeral, and a sports favourite crossing the alert threshold.
    Everything else stays in the in-app alert feed, which nobody has to
    be looking at to eventually see.

    `webpush()` is a blocking HTTP call; running it inline on the poll
    loop would stall every market's pricing for however long the push
    services take to answer, so it goes to a worker thread instead.
    """
    async def _send():
        try:
            delivered = await asyncio.to_thread(push.send_to_all, title, body, tag, url)
            if delivered:
                log.info("notify.sent", title=title, delivered=delivered)
        except Exception as exc:
            log.warning("notify.dispatch_failed", error=str(exc)[:160])

    asyncio.create_task(_send())


def persist_metric(record: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


async def fetch_book(client: httpx.AsyncClient, token_id: str) -> OrderBookSnapshot:
    resp = await client.get(
        f"{CLOB_URL}/book", params={"token_id": token_id}, headers=UA_HEADERS
    )
    resp.raise_for_status()
    data = resp.json()

    def levels(raw: list[dict], reverse: bool) -> list[OrderBookLevel]:
        parsed = [
            OrderBookLevel(price=float(l["price"]), size=float(l["size"]))
            for l in raw
        ]
        parsed.sort(key=lambda l: l.price, reverse=reverse)
        return parsed

    return OrderBookSnapshot(
        bids=levels(data.get("bids", []), reverse=True),
        asks=levels(data.get("asks", []), reverse=False),
        timestamp_ms=int(time.time() * 1000),
    )


async def fetch_spot_prices(client: httpx.AsyncClient) -> tuple[dict[str, float], dict[str, str]]:
    """Spot prices from OKX (primary) with Coinbase fallback.

    Binance is geo-blocked (HTTP 451) from this location.

    Returns the prices *and where each one came from*, because the source
    is not an implementation detail here. The opening price always comes
    from OKX 1m candles, and the model prices the distance between the
    two. OKX quotes USDT and Coinbase quotes USD, a basis of roughly 10
    bps — four times the per-minute volatility the signal is made of. The
    basis cancels only while both ends of the subtraction share a venue,
    so a silent failover turns a working model into a random one. The
    caller uses this to stop pricing rather than to price badly.
    """
    out: dict[str, float] = {}
    sources: dict[str, str] = {}
    for inst, cb_pair, key in [
        ("BTC-USDT", "BTC-USD", "btc_price"),
        ("ETH-USDT", "ETH-USD", "eth_price"),
        ("SOL-USDT", "SOL-USD", "sol_price"),
        ("XRP-USDT", "XRP-USD", "xrp_price"),
    ]:
        try:
            r = await client.get(OKX_URL, params={"instId": inst})
            out[key] = float(r.json()["data"][0]["last"])
            sources[key] = "okx"
        except Exception:
            try:
                r = await client.get(f"{COINBASE_URL}/{cb_pair}/spot")
                out[key] = float(r.json()["data"]["amount"])
                sources[key] = "coinbase"
            except Exception:
                out[key] = STATE.get(key, 0.0)
                sources[key] = "stale"
    return out, sources


async def fetch_pusd_balance(client: httpx.AsyncClient, address: str) -> float | None:
    """Real pUSD balance for `address`, read straight off Polygon.

    Tries each public RPC in turn — free endpoints rate-limit or blip
    individually often enough that a single one is not reliable for a
    number shown as truth on the dashboard.
    """
    data = "0x70a08231" + "000000000000000000000000" + address.removeprefix("0x").lower()
    for rpc in POLYGON_RPCS:
        try:
            resp = await client.post(
                rpc,
                json={
                    "jsonrpc": "2.0",
                    "method": "eth_call",
                    "params": [{"to": PUSD_CONTRACT, "data": data}, "latest"],
                    "id": 1,
                },
                timeout=8.0,
            )
            result = resp.json()
            if "result" not in result:
                continue
            return int(result["result"], 16) / 10**PUSD_DECIMALS
        except Exception:
            continue
    return None


def build_advisor(scanner_rows: list[dict], market_rows: list[dict]) -> list[dict]:
    """Rule-based contextual tips about the current market state."""
    tips: list[dict] = []

    edges = [r for r in scanner_rows if r["signal"] == "EDGE"]
    if edges:
        top = edges[0]
        tips.append({
            "icon": "▲",
            "kind": "action",
            "text": f"Edge neto de {top['edge']*100:.1f}% en {top['market']} {top['side']} "
                    f"(z={top.get('z', 0):.2f}) — verificar que no venga de liquidez "
                    f"fantasma antes de confiar en él.",
        })
    else:
        tips.append({
            "icon": "◼",
            "kind": "ok",
            "text": "Sin edge neto ejecutable ahora. Es lo normal y lo correcto: "
                    "el mercado de 5m suele estar bien valorado y la mayoría de "
                    "señales se descartan antes de enviar una orden.",
        })

    unpriceable = [m for m in market_rows if m.get("opening") is None]
    if unpriceable:
        tips.append({
            "icon": "!",
            "kind": "warn",
            "text": f"{len(unpriceable)} mercados sin precio de apertura real — "
                    f"no se valoran. Nunca estimar el opening: produce edges falsos.",
        })

    sigma = STATE.get("sigma_1m_bps")
    disp = STATE.get("dispersion_bps")
    if sigma and disp is not None:
        ratio = disp / sigma if sigma else 0
        if disp > MAX_DISPERSION_BPS:
            tips.append({
                "icon": "!",
                "kind": "warn",
                "text": f"Dispersión entre exchanges {disp} bps supera el límite "
                        f"({MAX_DISPERSION_BPS} bps) — trading pausado. El error de "
                        f"fuente sería mayor que la señal.",
            })
        else:
            tips.append({
                "icon": "σ",
                "kind": "ok",
                "text": f"σ BTC {sigma} bps/min · dispersión entre exchanges {disp} bps "
                        f"({ratio:.1f}× la vol por minuto). El fair value escala σ al "
                        f"tiempo restante: el mismo movimiento pesa mucho más al final.",
            })

    for r in scanner_rows:
        if r.get("pair_cost") and r["pair_cost"] < 1.0:
            profit = (1.0 - r["pair_cost"]) * 100
            tips.append({
                "icon": "$",
                "kind": "action",
                "text": f"Pair cost {r['pair_cost']*100:.0f}¢ < 100¢ en {r['market']} — "
                        f"arbitraje instantáneo de {profit:.1f}¢/par si la profundidad aguanta.",
            })
            break

    low_liq = [m for m in market_rows if m["liquidity"] < 1500]
    if low_liq:
        names = ", ".join(f"{m['asset']} {m['duration']}" for m in low_liq[:3])
        tips.append({
            "icon": "!",
            "kind": "warn",
            "text": f"Liquidez baja en {names} — el precio visible puede no ser ejecutable "
                    f"a tamaño real. Evitar o reducir block size.",
        })

    # The favourite track is the live version of the open question from the
    # backtest, so its reading belongs here — with the sample size attached,
    # because a 9-point gap over 30 bets is noise wearing a result's clothes.
    fav = STATE.get("fav_stats") or {}
    if fav.get("resolved"):
        tips.append({
            "icon": "%",
            "kind": "warn" if fav.get("significant") else "ok",
            "text": f"Backing al favorito ({fav['resolved']} resueltas): "
                    f"{fav.get('verdict', '')}",
        })

    storage = STATE.get("storage") or {}
    if storage.get("state") == "ephemeral":
        tips.insert(0, {
            "icon": "!",
            "kind": "warn",
            "text": f"Los datos NO persisten. {storage.get('reason', '')}",
        })

    tips.append({
        "icon": "»",
        "kind": "next",
        "text": "Dejar el paper trading corriendo horas/días: las métricas se guardan "
                "en data/paper_metrics.jsonl para validar la estrategia.",
    })
    tips.append({
        "icon": "»",
        "kind": "next",
        "text": "Cuando el win rate y PnL paper sean consistentes, configurar API key "
                "de Polymarket + wallet Polygon para pasar a órdenes reales.",
    })
    return tips[:6]


async def calibrate_from_candles(client: httpx.AsyncClient) -> None:
    """Set the model's volatility from recent realized BTC 1m returns."""
    try:
        resp = await client.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": "BTC-USDT", "bar": "1m", "limit": "300"},
        )
        closes = [float(c[4]) for c in resp.json()["data"]][::-1]
        sigma = fv_engine.calibrate_volatility(closes)
        STATE["sigma_1m_bps"] = round(sigma * 10000, 2)
        add_alert(
            "info",
            f"Volatilidad calibrada: sigma 1m = {sigma*10000:.2f} bps "
            f"({len(closes)} velas)",
        )
    except Exception as exc:
        add_alert("warn", f"Calibración de volatilidad falló: {str(exc)[:40]}")


async def poll_loop() -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        add_alert("info", "Sistema iniciado — modo PAPER, sin órdenes reales")
        await calibrate_from_candles(client)
        last_calibration = time.time()
        last_balance_check = 0.0
        while True:
            try:
                if time.time() - last_calibration > 900:
                    await calibrate_from_candles(client)
                    last_calibration = time.time()

                funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
                if funder and time.time() - last_balance_check > BALANCE_TTL:
                    last_balance_check = time.time()
                    balance = await fetch_pusd_balance(client, funder)
                    if balance is not None:
                        STATE["polymarket_balance"] = round(balance, 6)
                        curve = STATE["polymarket_balance_curve"]
                        curve.append([int(time.time() * 1000), round(balance, 6)])
                        STATE["polymarket_balance_curve"] = curve[-600:]

                spots, spot_sources = await fetch_spot_prices(client)
                STATE.update(spots)
                now_ms = int(time.time() * 1000)

                # Dispersion across venues tells us how much to trust the tick.
                await consensus_feed.refresh(client, ["BTC"])
                btc_dispersion = consensus_feed.dispersion_bps("BTC")
                STATE["dispersion_bps"] = (
                    round(btc_dispersion, 2) if btc_dispersion != float("inf") else None
                )
                # The opening price is an OKX candle, so the current tick has
                # to be OKX too or the USDT/USD basis stops cancelling and
                # swamps the signal. A fallback quote is better than no price
                # on screen, but it is not something to price a market with.
                btc_source = spot_sources.get("btc_price", "stale")
                STATE["btc_source"] = btc_source
                same_venue = btc_source == "okx"
                data_trusted = btc_dispersion <= MAX_DISPERSION_BPS and same_venue

                markets = await discover_active_markets(
                    client, assets=["BTC", "ETH", "SOL", "XRP"], durations=["5m", "15m"]
                )

                # Keep the live book subscription aligned with the BTC windows
                # currently in play; this reconnects when a window rolls over.
                clob_ws.set_tokens([
                    t
                    for mk in markets
                    if mk.asset == "BTC"
                    for t in (mk.token_id_up, mk.token_id_down)
                ])
                STATE["ws_connected"] = clob_ws.connected
                STATE["ws_updates_per_sec"] = round(clob_ws.updates_per_second, 1)

                market_rows = []
                scanner_rows = []

                for mk in markets:
                    spot_key = f"{mk.asset.lower()}_price"
                    spot = spots.get(spot_key, 0.0)
                    # Real opening price from exchange candles. None means we
                    # cannot price this market — never substitute an estimate.
                    opening = await opening_cache.get(
                        client, mk.asset, mk.window_start_epoch
                    )

                    row = {
                        "asset": mk.asset,
                        "duration": mk.duration,
                        "title": mk.title,
                        "slug": mk.slug,
                        "price_up": mk.outcome_price_up,
                        "price_down": mk.outcome_price_down,
                        "liquidity": mk.liquidity,
                        "seconds_remaining": round(mk.seconds_remaining),
                        "spot": spot,
                        "opening": opening,
                        "price_source": spot_sources.get(spot_key, "stale"),
                    }

                    # Pricing needs the tick and the opening candle to share a
                    # venue; on a failover the honest move is to show the market
                    # unpriced rather than to print a fair value built on a
                    # 10 bps basis error.
                    if mk.asset == "BTC" and opening is not None and same_venue:
                        try:
                            # Live books when the socket is current; REST only
                            # as a fallback while it connects or reconnects.
                            if clob_ws.has_fresh(mk.token_id_up) and clob_ws.has_fresh(
                                mk.token_id_down
                            ):
                                book_up = clob_ws.snapshot(mk.token_id_up)
                                book_down = clob_ws.snapshot(mk.token_id_down)
                                row["book_source"] = "ws"
                                row["book_age_ms"] = round(
                                    max(
                                        clob_ws.book(mk.token_id_up).age_ms,
                                        clob_ws.book(mk.token_id_down).age_ms,
                                    )
                                )
                            else:
                                book_up = await fetch_book(client, mk.token_id_up)
                                book_down = await fetch_book(client, mk.token_id_down)
                                row["book_source"] = "rest"
                                row["book_age_ms"] = 0

                            info = MarketInfo(
                                condition_id=mk.condition_id,
                                token_id_up=mk.token_id_up,
                                token_id_down=mk.token_id_down,
                                question=mk.title,
                                duration_minutes=5 if mk.duration == "5m" else 15,
                                opening_price_btc=opening,
                                open_timestamp_ms=mk.window_start_epoch * 1000,
                                close_timestamp_ms=mk.window_end_epoch * 1000,
                            )
                            state_obj = MarketState(
                                info=info,
                                book_up=book_up,
                                book_down=book_down,
                                btc_price=spot,
                                btc_price_timestamp_ms=now_ms,
                                timestamp_ms=now_ms,
                            )
                            fv = fv_engine.evaluate(state_obj)
                            imb = order_book_imbalance(book_up)
                            pair = combined_pair_cost(book_up, book_down, 25)
                            if pair is not None:
                                best = STATE.get("best_pair_seen")
                                if best is None or pair < best:
                                    STATE["best_pair_seen"] = pair

                            # -------- favourite-backing track ($1 per window)
                            # Entered once per window, roughly halfway through,
                            # so there is a real favourite to back rather than
                            # a coin flip at the open.
                            cid = mk.condition_id
                            seconds_left = mk.seconds_remaining
                            window_len = max(
                                mk.window_end_epoch - mk.window_start_epoch, 1
                            )
                            if (
                                cid not in fav_positions
                                and cid not in fav_settled
                                and seconds_left < window_len * 0.6
                                and seconds_left > 20
                                and book_up.best_ask is not None
                                and book_down.best_ask is not None
                            ):
                                # The favourite is the pricier side — the one
                                # the market rates more likely — not the
                                # cheaper one. Backing the cheap side would
                                # silently test the opposite hypothesis.
                                up_ask, down_ask = book_up.best_ask, book_down.best_ask
                                side = (
                                    OutcomeSide.UP if up_ask > down_ask else OutcomeSide.DOWN
                                )
                                price = max(up_ask, down_ask)
                                # The ask is what the bet costs; the mid is
                                # what the market actually thinks. Scoring the
                                # forecast against the ask charges it half a
                                # spread and invents a deficit the same size as
                                # the effect being measured, so both are kept.
                                mid = _favourite_mid(book_up, book_down, side)
                                if FAV_MIN_PRICE <= price <= FAV_MAX_PRICE:
                                    bet = {
                                        "side": side.value,
                                        "price": price,
                                        "mid": mid,
                                        "market": f"{mk.asset} {mk.duration}",
                                        "opening": opening,
                                        "window_end": mk.window_end_epoch,
                                        "asset": mk.asset,
                                    }
                                    fav_positions[cid] = bet
                                    # Held only in memory until now, so every
                                    # restart quietly dropped whatever was in
                                    # flight and the curve lost those windows.
                                    store.save_open_fav(cid, bet)

                            row.update({
                                "best_bid_up": book_up.best_bid,
                                "best_ask_up": book_up.best_ask,
                                "best_bid_down": book_down.best_bid,
                                "best_ask_down": book_down.best_ask,
                                "fair_up": round(fv.posterior_up, 4),
                                "net_edge_up": round(fv.net_edge_up, 4),
                                "net_edge_down": round(fv.net_edge_down, 4),
                                "book_imbalance": round(imb, 4),
                                "pair_cost": round(pair, 4) if pair else None,
                                "distance_bps": round(fv.distance_bps, 2),
                                "sigma_rem_bps": round(fv.sigma_remaining_bps, 2),
                                "z_score": round(fv.z_score, 2),
                            })
                            STATE["sigma_1m_bps"] = round(
                                fv_engine.sigma_per_minute * 10000, 2
                            )

                            if not fv.valid:
                                market_rows.append(row)
                                continue

                            # Up+Down below $1 pays out regardless of which
                            # side wins, so it does not depend on the model
                            # being right about anything. Only worth logging
                            # at a size the book can actually fill.
                            if pair is not None and pair < 1.0 - ARB_COST:
                                profit = 1.0 - pair - ARB_COST
                                STATE["stats"]["arbs_seen"] += 1
                                STATE["arb_events"].insert(0, {
                                    "ts": now_ms,
                                    "market": f"BTC {mk.duration}",
                                    "pair_cost": round(pair, 4),
                                    "profit_per_pair": round(profit, 4),
                                    "size": 25,
                                    "source": row.get("book_source", "rest"),
                                })
                                STATE["arb_events"] = STATE["arb_events"][:40]
                                add_alert(
                                    "edge",
                                    f"ARB BTC {mk.duration}: par a {pair*100:.1f}¢ "
                                    f"→ {profit*100:.2f}¢/par libre de riesgo",
                                )
                                # The condition can hold for several polls in a
                                # row; a push per 3s tick would be unusable.
                                if mk.slug not in alerted_arbs:
                                    alerted_arbs.add(mk.slug)
                                    notify_push(
                                        "Arbitraje libre de riesgo",
                                        f"BTC {mk.duration}: par a {pair*100:.1f}¢ → "
                                        f"{profit*100:.2f}¢/par sin riesgo direccional",
                                        tag=f"arb-{mk.slug}",
                                    )

                            best_edge = max(fv.net_edge_up, fv.net_edge_down)
                            edge_side = "UP" if fv.net_edge_up >= fv.net_edge_down else "DOWN"
                            signal_label = (
                                "EDGE" if best_edge > 0.02
                                else ("WATCH" if best_edge > 0 else "NONE")
                            )
                            scanner_rows.append({
                                "market": f"BTC {mk.duration}",
                                "slug": mk.slug,
                                "fair": round(fv.posterior_up if edge_side == "UP" else fv.posterior_down, 3),
                                "ask": row["best_ask_up"] if edge_side == "UP" else row["best_ask_down"],
                                "edge": round(best_edge, 4),
                                "side": edge_side,
                                "pair_cost": row["pair_cost"],
                                "signal": signal_label,
                                "z": fv.z_score,
                                "dist_bps": round(fv.distance_bps, 1),
                            })

                            edge_key = f"{mk.slug}:{edge_side}"
                            if signal_label == "EDGE" and edge_key not in alerted_edges:
                                alerted_edges.add(edge_key)
                                add_alert(
                                    "edge",
                                    f"EDGE {best_edge*100:.1f}% · BTC {mk.duration} {edge_side} "
                                    f"(fair vs ask)",
                                )
                                notify_push(
                                    "Edge neto en BTC",
                                    f"{best_edge*100:.1f}% neto · BTC {mk.duration} {edge_side} "
                                    f"(fair vs ask, tras costes)",
                                    tag=f"edge-{edge_key}",
                                )

                            # -------- paper trading (temporal arbitrage)
                            pos = paper_positions.get(
                                mk.condition_id,
                                Position(market_condition_id=mk.condition_id),
                            )
                            signal = strategy.evaluate(state_obj, fv, pos)
                            if signal is not None and signal.edge > 0 and data_trusted:
                                book = book_up if signal.side == OutcomeSide.UP else book_down
                                cost = book.executable_cost("buy", signal.target_size)
                                if cost is not None:
                                    pos.add_fill(signal.side, cost, signal.target_size)
                                    paper_positions[mk.condition_id] = pos
                                    paper_meta[mk.condition_id] = {
                                        "slug": mk.slug,
                                        "asset": mk.asset,
                                        "duration": mk.duration,
                                        "opening": opening,
                                        "window_end": mk.window_end_epoch,
                                        "spot_key": spot_key,
                                    }
                                    STATE["stats"]["fills"] += 1
                                    add_alert(
                                        "fill",
                                        f"FILL {signal.side.value.upper()} x{signal.target_size:.0f} "
                                        f"@ {cost*100:.0f}¢ · BTC {mk.duration}",
                                    )
                                    STATE["paper_trades"].insert(0, {
                                        "ts": now_ms,
                                        "market": f"BTC {mk.duration}",
                                        "side": signal.side.value,
                                        "price": round(cost, 4),
                                        "size": signal.target_size,
                                        "reason": signal.reason[:40],
                                    })
                                    STATE["paper_trades"] = STATE["paper_trades"][:30]
                        except Exception as exc:
                            # This block holds fair value, arbitrage detection,
                            # the favourite track and paper trading. Swallowing
                            # it silently meant any bug in ~190 lines showed up
                            # as nothing happening, which is also what a
                            # correctly quiet market looks like.
                            STATE["pricing_errors"] += 1
                            STATE["last_pricing_error"] = (
                                f"{type(exc).__name__}: {exc}"[:140]
                            )
                            log.warning(
                                "poll.market_failed",
                                slug=mk.slug,
                                error=str(exc)[:160],
                                exc_info=True,
                            )
                            row["error"] = type(exc).__name__

                    market_rows.append(row)

                # -------- settle the favourite-backing track
                now_s = time.time()
                for cid in list(fav_positions.keys()):
                    bet = fav_positions[cid]
                    if now_s <= bet["window_end"] + 5:
                        continue
                    final_spot = await opening_cache.get(
                        client, bet["asset"], bet["window_end"]
                    )
                    if final_spot is None:
                        continue  # retry next poll rather than guess

                    fav_positions.pop(cid)
                    fav_settled.add(cid)
                    store.drop_open_fav(cid)

                    winner = (
                        OutcomeSide.UP
                        if final_spot >= bet["opening"]
                        else OutcomeSide.DOWN
                    )
                    won = winner.value == bet["side"]
                    # $1 stake buys 1/price shares, each settling at $1.
                    pnl = (1.0 / bet["price"] - 1.0) if won else -1.0

                    store.save_fav_bet(
                        settled_at=int(now_s * 1000),
                        market=bet["market"],
                        side=bet["side"],
                        price=bet["price"],
                        won=won,
                        pnl=pnl,
                        mid=bet.get("mid"),
                    )
                    # Rebuild from the database so the curve reflects every
                    # bet ever settled, not just this process's lifetime.
                    STATE["fav_curve"], STATE["fav_stats"] = store.load_fav_history()
                    STATE["fav_pnl"] = round(store.fav_pnl(), 4)

                    STATE["paper_trades"].insert(0, {
                        "ts": int(now_s * 1000),
                        "market": bet["market"],
                        "side": f"FAV {bet['side'].upper()}",
                        "price": round(bet["price"], 4),
                        "size": 1.0,
                        "reason": f"{'ACIERTO' if won else 'fallo'} "
                                  f"{'+' if pnl >= 0 else ''}{pnl:.2f}$",
                    })
                    STATE["paper_trades"] = STATE["paper_trades"][:30]

                # -------- resolve expired paper positions (per position, precise)
                for cid in list(paper_positions.keys()):
                    meta = paper_meta.get(cid)
                    if meta is None:
                        continue
                    if now_s <= meta["window_end"] + 5:
                        continue
                    # Price exactly at window close: the open of the candle
                    # starting at window_end. Falls back to current spot only
                    # if candles are unavailable.
                    final_spot = await opening_cache.get(
                        client, meta["asset"], meta["window_end"]
                    )
                    if final_spot is None:
                        continue  # retry next poll rather than resolve blindly

                    pos = paper_positions.pop(cid)
                    paper_meta.pop(cid, None)

                    winner = (
                        OutcomeSide.UP
                        if final_spot >= meta["opening"]
                        else OutcomeSide.DOWN
                    )
                    pnl = pos.pnl_if_resolves(winner)
                    STATE["paper_pnl"] += pnl
                    stats = STATE["stats"]
                    stats["resolved"] += 1
                    if pnl >= 0:
                        stats["wins"] += 1
                    else:
                        stats["losses"] += 1
                    stats["win_rate"] = (
                        stats["wins"] / stats["resolved"] if stats["resolved"] else 0.0
                    )
                    store.save_paper_bet(
                        settled_at=int(now_s * 1000),
                        slug=meta["slug"],
                        winner=winner.value,
                        pnl=round(pnl, 4),
                        invested=round(pos.total_invested, 4),
                    )

                    add_alert(
                        "resolve",
                        f"RESUELTO {meta['asset']} {meta['duration']}: {winner.value.upper()} "
                        f"→ PnL {'+' if pnl >= 0 else ''}{pnl:.2f}$",
                    )
                    persist_metric({
                        "ts": int(now_s * 1000),
                        "slug": meta["slug"],
                        "winner": winner.value,
                        "opening": meta["opening"],
                        "final_spot": final_spot,
                        "shares_up": pos.shares_up,
                        "shares_down": pos.shares_down,
                        "avg_cost_up": round(pos.avg_cost_up, 4),
                        "avg_cost_down": round(pos.avg_cost_down, 4),
                        "invested": round(pos.total_invested, 4),
                        "pnl": round(pnl, 4),
                        "cum_pnl": round(STATE["paper_pnl"], 4),
                    })

                # drop edge/arb dedupe keys for windows that already closed
                for key in list(alerted_edges):
                    slug = key.rsplit(":", 1)[0]
                    try:
                        window_start = int(slug.rsplit("-", 1)[1])
                    except ValueError:
                        continue
                    dur = 900 if "15m" in slug else 300
                    if now_s > window_start + dur + 60:
                        alerted_edges.discard(key)
                for slug in list(alerted_arbs):
                    try:
                        window_start = int(slug.rsplit("-", 1)[1])
                    except ValueError:
                        continue
                    dur = 900 if "15m" in slug else 300
                    if now_s > window_start + dur + 60:
                        alerted_arbs.discard(slug)

                STATE["pnl_curve"].append([now_ms, round(STATE["paper_pnl"], 2)])
                STATE["pnl_curve"] = STATE["pnl_curve"][-600:]

                # A flat curve is indistinguishable from a broken one unless
                # the reason is on screen. The strategy needs the Up+Down pair
                # under `max_pair_cost`; in practice it sits just above $1, so
                # recording the best pair seen explains the silence.
                STATE["strategy_gate"] = {
                    "max_pair_cost": strategy.max_pair_cost,
                    "best_pair_seen": (
                        round(STATE["best_pair_seen"], 4)
                        if STATE.get("best_pair_seen") is not None else None
                    ),
                    "fills": STATE["stats"]["fills"],
                    "fav_open": len(fav_positions),
                }
                STATE["storage"] = store.storage_status()
                STATE["storage_durable"] = store.storage_is_durable()

                scanner_rows.sort(key=lambda r: r["edge"], reverse=True)
                STATE["markets"] = market_rows
                STATE["scanner"] = scanner_rows
                STATE["advisor"] = build_advisor(scanner_rows, market_rows)
                STATE["positions"] = {
                    cid: {
                        "up": p.shares_up, "down": p.shares_down,
                        "avg_up": round(p.avg_cost_up, 3),
                        "avg_down": round(p.avg_cost_down, 3),
                        "invested": round(p.total_invested, 2),
                        "market": f"{paper_meta.get(cid, {}).get('asset', '?')} "
                                  f"{paper_meta.get(cid, {}).get('duration', '')}",
                    }
                    for cid, p in paper_positions.items()
                }
                STATE["updated_ms"] = now_ms
                STATE["poll_count"] += 1
                STATE["status"] = "live"

            except Exception as exc:
                STATE["errors"] += 1
                STATE["status"] = f"error: {exc}"
                add_alert("warn", f"Error de datos: {str(exc)[:60]}")

            await asyncio.sleep(3)


async def sync_trend_bets(client: httpx.AsyncClient) -> None:
    """Mark open trending positions to market, then settle finished ones.

    Two sources are needed, not one. The trending feed carries live prices
    but only lists events that are still open and still high-volume, so a
    position whose event resolved simply disappears from it — and a
    disappearance is ambiguous: the event may have settled, or it may have
    merely fallen out of the top-volume window. Only a direct lookup can
    tell those apart, so anything missing from the feed gets one.
    """
    open_bets, _ = store.load_trend_bets()
    if not open_bets:
        return

    now_ms = int(time.time() * 1000)
    live: dict[str, dict[str, float]] = {}
    for row in TRENDING_CACHE["rows"]:
        live[str(row["id"])] = {
            o["label"]: o["prob"] for o in (row.get("outcomes") or [])
        }

    for bet in open_bets:
        prices = live.get(str(bet["event_id"]))
        if prices and bet["outcome"] in prices:
            store.mark_trend_bet(bet["id"], prices[bet["outcome"]], now_ms)
            continue

        try:
            resp = await client.get(
                GAMMA_EVENTS,
                params={"id": bet["event_id"]},
                headers=UA_HEADERS,
                timeout=12,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            log.warning("trend.lookup_failed", error=str(exc)[:80])
            continue
        if not payload:
            continue

        final = trend_sim.find_settlement(payload[0], bet["outcome"])
        if final is None:
            continue  # still open, just not trending any more

        pnl = trend_sim.position_pnl(
            bet["entry_price"], bet["stake"] or 1.0, final
        )
        store.close_trend_bet(bet["id"], final, now_ms, "resolved", pnl)
        add_alert(
            "resolve",
            f"DESTACADO resuelto: {str(bet['outcome'])[:40]} → "
            f"{'+' if pnl >= 0 else ''}{pnl:.2f}$",
        )


async def prediction_loop() -> None:
    """Back the favourite on new fixtures, then settle the ones that ended.

    Runs on its own slow cadence: fixtures appear over hours and resolve
    over hours, so tying this to the 3-second market loop would burn
    requests for nothing.
    """
    PREDICTIONS.load()
    log.info("predictions.loaded", total=len(PREDICTIONS.predictions))

    # The first tick after every restart would otherwise fire one push per
    # fixture already sitting above the threshold — every UFC card and
    # every esports BO3 already 80%+ in, all at once. Seeding without
    # pushing means only *new* crossings notify, which is the entire point.
    seeded_favourites = False

    while True:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                matches = await fetch_todays_matches(client)
                added = PREDICTIONS.record(matches, min_prob=PREDICTION_MIN_PROB)
                settled = await PREDICTIONS.resolve(client)
                await sync_trend_bets(client)

            if added or settled:
                log.info("predictions.tick", added=added, settled=settled)
            if settled:
                stats = PREDICTIONS.stats("ia")
                add_alert(
                    "info",
                    f"{settled} predicción(es) resueltas · "
                    f"{stats['wins']}/{stats['resolved']} · {stats['verdict']}",
                )

            # A strong favourite is the live, cross-sport version of the
            # question the whole project exists to answer. Alerted once per
            # fixture — not once per tick — and only while it is still
            # something to act on, not after the market has already called it.
            live_ids: set[str] = set()
            for match in matches:
                live_ids.add(match.event_id)
                if match.decided or match.event_id in alerted_favourites:
                    continue
                favourite = match.favourite()
                if favourite is None:
                    continue
                outcome, prob = favourite
                if prob < FAV_PUSH_THRESHOLD:
                    continue
                alerted_favourites.add(match.event_id)
                if not seeded_favourites:
                    continue
                notify_push(
                    "Favorito fuerte",
                    f"{match.title[:70]} — {outcome.label} al {prob*100:.0f}%",
                    tag=f"fav-{match.event_id}",
                    url=match.url,
                )
            seeded_favourites = True
            # A fixture that dropped out of the feed has ended; forgetting it
            # is what keeps this set from growing forever.
            alerted_favourites.intersection_update(live_ids)
        except Exception as exc:
            log.warning("predictions.loop_failed", error=str(exc)[:120])

        await asyncio.sleep(PREDICTION_INTERVAL)


@app.on_event("startup")
async def startup() -> None:
    # Whether the data is safe is a measurement, not a configuration flag:
    # setting MONEYCLUSH_DB without mounting a volume gives a perfectly
    # writable directory that vanishes at the next deploy.
    status = store.record_boot()
    STATE["storage"] = status
    STATE["storage_durable"] = status["state"] == "durable"

    # Restore the favourite track so a redeploy does not reset the curve.
    STATE["fav_curve"], STATE["fav_stats"] = store.load_fav_history()
    STATE["fav_pnl"] = round(store.fav_pnl(), 4)

    # Same restoration for the arbitrage paper track: every settlement was
    # already durable in paper_metrics.jsonl, but paper_pnl/pnl_curve lived
    # only in STATE and reset to 0 on every restart — the curve looked like
    # a fresh account each redeploy even though the history was intact.
    STATE["pnl_curve"], paper_stats = store.load_paper_history()
    STATE["paper_pnl"] = round(store.paper_pnl(), 4)
    STATE["stats"]["resolved"] = paper_stats["resolved"]
    STATE["stats"]["wins"] = paper_stats["wins"]
    STATE["stats"]["losses"] = paper_stats["losses"]
    STATE["stats"]["win_rate"] = paper_stats["win_rate"]

    # Bets that were still open when the process last stopped. Windows
    # that closed in the meantime settle on the next poll.
    fav_positions.update(store.load_open_favs())
    if fav_positions:
        log.info("fav.restored_open_bets", count=len(fav_positions))

    if status["state"] == "ephemeral":
        add_alert("warn", f"Almacenamiento efímero — {status['reason']}")
        # Silent data loss: nothing else in this project fails loudly
        # enough to justify a push, but losing weeks of the sample the
        # whole project exists to accumulate does.
        notify_push(
            "MoneyClush: almacenamiento efímero",
            status["reason"],
            tag="storage-ephemeral",
        )
    elif status["state"] == "unproven":
        add_alert("info", f"Persistencia sin verificar — {status['reason']}")
    else:
        add_alert("info", f"Almacenamiento persistente — {status['reason']}")

    clob_ws.start()
    asyncio.create_task(poll_loop())
    asyncio.create_task(prediction_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    await clob_ws.stop()


@app.get("/api/state")
async def get_state() -> JSONResponse:
    # Cheap (two file-existence checks) — computed fresh on every fetch so
    # the dashboard reflects a button press immediately, not on the next
    # 3s poll tick.
    STATE["guardrails"] = {
        **kill_switch.status(),
        "limits": asdict(safety_limits),
    }
    return JSONResponse(STATE)


@app.post("/api/emergency-stop")
async def emergency_stop() -> JSONResponse:
    """The panic button. No confirmation step — an emergency stop that
    makes you click twice is a worse emergency stop.
    """
    kill_switch.trigger_stop("manual dashboard stop")
    add_alert("warn", "PARADA DE EMERGENCIA activada — ninguna orden real se enviará")
    log.warning("guardrails.emergency_stop")
    return JSONResponse({**kill_switch.status(), "limits": asdict(safety_limits)})


@app.post("/api/emergency-resume")
async def emergency_resume() -> JSONResponse:
    """Deliberately a separate call from emergency-stop, not a toggle of
    the same button — resuming after a stop should never be one accidental
    click away from the button that caused it.
    """
    kill_switch.clear_stop()
    add_alert("info", "Parada de emergencia liberada")
    log.warning("guardrails.emergency_resume")
    return JSONResponse({**kill_switch.status(), "limits": asdict(safety_limits)})


@app.get("/api/sports")
async def get_sports() -> JSONResponse:
    """Today's fixtures with their market-implied probabilities.

    Fixtures change on the scale of hours, not seconds, so this is fetched
    on request and cached rather than driven from the polling loop.
    """
    now = time.time()
    if SPORTS_CACHE["rows"] and now - SPORTS_CACHE["fetched_at"] < SPORTS_TTL:
        return JSONResponse(
            {"matches": SPORTS_CACHE["rows"], "cached": True,
             "age_s": round(now - SPORTS_CACHE["fetched_at"])}
        )

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            matches = await fetch_todays_matches(client)
        rows = to_rows(matches)
        SPORTS_CACHE.update(rows=rows, fetched_at=now, error=None)
        return JSONResponse({"matches": rows, "cached": False, "age_s": 0})
    except Exception as exc:
        log.warning("sports.fetch_failed", error=str(exc)[:120])
        SPORTS_CACHE["error"] = str(exc)[:120]
        # Stale rows beat an empty screen, as long as the age is shown.
        return JSONResponse(
            {
                "matches": SPORTS_CACHE["rows"],
                "cached": True,
                "age_s": round(now - SPORTS_CACHE["fetched_at"])
                if SPORTS_CACHE["fetched_at"] else None,
                "error": SPORTS_CACHE["error"],
            }
        )


@app.get("/api/news")
async def get_news(category: str = "general") -> JSONResponse:
    if category not in NEWS_CACHE:
        return JSONResponse({"error": "unknown category"}, status_code=400)

    cache = NEWS_CACHE[category]
    now = time.time()
    if cache["rows"] and now - cache["fetched_at"] < NEWS_TTL:
        return JSONResponse(
            {"headlines": cache["rows"], "cached": True,
             "age_s": round(now - cache["fetched_at"])}
        )

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            headlines = await fetch_headlines(client, category)
        rows = news_to_rows(headlines)
        cache.update(rows=rows, fetched_at=now, error=None)
        return JSONResponse({"headlines": rows, "cached": False, "age_s": 0})
    except Exception as exc:
        log.warning("news.fetch_failed", category=category, error=str(exc)[:120])
        cache["error"] = str(exc)[:120]
        return JSONResponse(
            {
                "headlines": cache["rows"],
                "cached": True,
                "age_s": round(now - cache["fetched_at"]) if cache["fetched_at"] else None,
                "error": cache["error"],
            }
        )


@app.get("/api/trending")
async def get_trending() -> JSONResponse:
    """Highest-volume Polymarket events across every category, not just BTC."""
    now = time.time()
    if TRENDING_CACHE["rows"] and now - TRENDING_CACHE["fetched_at"] < TRENDING_TTL:
        return JSONResponse(
            {"events": TRENDING_CACHE["rows"], "cached": True,
             "age_s": round(now - TRENDING_CACHE["fetched_at"])}
        )

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            events = await fetch_trending(client)
        rows = trending_to_rows(events)
        TRENDING_CACHE.update(rows=rows, fetched_at=now, error=None)
        return JSONResponse({"events": rows, "cached": False, "age_s": 0})
    except Exception as exc:
        log.warning("trending.fetch_failed", error=str(exc)[:120])
        TRENDING_CACHE["error"] = str(exc)[:120]
        return JSONResponse(
            {
                "events": TRENDING_CACHE["rows"],
                "cached": True,
                "age_s": round(now - TRENDING_CACHE["fetched_at"])
                if TRENDING_CACHE["fetched_at"] else None,
                "error": TRENDING_CACHE["error"],
            }
        )


# Sigma is calibrated from BTC 1-minute candles for 5-15 minute windows.
# Reusing that number for SOL or XRP prices a different asset's volatility
# with Bitcoin's, and stretching it across a 13-hour daily settlement
# extrapolates far past where it was measured. Both produce confident
# nonsense — a 100c "fair value" against a market quoting 88c. The project
# already settled this question for opening prices: if the input cannot be
# obtained honestly, do not value the market rather than substitute an
# estimate. The same rule applies here.
MODEL_SERIES = {"KXBTCD", "KXBTC15M"}
MODEL_MAX_HORIZON_MIN = 120.0


def annotate_with_model(rows: list[dict]) -> list[dict]:
    """Price Kalshi strikes with our own model where that is defensible.

    A ladder rung is the same object the BTC scanner already prices — a
    digital on where the asset lands — so the Brownian model applies
    unchanged: P(above K) = Phi(distance / sigma_remaining), with distance
    measured from spot to the strike instead of to a window's opening.

    Rungs outside the calibrated asset or horizon get None, which the
    panel renders as a dash. The gap is reported, never acted on: the
    299-window backtest showed this model does not beat Polymarket's
    pricing, and nothing about Kalshi changes that. The column exists to
    flag rungs where the two venues disagree by more than fees explain.
    """
    for row in rows:
        spot = _as_float(STATE.get(KALSHI_SPOT_KEY.get(row["series"], ""), 0.0)) or 0.0
        minutes = row.get("minutes_left")
        priceable = (
            row["series"] in MODEL_SERIES
            and spot > 0
            and minutes is not None
            and 0 < minutes <= MODEL_MAX_HORIZON_MIN
        )

        for market in row["markets"]:
            strike, mid = market.get("strike"), market.get("mid")
            if not priceable or not strike or mid is None:
                market["model"] = None
                market["divergence"] = None
                continue

            distance = (spot - strike) / spot
            model_p = brownian_probability(
                distance, fv_engine.sigma_per_minute, minutes * 60.0
            )
            market["model"] = round(model_p, 4)
            # Positive means the model thinks the rung is too cheap.
            market["divergence"] = round(model_p - mid, 4)
    return rows


@app.get("/api/kalshi")
async def get_kalshi() -> JSONResponse:
    """Kalshi's crypto ladders, their internal arbitrages, and the model gap.

    This is a second venue asking questions Polymarket also asks, but
    settling them against CF Benchmarks rather than Chainlink and charging
    a taker fee where Polymarket charges none. Both differences are shown
    per row so a gap is never mistaken for free money.
    """
    now = time.time()
    if KALSHI_CACHE["rows"] and now - KALSHI_CACHE["fetched_at"] < KALSHI_TTL:
        return JSONResponse(
            {"ladders": KALSHI_CACHE["rows"], "cached": True,
             "age_s": round(now - KALSHI_CACHE["fetched_at"]),
             "error": KALSHI_CACHE["error"]}
        )

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            groups = await fetch_crypto_ladders(client)
        rows = annotate_with_model(kalshi_to_rows(groups))
        KALSHI_CACHE.update(rows=rows, fetched_at=now, error=None)

        total_arbs = sum(r["arb_count"] for r in rows)
        if total_arbs:
            best = max((r["best_arb"] or 0) for r in rows)
            add_alert("info", f"KALSHI {total_arbs} arb(s) de escalera, mejor {best*100:.1f}¢")

        return JSONResponse({"ladders": rows, "cached": False, "age_s": 0})
    except Exception as exc:
        log.warning("kalshi.fetch_failed", error=str(exc)[:120])
        KALSHI_CACHE["error"] = str(exc)[:120]
        return JSONResponse(
            {
                "ladders": KALSHI_CACHE["rows"],
                "cached": True,
                "age_s": round(now - KALSHI_CACHE["fetched_at"])
                if KALSHI_CACHE["fetched_at"] else None,
                "error": KALSHI_CACHE["error"],
            }
        )


# The category scanners are stateless, so one instance each is enough.
CATEGORY_STRATEGIES = (SportsBasketArb(), LadderArb(), FavouriteFade())

# Signals the user said they acted on. Kept on disk because the only
# question that matters about a recommendation tool — did following it
# make money — cannot be answered from a log that resets every restart.
TAKEN_FILE = DATA_DIR / "signals_taken.jsonl"
OPPS_TTL = 60.0
OPPS_CACHE: dict = {"rows": [], "fetched_at": 0.0}


async def scan_opportunities() -> list[dict]:
    """Category-strategy rows, cached so the trading desk can poll fast.

    The desk refreshes on a few seconds; the sports and Kalshi fetches
    behind it take seconds each. Without a cache between them, every tick
    of the trading tab would re-hit both venues.
    """
    now = time.time()
    if OPPS_CACHE["rows"] and now - OPPS_CACHE["fetched_at"] < OPPS_TTL:
        return OPPS_CACHE["rows"]

    async with httpx.AsyncClient(timeout=30) as client:
        ladders, matches = await asyncio.gather(
            fetch_crypto_ladders(client),
            fetch_todays_matches(client),
        )

    source = {"sports_basket_arb": matches, "kalshi_ladder_arb": ladders,
              "favourite_fade": matches}

    rows = []
    for strategy in CATEGORY_STRATEGIES:
        for opp in strategy.scan(source[strategy.name])[:12]:
            rows.append(
                {
                    "strategy": opp.strategy,
                    "venue": opp.venue,
                    "category": opp.category,
                    "kind": opp.kind.value,
                    "actionable": opp.actionable,
                    "label": opp.label,
                    "edge": round(opp.edge, 4),
                    "rationale": opp.rationale,
                    "evidence": opp.evidence,
                    "url": opp.url,
                }
            )

    OPPS_CACHE.update(rows=rows, fetched_at=now)
    return rows


@app.get("/api/signals")
async def get_signals(asset: str = "todos", timeframe: str = "todos") -> JSONResponse:
    """What to do right now, ranked, with each signal's evidence attached.

    Decision support only — nothing here places an order. `asset` and
    `timeframe` mirror how the tab is actually used: pick the market you
    are watching and the horizon you will hold, then read one list.
    """
    try:
        opportunities = await scan_opportunities()
    except Exception as exc:
        log.warning("signals.opps_failed", error=str(exc)[:120])
        opportunities = OPPS_CACHE["rows"]

    desk = build_desk(
        arb_events=STATE.get("arb_events", []),
        markets=STATE.get("markets", []),
        opportunities=opportunities,
        scanner=STATE.get("scanner", []),
        asset=asset,
        timeframe=timeframe,
    )

    return JSONResponse(
        {
            "signals": desk.rows(),
            "actionable": desk.actionable,
            "sampling": desk.sampling,
            "informational": desk.informational,
            "scanned": desk.scanned,
            "no_trade_reason": desk.no_trade_reason,
            "assets": sorted({m.get("asset", "") for m in STATE.get("markets", [])} - {""}),
            "timeframes": sorted({m.get("duration", "") for m in STATE.get("markets", [])} - {""}),
        }
    )


@app.post("/api/signals/taken")
async def post_signal_taken(payload: dict = Body(...)) -> JSONResponse:
    """Record that the user executed a signal by hand.

    The desk cannot see the user's broker, so the only way to learn
    whether its advice was worth following is for the user to say they
    followed it. Each entry is appended, never rewritten, so the record
    cannot be tidied up after the fact.
    """
    entry = {
        "ts": int(time.time() * 1000),
        "source": str(payload.get("source", ""))[:60],
        "asset": str(payload.get("asset", ""))[:20],
        "action": str(payload.get("action", ""))[:200],
        "confidence": str(payload.get("confidence", ""))[:20],
        "edge": _as_float(payload.get("edge")) or 0.0,
        "stake": _as_float(payload.get("stake")) or 0.0,
    }

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with TAKEN_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        log.warning("signals.taken_write_failed", error=str(exc)[:120])
        return JSONResponse({"error": "no se pudo guardar"}, status_code=500)

    add_alert("info", f"SEÑAL TOMADA · {entry['action'][:60]}")
    return JSONResponse({"ok": True, "entry": entry})


@app.get("/api/signals/taken")
async def get_signals_taken() -> JSONResponse:
    """The log of signals the user said they acted on, newest first."""
    rows: list[dict] = []
    try:
        if TAKEN_FILE.exists():
            with TAKEN_FILE.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError as exc:
        log.warning("signals.taken_read_failed", error=str(exc)[:120])

    rows.reverse()
    by_tier: dict[str, int] = {}
    for row in rows:
        tier = row.get("confidence", "?")
        by_tier[tier] = by_tier.get(tier, 0) + 1

    return JSONResponse({"taken": rows[:80], "total": len(rows), "by_tier": by_tier})


@app.get("/api/opportunities")
async def get_opportunities() -> JSONResponse:
    """Every category strategy run over the current snapshot.

    Opportunities are returned with their `kind` intact so the panel can
    keep structural edges (arithmetic, tradeable) visually separate from
    statistical ones (a measured but unconfirmed bias). Collapsing the two
    into a single "edge" column is exactly how an unproven hypothesis
    turns into a live position by accident.
    """
    try:
        rows = list(await scan_opportunities())
    except Exception as exc:
        log.warning("opportunities.fetch_failed", error=str(exc)[:120])
        return JSONResponse({"opportunities": [], "error": str(exc)[:120]})

    # Structural first: those are the ones worth acting on.
    rows.sort(key=lambda r: (not r["actionable"], -r["edge"]))
    return JSONResponse({"opportunities": rows, "count": len(rows)})


@app.get("/api/predictions")
async def get_predictions() -> JSONResponse:
    """Both ledgers side by side: the automatic picks and the user's own."""
    return JSONResponse(
        {
            "ia": {
                "stats": PREDICTIONS.stats("ia"),
                "recent": PREDICTIONS.recent(60, "ia"),
            },
            "manual": {
                "stats": PREDICTIONS.stats("manual"),
                "recent": PREDICTIONS.recent(60, "manual"),
            },
            "min_prob": PREDICTION_MIN_PROB,
            "stake": PREDICTION_STAKE,
        }
    )


@app.post("/api/predictions/manual")
async def post_manual_prediction(payload: dict = Body(...)) -> JSONResponse:
    """Record a pick the user made by hand, at the price showing right now."""
    required = ("event_id", "title", "pick", "pick_ask")
    missing = [f for f in required if payload.get(f) in (None, "")]
    if missing:
        return JSONResponse(
            {"error": f"faltan campos: {', '.join(missing)}"}, status_code=400
        )

    event_id = str(payload["event_id"])
    if PREDICTIONS.has_manual(event_id):
        return JSONResponse(
            {"error": "ya hay una predicción tuya en este evento"}, status_code=409
        )

    try:
        ask = float(payload["pick_ask"])
        prob = float(payload.get("pick_prob") or ask)
    except (TypeError, ValueError):
        return JSONResponse({"error": "precio inválido"}, status_code=400)

    # Both extremes are already-decided markets: a side at 1 has won, and one
    # at a fraction of a cent has lost. Neither is a prediction worth logging.
    if not MIN_PICK_PRICE <= ask <= MAX_PICK_PRICE:
        return JSONResponse(
            {
                "error": (
                    f"ese resultado cotiza a {ask * 100:.1f}¢ — fuera del rango "
                    f"apostable ({MIN_PICK_PRICE * 100:.0f}¢–{MAX_PICK_PRICE * 100:.0f}¢). "
                    "Está prácticamente resuelto."
                )
            },
            status_code=400,
        )

    prediction = PREDICTIONS.record_manual(
        event_id=event_id,
        title=str(payload["title"]),
        url=str(payload.get("url") or ""),
        pick=str(payload["pick"]),
        pick_prob=prob,
        pick_ask=ask,
        league=str(payload.get("league") or ""),
        discipline=str(payload.get("discipline") or ""),
        stake=PREDICTION_STAKE,
        volume_24h=_as_float(payload.get("volume_24h")) or 0.0,
    )
    add_alert(
        "info",
        f"Predicción manual: {prediction.pick} @ {ask * 100:.0f}¢ "
        f"· paga ${prediction.potential_payout:.2f}",
    )
    return JSONResponse({"ok": True, "prediction": asdict(prediction)})


@app.get("/api/trend")
async def get_trend_sim() -> JSONResponse:
    """Open trending positions with their running mark, plus closed history."""
    open_bets, closed = store.load_trend_bets()
    for bet in open_bets:
        mark = bet.get("last_price")
        if mark is None:
            mark = bet["entry_price"]
        bet["unrealised"] = round(
            trend_sim.position_pnl(bet["entry_price"], bet["stake"] or 1.0, mark), 4
        )
    return JSONResponse(
        {
            "open": open_bets,
            "closed": closed,
            "stats": trend_sim.summarise(closed, open_bets),
        }
    )


@app.post("/api/trend/open")
async def post_trend_open(payload: dict = Body(...)) -> JSONResponse:
    """Enter a trending outcome, but only inside the strong-favourite band."""
    required = ("event_id", "outcome", "price")
    missing = [f for f in required if payload.get(f) in (None, "")]
    if missing:
        return JSONResponse(
            {"error": f"faltan campos: {', '.join(missing)}"}, status_code=400
        )

    try:
        price = float(payload["price"])
    except (TypeError, ValueError):
        return JSONResponse({"error": "precio inválido"}, status_code=400)

    rejection = trend_sim.entry_rejection(price)
    if rejection:
        return JSONResponse({"error": rejection}, status_code=400)

    event_id = str(payload["event_id"])
    outcome = str(payload["outcome"])
    if store.has_open_trend_bet(event_id, outcome):
        return JSONResponse(
            {"error": "ya tienes una posición abierta en ese resultado"},
            status_code=409,
        )

    now_ms = int(time.time() * 1000)
    bet_id = store.open_trend_bet(
        event_id=event_id,
        title=str(payload.get("title") or ""),
        url=str(payload.get("url") or ""),
        category=str(payload.get("category") or ""),
        outcome=outcome,
        entry_price=price,
        entry_at=now_ms,
        stake=PREDICTION_STAKE,
    )
    add_alert(
        "info",
        f"Destacados: entrada {outcome[:32]} @ {price * 100:.0f}¢",
    )
    return JSONResponse({"ok": True, "id": bet_id})


@app.post("/api/trend/close")
async def post_trend_close(payload: dict = Body(...)) -> JSONResponse:
    """Sell an open position at its current mark, before resolution."""
    try:
        bet_id = int(payload.get("id"))
    except (TypeError, ValueError):
        return JSONResponse({"error": "id inválido"}, status_code=400)

    open_bets, _ = store.load_trend_bets()
    bet = next((b for b in open_bets if b["id"] == bet_id), None)
    if bet is None:
        return JSONResponse({"error": "posición no encontrada"}, status_code=404)

    mark = bet.get("last_price")
    if mark is None:
        mark = bet["entry_price"]
    pnl = trend_sim.position_pnl(bet["entry_price"], bet["stake"] or 1.0, mark)
    store.close_trend_bet(bet_id, mark, int(time.time() * 1000), "manual", pnl)
    add_alert(
        "info",
        f"Destacados: salida {str(bet['outcome'])[:32]} @ {mark * 100:.0f}¢ "
        f"· {'+' if pnl >= 0 else ''}{pnl:.2f}$",
    )
    return JSONResponse({"ok": True, "pnl": round(pnl, 4)})


@app.get("/api/standings")
async def get_standings() -> JSONResponse:
    now = time.time()
    if STANDINGS_CACHE["rows"] and now - STANDINGS_CACHE["fetched_at"] < STANDINGS_TTL:
        return JSONResponse({"leagues": STANDINGS_CACHE["rows"], "cached": True})

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            standings = await fetch_standings(client)
        rows = standings_to_rows(standings)
        STANDINGS_CACHE.update(rows=rows, fetched_at=now, error=None)
        return JSONResponse({"leagues": rows, "cached": False})
    except Exception as exc:
        log.warning("standings.fetch_failed", error=str(exc)[:120])
        return JSONResponse(
            {"leagues": STANDINGS_CACHE["rows"], "cached": True, "error": str(exc)[:120]}
        )


@app.get("/api/push/vapid-public-key")
async def push_vapid_key() -> JSONResponse:
    return JSONResponse({"key": push.public_key_b64url()})


@app.post("/api/push/subscribe")
async def push_subscribe(payload: dict = Body(...)) -> JSONResponse:
    """Save a browser's push subscription, as handed back by `PushManager`."""
    endpoint = payload.get("endpoint")
    keys = payload.get("keys") or {}
    p256dh, auth = keys.get("p256dh"), keys.get("auth")
    if not endpoint or not p256dh or not auth:
        return JSONResponse({"error": "suscripción incompleta"}, status_code=400)
    store.save_push_subscription(str(endpoint), str(p256dh), str(auth))
    return JSONResponse({"ok": True})


@app.post("/api/push/unsubscribe")
async def push_unsubscribe(payload: dict = Body(...)) -> JSONResponse:
    endpoint = payload.get("endpoint")
    if endpoint:
        store.delete_push_subscription(str(endpoint))
    return JSONResponse({"ok": True})


@app.get("/sw.js")
async def service_worker() -> FileResponse:
    # Served from the root, not /static/, so its default scope is the
    # whole origin rather than just the static directory — matters if
    # anything ever needs `clients.matchAll()` scoped to the app.
    return FileResponse(STATIC_DIR / "sw.js", media_type="application/javascript")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8642, log_level="warning")
