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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from moneyclush.data.clob_websocket import ClobWebSocket
from moneyclush.data.consensus_price import ConsensusFeed
from moneyclush.data.market_discovery import UA_HEADERS, discover_active_markets
from moneyclush.data.sports import fetch_todays_matches, to_rows
from moneyclush.data.opening_prices import OpeningPriceCache
from moneyclush.data.models import (
    MarketInfo,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    OutcomeSide,
    Position,
)
from moneyclush.pricing.fair_value import FairValueEngine
from moneyclush.signals.order_book import combined_pair_cost, order_book_imbalance
from moneyclush.strategies.temporal_arbitrage import TemporalArbitrageStrategy

CLOB_URL = "https://clob.polymarket.com"
OKX_URL = "https://www.okx.com/api/v5/market/ticker"
COINBASE_URL = "https://api.coinbase.com/v2/prices"

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
}

fv_engine = FairValueEngine()
strategy = TemporalArbitrageStrategy(block_size=25)

paper_positions: dict[str, Position] = {}
# condition_id -> {slug, asset, duration, opening, window_end}
paper_meta: dict[str, dict] = {}
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
SPORTS_TTL = 90.0
SPORTS_CACHE: dict = {"rows": [], "fetched_at": 0.0, "error": None}
alerted_edges: set[str] = set()            # dedupe: slug+side alerted once per window


def add_alert(level: str, text: str) -> None:
    """level: edge | fill | resolve | warn | info"""
    STATE["alerts"].insert(0, {
        "ts": int(time.time() * 1000),
        "level": level,
        "text": text,
    })
    STATE["alerts"] = STATE["alerts"][:50]


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


async def fetch_spot_prices(client: httpx.AsyncClient) -> dict[str, float]:
    """Spot prices from OKX (primary) with Coinbase fallback.

    Binance is geo-blocked (HTTP 451) from this location.
    """
    out = {}
    for inst, cb_pair, key in [
        ("BTC-USDT", "BTC-USD", "btc_price"),
        ("ETH-USDT", "ETH-USD", "eth_price"),
        ("SOL-USDT", "SOL-USD", "sol_price"),
        ("XRP-USDT", "XRP-USD", "xrp_price"),
    ]:
        try:
            r = await client.get(OKX_URL, params={"instId": inst})
            out[key] = float(r.json()["data"][0]["last"])
        except Exception:
            try:
                r = await client.get(f"{COINBASE_URL}/{cb_pair}/spot")
                out[key] = float(r.json()["data"]["amount"])
            except Exception:
                out[key] = STATE.get(key, 0.0)
    return out


def build_advisor(scanner_rows: list[dict], market_rows: list[dict]) -> list[dict]:
    """Rule-based contextual tips about the current market state."""
    tips: list[dict] = []
    stats = STATE["stats"]

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

    if stats["resolved"] >= 5:
        wr = stats["win_rate"] * 100
        if wr < 50:
            tips.append({
                "icon": "%",
                "kind": "warn",
                "text": f"Win rate {wr:.0f}% en {stats['resolved']} resoluciones — revisar "
                        f"si el fair value está mal calibrado antes de subir tamaño.",
            })
        else:
            tips.append({
                "icon": "%",
                "kind": "ok",
                "text": f"Win rate {wr:.0f}% en {stats['resolved']} resoluciones. "
                        f"Aún poca muestra: esperar 30+ antes de sacar conclusiones.",
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
        while True:
            try:
                if time.time() - last_calibration > 900:
                    await calibrate_from_candles(client)
                    last_calibration = time.time()

                spots = await fetch_spot_prices(client)
                STATE.update(spots)
                now_ms = int(time.time() * 1000)

                # Dispersion across venues tells us how much to trust the tick.
                await consensus_feed.refresh(client, ["BTC"])
                btc_dispersion = consensus_feed.dispersion_bps("BTC")
                STATE["dispersion_bps"] = (
                    round(btc_dispersion, 2) if btc_dispersion != float("inf") else None
                )
                data_trusted = btc_dispersion <= MAX_DISPERSION_BPS

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
                    }

                    if mk.asset == "BTC" and opening is not None:
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
                        except Exception:
                            pass

                    market_rows.append(row)

                # -------- resolve expired paper positions (per position, precise)
                now_s = time.time()
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

                # drop edge dedupe keys for windows that already closed
                for key in list(alerted_edges):
                    slug = key.rsplit(":", 1)[0]
                    try:
                        window_start = int(slug.rsplit("-", 1)[1])
                    except ValueError:
                        continue
                    dur = 900 if "15m" in slug else 300
                    if now_s > window_start + dur + 60:
                        alerted_edges.discard(key)

                STATE["pnl_curve"].append([now_ms, round(STATE["paper_pnl"], 2)])
                STATE["pnl_curve"] = STATE["pnl_curve"][-600:]

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


@app.on_event("startup")
async def startup() -> None:
    clob_ws.start()
    asyncio.create_task(poll_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    await clob_ws.stop()


@app.get("/api/state")
async def get_state() -> JSONResponse:
    return JSONResponse(STATE)


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


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8642, log_level="warning")
