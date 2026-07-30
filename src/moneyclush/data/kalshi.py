"""Kalshi markets — a second venue for the same questions.

Kalshi is a CFTC-regulated exchange, and that regulatory difference shows
up in three places that matter for trading rather than for compliance:

    settlement   Kalshi crypto settles on CF Benchmarks (the BRTI real-time
                 index). Polymarket settles BTC on Chainlink. Two venues
                 asking the same question resolve against *different*
                 numbers, so a price gap between them is not automatically
                 an arbitrage — part of it is index basis.

    fees         Kalshi charges takers `0.07 x P x (1-P)` per contract,
                 which peaks at 1.75c at a 50c price. Polymarket charges
                 nothing. A 2c gap that is free money on Polymarket is a
                 loss here once both legs are paid for.

    shape        Kalshi lists BTC as a *ladder* of strikes on one hour
                 ("above $63,999.99", "above $64,249.99", ...) rather than
                 a single Up/Down binary. A ladder has an internal
                 consistency requirement that a lone binary does not, and
                 that requirement is checkable — see `ladder_arbitrages`.

The public read endpoints need no authentication, so market data is
available without an account. Placing orders would need one; nothing here
places orders.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import structlog

log = structlog.get_logger()

API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Taker fee coefficient. Kalshi rounds the per-contract fee up to the next
# cent, so a "cheap" 3c edge can still be eaten alive by two legs of it.
FEE_COEFFICIENT = 0.07

# Series worth watching. Kalshi lists hundreds of crypto series, most of
# them annual one-offs that never move; these are the short-dated ones
# whose questions Polymarket also asks.
CRYPTO_SERIES = {
    "KXBTCD": "BTC horario (ladder)",
    "KXBTC15M": "BTC 15 min arriba/abajo",
    "KXETHD": "ETH horario (ladder)",
    "KXETH15M": "ETH 15 min arriba/abajo",
    "KXSOLD": "SOL horario",
    "KXXRPD": "XRP horario",
}

# A strike ticker looks like KXBTCD-26JUL3017-T63999.99. The trailing
# component carries the threshold; `T` means "above", `B` a between-band.
_STRIKE_RE = re.compile(r"-([TB])(-?[\d.]+)$")


def taker_fee(price: float, contracts: float = 1.0) -> float:
    """Kalshi's taker fee in dollars, rounded up to the cent as they do.

    The fee is largest exactly where prediction markets are most active —
    around 50c — which is why edges near the money need to be much bigger
    here than on a zero-fee venue to survive.
    """
    if not 0.0 < price < 1.0:
        return 0.0
    raw = FEE_COEFFICIENT * price * (1.0 - price) * contracts
    return math.ceil(raw * 100.0) / 100.0


def round_trip_cost(price: float) -> float:
    """Fee to take both sides of one contract at `price`.

    Any two-leg structure (a ladder spread, a cross-venue pair) pays this,
    so it is the hurdle every edge below has to clear.
    """
    return taker_fee(price) + taker_fee(1.0 - price)


def _as_float(value) -> float | None:
    """Kalshi returns prices as decimal strings like "0.4500"."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class KalshiMarket:
    """One binary contract on Kalshi.

    Kalshi quotes a single book with YES bids and NO bids. The YES ask is
    not published directly — it is `1 - no_bid`, because selling NO and
    buying YES are the same trade. `yes_ask` below already does that
    conversion, so the rest of the code can treat it like any other book.
    """

    ticker: str
    event_ticker: str
    series: str
    title: str
    subtitle: str
    yes_bid: float | None
    yes_ask: float | None
    volume: float
    open_interest: float
    liquidity: float
    close_time: datetime | None
    status: str
    strike: float | None = None

    @property
    def mid(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    @property
    def tradeable(self) -> bool:
        """Whether both sides carry a genuine quote.

        Kalshi lists every strike in the ladder whether or not anyone is
        making a market on it, so the far wings come back as 0 bid / 100
        ask. That is not a market at a 50c mid — it is the absence of one,
        and treating its midpoint as a price invents a number nobody is
        willing to trade at. Requiring a real quote on both sides is what
        keeps those rungs out of the distribution and out of the scan.
        """
        if self.yes_bid is None or self.yes_ask is None:
            return False
        if self.yes_bid <= 0.0 or self.yes_ask >= 1.0:
            return False
        return self.yes_ask > self.yes_bid

    @property
    def minutes_left(self) -> float | None:
        if not self.close_time:
            return None
        return (self.close_time - datetime.now(timezone.utc)).total_seconds() / 60

    @property
    def url(self) -> str:
        return f"https://kalshi.com/markets/{self.series.lower()}"


@dataclass
class LadderArb:
    """A monotonicity violation between two strikes of the same ladder.

    "BTC above $63,000" cannot be less likely than "BTC above $64,000" —
    every world where the higher strike pays also pays the lower one. So
    buying the low strike and selling the high strike gives a payoff that
    is 1 inside the band and 0 outside, never negative. When the low
    strike's ask is *below* the high strike's bid, that non-negative
    payoff comes with cash up front, which is free money.

    This structure has no analogue in a plain Up/Down binary — it exists
    only because Kalshi lists many strikes on the same underlying and the
    same hour, and lets their books drift apart.
    """

    low: KalshiMarket
    high: KalshiMarket
    credit: float          # cash received per contract before fees
    fees: float
    net: float             # profit per contract after fees

    @property
    def label(self) -> str:
        return f"{self.low.subtitle or self.low.ticker} vs {self.high.subtitle or self.high.ticker}"


@dataclass
class LadderGroup:
    """Every strike listed on one event — one asset, one expiry."""

    event_ticker: str
    series: str
    title: str
    close_time: datetime | None
    markets: list[KalshiMarket] = field(default_factory=list)

    @property
    def total_volume(self) -> float:
        return sum(m.volume for m in self.markets)

    @property
    def minutes_left(self) -> float | None:
        if not self.close_time:
            return None
        return (self.close_time - datetime.now(timezone.utc)).total_seconds() / 60

    def implied_distribution(self) -> list[tuple[KalshiMarket, float]]:
        """Probability the market assigns to each band between strikes.

        Consecutive "above X" digitals differ by exactly the probability of
        landing between them, so differencing the ladder recovers the
        market's whole view of where BTC ends up — something a single
        Up/Down contract cannot express. A negative entry here means the
        ladder is internally inconsistent.
        """
        rungs = sorted(
            (m for m in self.markets if m.tradeable and m.strike is not None),
            key=lambda m: m.strike,
        )
        out: list[tuple[KalshiMarket, float]] = []
        for i, m in enumerate(rungs):
            nxt = rungs[i + 1] if i + 1 < len(rungs) else None
            here = m.mid or 0.0
            above = (nxt.mid or 0.0) if nxt else 0.0
            out.append((m, here - above))
        return out

    def distribution_bands(self) -> list[dict]:
        """The implied distribution as plottable bands, including the tails.

        Two bands exist that no single rung represents: everything below
        the lowest strike, and everything above the highest. Leaving them
        out draws a histogram that does not sum to 1 and quietly hides
        where most of the probability often sits — on a ladder whose
        strikes all sit under spot, the entire upper tail is the story.
        """
        rungs = sorted(
            (m for m in self.markets if m.tradeable and m.strike is not None),
            key=lambda m: m.strike,
        )
        if not rungs:
            return []

        bands: list[dict] = [
            {
                "label": f"< {rungs[0].strike:,.0f}",
                "low": None,
                "high": rungs[0].strike,
                "prob": max(0.0, 1.0 - (rungs[0].mid or 0.0)),
            }
        ]

        for i, m in enumerate(rungs[:-1]):
            nxt = rungs[i + 1]
            bands.append(
                {
                    "label": f"{m.strike:,.0f}–{nxt.strike:,.0f}",
                    "low": m.strike,
                    "high": nxt.strike,
                    "prob": (m.mid or 0.0) - (nxt.mid or 0.0),
                }
            )

        top = rungs[-1]
        bands.append(
            {
                "label": f"> {top.strike:,.0f}",
                "low": top.strike,
                "high": None,
                "prob": max(0.0, top.mid or 0.0),
            }
        )
        return bands


def ladder_arbitrages(group: LadderGroup, min_net: float = 0.005) -> list[LadderArb]:
    """Strike pairs whose quotes contradict each other, net of fees.

    Only adjacent-or-wider pairs in the *wrong* order count: the low
    strike must be buyable for less than the high strike is sellable for.
    Fees are charged on both legs at their own prices, because Kalshi
    prices the fee off each contract separately rather than off the net.
    """
    rungs = sorted(
        (m for m in group.markets if m.tradeable and m.strike is not None),
        key=lambda m: m.strike,
    )

    found: list[LadderArb] = []
    for i, low in enumerate(rungs):
        for high in rungs[i + 1:]:
            # Long the low strike, short the high strike.
            credit = high.yes_bid - low.yes_ask
            if credit <= 0:
                continue
            fees = taker_fee(low.yes_ask) + taker_fee(high.yes_bid)
            net = credit - fees
            if net >= min_net:
                found.append(LadderArb(low=low, high=high, credit=credit,
                                       fees=fees, net=net))

    found.sort(key=lambda a: -a.net)
    return found


def _parse_strike(ticker: str) -> float | None:
    m = _STRIKE_RE.search(ticker)
    if not m:
        return None
    return _as_float(m.group(2))


def _build_market(raw: dict) -> KalshiMarket | None:
    ticker = raw.get("ticker")
    if not ticker:
        return None

    yes_bid = _as_float(raw.get("yes_bid_dollars"))
    yes_ask = _as_float(raw.get("yes_ask_dollars"))

    # Kalshi publishes NO bids, not YES asks. When the YES ask is missing,
    # reconstruct it from the NO side rather than dropping the market.
    if yes_ask is None:
        no_bid = _as_float(raw.get("no_bid_dollars"))
        yes_ask = (1.0 - no_bid) if no_bid is not None else None

    return KalshiMarket(
        ticker=ticker,
        event_ticker=raw.get("event_ticker", ""),
        series=ticker.split("-")[0],
        title=raw.get("title", ""),
        subtitle=raw.get("yes_sub_title") or raw.get("no_sub_title") or "",
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        volume=_as_float(raw.get("volume_fp")) or 0.0,
        open_interest=_as_float(raw.get("open_interest_fp")) or 0.0,
        liquidity=_as_float(raw.get("liquidity_dollars")) or 0.0,
        close_time=_parse_dt(raw.get("close_time")),
        status=raw.get("status", ""),
        strike=_parse_strike(ticker),
    )


async def _fetch_series(
    client: httpx.AsyncClient, series_ticker: str, status: str = "open"
) -> list[KalshiMarket]:
    resp = await client.get(
        f"{API_BASE}/markets",
        params={"series_ticker": series_ticker, "status": status, "limit": 200},
        headers=UA_HEADERS,
    )
    resp.raise_for_status()
    built = (_build_market(m) for m in resp.json().get("markets", []))
    return [m for m in built if m is not None]


async def fetch_crypto_ladders(
    client: httpx.AsyncClient,
    series: dict[str, str] | None = None,
) -> list[LadderGroup]:
    """Open crypto markets on Kalshi, grouped into ladders by expiry.

    Series are fetched concurrently; a single failing series is skipped
    rather than taking the whole panel down with it, because Kalshi
    de-lists series without notice and a 404 on one is routine.
    """
    wanted = series or CRYPTO_SERIES

    results = await asyncio.gather(
        *(_fetch_series(client, s) for s in wanted),
        return_exceptions=True,
    )

    groups: dict[str, LadderGroup] = {}
    for series_ticker, result in zip(wanted, results):
        if isinstance(result, BaseException):
            log.warning("kalshi.series_failed", series=series_ticker,
                        error=str(result)[:100])
            continue

        for market in result:
            key = market.event_ticker or market.ticker
            group = groups.get(key)
            if group is None:
                group = LadderGroup(
                    event_ticker=key,
                    series=series_ticker,
                    title=market.title,
                    close_time=market.close_time,
                )
                groups[key] = group
            group.markets.append(market)

    live = [g for g in groups.values() if any(m.tradeable for m in g.markets)]
    live.sort(key=lambda g: (g.minutes_left if g.minutes_left is not None else 1e9))
    return live


async def fetch_orderbook(
    client: httpx.AsyncClient, ticker: str, depth: int = 5
) -> dict:
    """Raw depth for one contract.

    Both sides come back as *bids* — `yes_dollars` are bids to buy YES and
    `no_dollars` are bids to buy NO. There is no ask side to read; an ask
    is just the mirror of the opposite bid.
    """
    resp = await client.get(
        f"{API_BASE}/markets/{ticker}/orderbook",
        params={"depth": depth},
        headers=UA_HEADERS,
    )
    resp.raise_for_status()
    return resp.json().get("orderbook_fp", {})


def to_rows(groups: list[LadderGroup], max_groups: int = 12) -> list[dict]:
    """Flatten ladders for the dashboard, richest strike first."""
    rows = []
    for g in groups[:max_groups]:
        arbs = ladder_arbitrages(g)
        tradeable = [m for m in g.markets if m.tradeable]
        tradeable.sort(key=lambda m: -m.volume)

        rows.append(
            {
                "event_ticker": g.event_ticker,
                "series": g.series,
                "series_label": CRYPTO_SERIES.get(g.series, g.series),
                "title": g.title,
                "close_time": g.close_time.isoformat() if g.close_time else None,
                "minutes_left": (
                    round(g.minutes_left) if g.minutes_left is not None else None
                ),
                "volume": round(g.total_volume),
                "strikes": len(tradeable),
                "arb_count": len(arbs),
                "best_arb": round(arbs[0].net, 4) if arbs else None,
                "distribution": [
                    {"label": b["label"], "prob": round(b["prob"], 4)}
                    for b in g.distribution_bands()
                ],
                "arbs": [
                    {
                        "label": a.label,
                        "low": a.low.subtitle or a.low.ticker,
                        "high": a.high.subtitle or a.high.ticker,
                        "credit": round(a.credit, 4),
                        "fees": round(a.fees, 4),
                        "net": round(a.net, 4),
                    }
                    for a in arbs[:4]
                ],
                "markets": [
                    {
                        "ticker": m.ticker,
                        "label": m.subtitle or m.ticker,
                        "strike": m.strike,
                        "bid": m.yes_bid,
                        "ask": m.yes_ask,
                        "mid": round(m.mid, 4) if m.mid is not None else None,
                        "spread": round(m.spread, 4) if m.spread is not None else None,
                        "fee_rt": round(round_trip_cost(m.mid), 4) if m.mid else None,
                        "volume": round(m.volume),
                        "open_interest": round(m.open_interest),
                        "url": m.url,
                    }
                    for m in tradeable[:14]
                ],
            }
        )
    return rows
