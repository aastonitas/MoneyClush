"""The trading desk: one ranked list of what to do right now.

This is decision support, not an autopilot. Nothing here places an order.
Every signal carries the exact instruction a human would type into
Polymarket or Kalshi, plus how long it has before the window closes, so
the work left to the person is execution rather than deliberation.

Two design choices are worth stating up front, because they are what
separate this from the signal bots it resembles.

**Most of the time it says nothing.** A tool that always has a
recommendation is not reading the market, it is generating text. The
honest output of a scan over efficient markets is usually an empty list,
and `no_trade_reason` exists to say so in words rather than leaving a
blank panel that looks broken.

**Confidence is tied to evidence, not to a feeling.** Each signal
declares what has to be true for it to pay:

    ALTA     Arithmetic. A basket that costs less than its guaranteed
             payout, or a strike ladder contradicting itself. Pays
             regardless of what the underlying does.

    BAJA     A measured but unconfirmed bias -- the favourite overpricing
             found at p=0.031 over 299 windows, which needs roughly 1000
             to settle. Tracked, sized small, never presented as a call.

    NINGUNA  The model disagrees with the market. This project already
             measured that disagreement over 299 resolved windows: model
             Brier 0.202 against the market's 0.196. The model is worse.
             These are surfaced as context and explicitly marked
             not-to-trade, because dressing them as signals would be
             rebuilding the exact thing that backtest disproved.

There is deliberately no MEDIA tier. The evidence is either arithmetic or
it is a hypothesis; a middle label would only be a place to hide things
that failed to qualify as either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class Confidence(str, Enum):
    ALTA = "alta"
    BAJA = "baja"
    NINGUNA = "ninguna"


# How much to put behind each tier. Structural edges are capped by what
# the book can absorb rather than by conviction; the statistical tier is
# deliberately one dollar, because its purpose is to accumulate a sample,
# not to make money before the sample exists.
STAKE_BY_CONFIDENCE = {
    Confidence.ALTA: 25.0,
    Confidence.BAJA: 1.0,
    Confidence.NINGUNA: 0.0,
}


@dataclass
class Signal:
    """One concrete thing a person could do, with its expiry and evidence."""

    source: str
    venue: str
    asset: str
    timeframe: str
    action: str                  # the instruction, in plain Spanish
    confidence: Confidence
    edge: float                  # profit per $1 committed, after costs
    stake: float
    rationale: str
    evidence: str
    url: str = ""
    expires_ms: int = 0          # epoch ms; 0 when the window is open-ended
    legs: list[dict] = field(default_factory=list)

    @property
    def seconds_left(self) -> float | None:
        if not self.expires_ms:
            return None
        return max(0.0, (self.expires_ms - time.time() * 1000) / 1000.0)

    @property
    def tradeable(self) -> bool:
        return self.confidence is not Confidence.NINGUNA and self.edge > 0

    @property
    def expired(self) -> bool:
        left = self.seconds_left
        return left is not None and left <= 0

    def to_row(self) -> dict:
        return {
            "source": self.source,
            "venue": self.venue,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "action": self.action,
            "confidence": self.confidence.value,
            "edge": round(self.edge, 4),
            "stake": self.stake,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "url": self.url,
            "expires_ms": self.expires_ms,
            "seconds_left": (
                round(self.seconds_left) if self.seconds_left is not None else None
            ),
            "tradeable": self.tradeable,
            "legs": self.legs,
        }


# --------------------------------------------------------------------------


def _expiry_ms(market: dict | None) -> int:
    """Turn the poll loop's countdown into a wall-clock deadline.

    Market rows carry `seconds_remaining`, measured when the row was
    built. Storing that number raw would freeze the countdown between
    polls; converting it to an absolute timestamp lets the browser tick it
    down every second, which matters when the whole window is five
    minutes long.
    """
    if not market:
        return 0
    remaining = market.get("seconds_remaining")
    if remaining is None:
        return 0
    return int(time.time() * 1000 + float(remaining) * 1000)


def from_btc_arbitrage(arb_events: list[dict], markets: list[dict]) -> list[Signal]:
    """Up+Down costing under $1 on a Polymarket BTC window.

    Exactly one side pays $1, so buying both at a combined cost below that
    is riskless whatever BTC does. The cost figure already carries fees
    and slippage, and it is measured against real book depth for the size
    quoted rather than against the best price on screen — those two
    numbers diverge exactly when the opportunity looks best.

    These windows are short. An arb event that fired forty seconds ago on
    a five-minute window may already be gone, which is why every signal
    carries its own expiry rather than a global freshness flag.
    """
    signals: list[Signal] = []

    for event in arb_events[:12]:
        label = event.get("market", "BTC")
        profit = event.get("profit_per_pair")
        pair_cost = event.get("pair_cost")
        if profit is None or profit <= 0:
            continue

        market = next(
            (m for m in markets if f"BTC {m.get('duration')}" == label), None
        )
        expires = _expiry_ms(market)
        slug = (market or {}).get("slug", "")

        signals.append(
            Signal(
                source="arbitraje_btc",
                venue="polymarket",
                asset="BTC",
                timeframe=label.replace("BTC ", ""),
                action=(
                    f"Comprar UP y DOWN a la vez en {label} "
                    f"({event.get('size', 25)} shares de cada uno)"
                ),
                confidence=Confidence.ALTA,
                edge=profit,
                stake=STAKE_BY_CONFIDENCE[Confidence.ALTA],
                rationale=(
                    f"El par cuesta {pair_cost * 100:.1f}¢ y paga $1 seguro: "
                    f"{profit * 100:.2f}¢ por par gane quien gane."
                ),
                evidence=(
                    "Aritmética pura, no depende de acertar la dirección. "
                    "Medido contra la profundidad real del libro, no contra el "
                    "mejor precio de pantalla."
                ),
                url=f"https://polymarket.com/event/{slug}" if slug else "",
                expires_ms=expires,
                legs=[{"side": "UP"}, {"side": "DOWN"}],
            )
        )

    return signals


def from_opportunities(rows: list[dict]) -> list[Signal]:
    """Category-strategy output, mapped onto the desk's confidence tiers.

    The strategies already classify themselves as structural, statistical
    or speculative. That classification is the thing being preserved here:
    a structural edge becomes ALTA and gets a real stake, everything else
    becomes BAJA at a dollar. Nothing gets promoted.
    """
    tier = {"structural": Confidence.ALTA}
    signals: list[Signal] = []

    for row in rows:
        confidence = tier.get(row.get("kind"), Confidence.BAJA)
        edge = row.get("edge") or 0.0
        if edge <= 0:
            continue

        venue = row.get("venue", "")
        strategy = row.get("strategy", "")
        label = row.get("label", "")

        # Each strategy means a different physical action, and the label
        # alone does not say which. A fade label already reads "X vs Y —
        # contra Z", so prefixing it with "Comprar en" produces an
        # instruction that contradicts itself.
        if strategy == "kalshi_ladder_arb" or venue == "kalshi":
            action, asset = f"Escalera Kalshi: {label}", "BTC"
        elif strategy == "favourite_fade":
            action = f"Apostar $1 {label}"
            asset = row.get("category", "")
        elif strategy == "sports_basket_arb":
            action = f"Comprar TODOS los resultados: {label}"
            asset = row.get("category", "")
        else:
            action = f"Comprar en {label}"
            asset = row.get("category", "")

        signals.append(
            Signal(
                source=row.get("strategy", "categoria"),
                venue=venue,
                asset=asset,
                timeframe=row.get("timeframe", "evento"),
                action=action,
                confidence=confidence,
                edge=edge,
                stake=STAKE_BY_CONFIDENCE[confidence],
                rationale=row.get("rationale", ""),
                evidence=row.get("evidence", ""),
                url=row.get("url", ""),
            )
        )

    return signals


def from_model_edges(scanner: list[dict], markets: list[dict] | None = None) -> list[Signal]:
    """Model-versus-market disagreements, marked explicitly as not-to-trade.

    The scanner flags these EDGE when the Brownian fair value sits far
    enough from the ask. Over 299 resolved windows that disagreement was
    measured and the model came out behind: Brier 0.202 against the
    market's 0.196, with the PnL from trading the difference
    indistinguishable from zero.

    They are still worth showing. A large disagreement usually means the
    inputs are stale or a window is about to resolve, which is useful to
    see. What it is not is a reason to buy, and the NINGUNA tier plus a
    zero stake is how that stays true on screen.
    """
    by_slug = {m.get("slug"): m for m in (markets or [])}
    signals: list[Signal] = []

    for row in scanner:
        if row.get("signal") != "EDGE":
            continue

        edge = row.get("edge") or 0.0
        side = row.get("side", "")
        label = row.get("market", "BTC")

        signals.append(
            Signal(
                source="modelo_vs_mercado",
                venue="polymarket",
                asset="BTC",
                timeframe=label.replace("BTC ", ""),
                action=f"NO OPERAR — el modelo ve {side} barato en {label}",
                confidence=Confidence.NINGUNA,
                edge=edge,
                stake=0.0,
                rationale=(
                    f"Modelo {row.get('fair', 0) * 100:.0f}¢ contra ask "
                    f"{(row.get('ask') or 0) * 100:.0f}¢ (z={row.get('z', 0):.2f})."
                ),
                evidence=(
                    "El backtest de 299 ventanas dio Brier 0.202 al modelo contra "
                    "0.196 del mercado: el modelo es PEOR. Operar estas diferencias "
                    "dio PnL indistinguible de cero. Se muestra como contexto — una "
                    "diferencia grande suele significar dato viejo o ventana a punto "
                    "de resolver."
                ),
                url=f"https://polymarket.com/event/{row.get('slug', '')}",
                expires_ms=_expiry_ms(by_slug.get(row.get("slug"))),
            )
        )

    return signals


# --------------------------------------------------------------------------


@dataclass
class DeskView:
    """Everything the trading tab needs for one render."""

    signals: list[Signal]
    no_trade_reason: str
    scanned: int

    def rows(self) -> list[dict]:
        return [s.to_row() for s in self.signals]

    def _count(self, tier: Confidence) -> int:
        return sum(1 for s in self.signals if s.confidence is tier and s.edge > 0)

    @property
    def actionable(self) -> int:
        """Arithmetic edges only.

        Deliberately excludes the statistical tier. Those carry a stake and
        so are things to *do*, but counting them here would report a dozen
        unconfirmed $1 samples as a dozen profitable opportunities — which
        is the single most misleading number this panel could show.
        """
        return self._count(Confidence.ALTA)

    @property
    def sampling(self) -> int:
        """Unconfirmed-bias bets, sized to build a sample rather than to earn."""
        return self._count(Confidence.BAJA)

    @property
    def informational(self) -> int:
        """Shown as context, explicitly not to be traded."""
        return self._count(Confidence.NINGUNA)


# Ranking. Structural first, then by edge — but an expiring window
# outranks a fatter edge that is not going anywhere, because a signal you
# cannot execute in time is worth nothing regardless of its size.
_TIER_ORDER = {Confidence.ALTA: 0, Confidence.BAJA: 1, Confidence.NINGUNA: 2}


def _sort_key(signal: Signal) -> tuple:
    left = signal.seconds_left
    urgent = 0 if (left is not None and left < 120) else 1
    return (_TIER_ORDER[signal.confidence], urgent, -signal.edge)


def build_desk(
    arb_events: list[dict],
    markets: list[dict],
    opportunities: list[dict],
    scanner: list[dict],
    asset: str = "todos",
    timeframe: str = "todos",
) -> DeskView:
    """Collect every source into one ranked list, filtered as asked.

    `asset` and `timeframe` mirror how a person actually uses this: pick
    the market you are watching, pick the horizon you are willing to hold,
    and see what is there. Filtering happens after collection so the
    "nothing here" message can distinguish between a quiet market and a
    filter that hid everything.
    """
    collected = (
        from_btc_arbitrage(arb_events, markets)
        + from_opportunities(opportunities)
        + from_model_edges(scanner, markets)
    )

    live = [s for s in collected if not s.expired]
    scanned = len(live)

    shown = live
    if asset != "todos":
        shown = [s for s in shown if s.asset.lower() == asset.lower()]
    if timeframe != "todos":
        shown = [s for s in shown if s.timeframe.lower() == timeframe.lower()]

    shown.sort(key=_sort_key)

    # Whether anything was *looked at* is a different question from whether
    # anything was *found*, and the two must not collapse: a healthy market
    # with no opportunity would otherwise report itself as a dead feed.
    had_data = bool(markets or opportunities or scanner)

    return DeskView(
        signals=shown,
        no_trade_reason=_no_trade_reason(shown, live, had_data),
        scanned=scanned,
    )


def _no_trade_reason(shown: list[Signal], live: list[Signal], had_data: bool) -> str:
    """Say why there is nothing, rather than showing an empty box.

    An empty panel is indistinguishable from a broken one, and on a tool
    whose whole purpose is removing hesitation, "I don't know if this is
    working" is the worst possible state to leave someone in.
    """
    if any(s.confidence is Confidence.ALTA and s.edge > 0 for s in shown):
        return ""

    if not had_data:
        return (
            "Sin datos de mercado ahora mismo. No es que no haya oportunidad: "
            "es que no se ha podido mirar."
        )

    sampling = [s for s in shown if s.confidence is Confidence.BAJA and s.edge > 0]
    if sampling:
        return (
            f"Ningún edge aritmético. Las {len(sampling)} señales de abajo son "
            f"apuestas de $1 para acumular muestra sobre el sesgo del favorito, "
            f"no oportunidades rentables — la hipótesis aún no está confirmada. "
            f"Tomarlas está bien; esperar grandes ganancias de ellas no."
        )

    if shown:
        return (
            "Nada operable. Lo que hay son diferencias del modelo contra el "
            "mercado, y el backtest ya demostró que ese modelo pierde contra el "
            "mercado. No operar es la recomendación."
        )

    if live:
        return (
            f"Hay {len(live)} señal(es) activas, pero ninguna pasa el filtro "
            f"elegido. Prueba con TODOS."
        )

    return (
        "Ninguna casa se está contradiciendo ahora mismo: no hay cestas por "
        "debajo de $1 ni escaleras invertidas. Esto es lo normal — un mercado "
        "eficiente no regala nada casi nunca. Esperar es la jugada."
    )
