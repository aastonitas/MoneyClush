"""Strategies for the categories beyond BTC Up/Down.

The BTC work established one hard result: a model built only from price
cannot beat a market that aggregates order flow. Over 299 resolved
windows the model's Brier score was 0.202 against the market's 0.196, and
trading the difference paid nothing. That result is not a reason to try
the same thing on a different asset — it is a reason to stop looking for
edges that require forecasting better than the market does.

So the strategies here are sorted by what they actually depend on, and
that ordering is the point:

    structural   The edge is arithmetic. A basket of mutually exclusive
                 outcomes that costs less than its guaranteed payout is
                 free money whatever happens, and no forecast is
                 involved. `SportsBasketArb` and `LadderArb` are these.

    statistical  The edge depends on a bias in the market that we have
                 measured but not yet confirmed. `FavouriteFade` is this,
                 and it is explicitly not significant yet.

    speculative  The edge depends on two venues disagreeing. Part of any
                 disagreement is settlement basis that never converges,
                 so this one reports rather than recommends.
                 `CrossVenueBasis` is this.

Nothing here places orders. Each strategy returns `Opportunity` objects
that carry their own evidence, so the caller can tell a proven edge from
an unproven one without reading the source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from moneyclush.data.kalshi import LadderGroup, ladder_arbitrages, round_trip_cost
from moneyclush.data.sports import Match


class EdgeKind(str, Enum):
    """What has to be true for the trade to make money.

    This is the most important field on an opportunity. A structural edge
    pays regardless of the outcome; a statistical one pays only if a
    measured bias is real; a speculative one pays only if a price gap is
    mispricing rather than basis.
    """

    STRUCTURAL = "structural"
    STATISTICAL = "statistical"
    SPECULATIVE = "speculative"


@dataclass
class Opportunity:
    """One actionable (or merely observable) situation in a market."""

    strategy: str
    venue: str
    category: str
    kind: EdgeKind
    label: str
    edge: float                 # expected profit per $1 committed, after costs
    stake_hint: float           # dollars this strategy would put on it
    rationale: str
    evidence: str               # what we actually know about this edge holding
    url: str = ""
    legs: list[dict] = field(default_factory=list)

    @property
    def actionable(self) -> bool:
        """Structural edges are the only ones this project trades.

        The rest are surfaced for measurement. Keeping the distinction in
        code rather than in a comment is what stops an unproven bias from
        quietly becoming a live position.
        """
        return self.kind is EdgeKind.STRUCTURAL and self.edge > 0


class CategoryStrategy(ABC):
    """A scanner over one category's markets."""

    name: str = "base"
    venue: str = ""
    category: str = ""
    kind: EdgeKind = EdgeKind.SPECULATIVE

    @abstractmethod
    def scan(self, markets) -> list[Opportunity]:
        """Return every opportunity visible in the current snapshot."""
        ...


# --------------------------------------------------------------------------
# Structural


class SportsBasketArb(CategoryStrategy):
    """Buy every outcome of a fixture when the basket costs under $1.

    A 1X2 football market on Polymarket is three separate binary markets
    that happen to be mutually exclusive and collectively exhaustive.
    Exactly one pays $1. So buying all three costs the sum of their asks
    and returns $1 with certainty — if that sum is below $1, the
    difference is riskless profit, no forecast required.

    Two details decide whether this is real or imaginary:

    Asks, not last-traded prices. A book showing 100.5c on last trades and
    102c on asks is not arbitrageable; only the second number is what you
    would actually pay. Using last trades makes books look far more
    arbitrageable than they are.

    Moneyline fixtures cannot work. When both sides come from one market
    as complementary tokens, their asks sum to $1 plus the spread by
    construction. `Match.arb_profit` already excludes those, and this
    strategy relies on that exclusion rather than repeating it.
    """

    name = "sports_basket_arb"
    venue = "polymarket"
    category = "deportes"
    kind = EdgeKind.STRUCTURAL

    def __init__(self, min_edge: float = 0.005, stake: float = 100.0):
        self.min_edge = min_edge
        self.stake = stake

    def scan(self, matches: list[Match]) -> list[Opportunity]:
        found: list[Opportunity] = []
        for match in matches:
            profit = match.arb_profit
            if profit is None or profit < self.min_edge:
                continue

            found.append(
                Opportunity(
                    strategy=self.name,
                    venue=self.venue,
                    category=self.category,
                    kind=self.kind,
                    label=match.title,
                    edge=profit,
                    stake_hint=self.stake,
                    rationale=(
                        f"Comprar los {len(match.outcomes)} resultados cuesta "
                        f"{match.overround:.4f} y devuelve $1 seguro."
                    ),
                    evidence=(
                        "Aritmética, no pronóstico: exactamente un resultado paga $1. "
                        "El único riesgo es de ejecución — que una pata no se llene "
                        "al precio visto."
                    ),
                    url=match.url,
                    legs=[
                        {"label": o.label, "kind": o.kind, "ask": o.best_ask,
                         "token_id": o.token_id_yes}
                        for o in match.outcomes
                    ],
                )
            )

        found.sort(key=lambda o: -o.edge)
        return found


class LadderArb(CategoryStrategy):
    """Exploit a Kalshi strike ladder contradicting itself.

    "BTC above $63,000" cannot be less likely than "BTC above $64,000":
    every world where the higher strike pays also pays the lower one. So
    going long the low strike and short the high strike yields $1 inside
    the band between them and $0 outside — a payoff that is never
    negative. When the low strike's ask sits below the high strike's bid,
    that non-negative payoff comes with cash up front.

    This structure exists only on Kalshi. Polymarket's BTC market is a
    single Up/Down binary with nothing to be inconsistent against; Kalshi
    lists twenty-odd strikes on the same asset and the same hour, each
    with its own book, and those books drift apart.

    The catch is fees. Kalshi charges takers on both legs, up to 1.75c per
    contract near 50c, so the raw credit has to clear roughly 3-4c before
    anything is left. `ladder_arbitrages` nets that out.
    """

    name = "kalshi_ladder_arb"
    venue = "kalshi"
    category = "crypto"
    kind = EdgeKind.STRUCTURAL

    def __init__(self, min_net: float = 0.005, stake: float = 100.0):
        self.min_net = min_net
        self.stake = stake

    def scan(self, ladders: list[LadderGroup]) -> list[Opportunity]:
        found: list[Opportunity] = []
        for group in ladders:
            for arb in ladder_arbitrages(group, min_net=self.min_net):
                found.append(
                    Opportunity(
                        strategy=self.name,
                        venue=self.venue,
                        category=self.category,
                        kind=self.kind,
                        label=f"{group.series} · {arb.label}",
                        edge=arb.net,
                        stake_hint=self.stake,
                        rationale=(
                            f"Comprar {arb.low.subtitle} a {arb.low.yes_ask:.2f} y "
                            f"vender {arb.high.subtitle} a {arb.high.yes_bid:.2f} "
                            f"deja {arb.credit:.3f} de crédito por un pago que nunca "
                            f"es negativo."
                        ),
                        evidence=(
                            "Monotonía: un strike más bajo no puede ser menos probable "
                            f"que uno más alto. Comisión de las dos patas ya descontada "
                            f"({arb.fees:.3f})."
                        ),
                        legs=[
                            {"ticker": arb.low.ticker, "side": "buy_yes",
                             "price": arb.low.yes_ask},
                            {"ticker": arb.high.ticker, "side": "sell_yes",
                             "price": arb.high.yes_bid},
                        ],
                    )
                )

        found.sort(key=lambda o: -o.edge)
        return found


# --------------------------------------------------------------------------
# Statistical


class FavouriteFade(CategoryStrategy):
    """Back the underdog where the market overrates the favourite.

    The BTC backtest found the market systematically overprices favourites
    in the 0.60-0.90 band, worth +2-4c per trade against the underdog. At
    the window level the 0.80-0.90 band gave p=0.031 and the combined
    0.60-0.90 band p=0.084 across 299 windows. Neither clears significance
    once you account for having looked at several bands, which is exactly
    why this class is STATISTICAL and never STRUCTURAL: the edge might be
    noise, and roughly 1000 windows are needed to tell.

    Running it on sports and on trending events is not a second bet on the
    same hypothesis — it is an independent test of it. Sports fixtures are
    priced by different traders on different information than BTC windows,
    so if the same bias shows up there, that is real evidence it is a
    property of how people price favourites rather than a quirk of one
    market. If it does not, the BTC result was probably noise.
    """

    name = "favourite_fade"
    venue = "polymarket"
    category = "deportes"
    kind = EdgeKind.STATISTICAL

    # The band the backtest flagged. Outside it there is either no
    # meaningful favourite or the outcome is effectively decided.
    BAND_LOW = 0.60
    BAND_HIGH = 0.90

    def __init__(self, stake: float = 1.0):
        self.stake = stake

    def scan(self, matches: list[Match]) -> list[Opportunity]:
        found: list[Opportunity] = []
        for match in matches:
            if match.decided or match.status == "ended":
                continue

            fav = match.favourite()
            if fav is None:
                continue
            fav_outcome, fav_prob = fav
            if not self.BAND_LOW <= fav_prob <= self.BAND_HIGH:
                continue

            # Fading the favourite means backing everything else. With a
            # draw on the board that is two tickets, not one.
            others = [
                (o, p) for o, p in match.fair_probabilities()
                if o is not fav_outcome
            ]
            if not others:
                continue

            found.append(
                Opportunity(
                    strategy=self.name,
                    venue=self.venue,
                    category=self.category,
                    kind=self.kind,
                    label=f"{match.title} — contra {fav_outcome.label}",
                    # The measured effect, not a computed edge. Naming it
                    # honestly matters: this number is a hypothesis.
                    edge=0.03,
                    stake_hint=self.stake,
                    rationale=(
                        f"El mercado cotiza a {fav_outcome.label} al "
                        f"{fav_prob * 100:.0f}%, dentro de la banda donde el backtest "
                        f"encontró sobrevaloración."
                    ),
                    evidence=(
                        "NO CONFIRMADO. +2-4¢/operación sobre 299 ventanas de BTC, "
                        "p=0.031 en 0.80-0.90 y p=0.084 en 0.60-0.90. Hacen falta "
                        "~1000 ventanas para concluir. Deportes es un test "
                        "independiente de la misma hipótesis."
                    ),
                    url=match.url,
                    legs=[
                        {"label": o.label, "kind": o.kind, "ask": o.best_ask,
                         "prob": round(p, 4)}
                        for o, p in others
                    ],
                )
            )

        found.sort(key=lambda o: -o.edge)
        return found


# --------------------------------------------------------------------------
# Speculative


class CrossVenueBasis(CategoryStrategy):
    """Compare the same question priced on Kalshi and on Polymarket.

    Both venues run BTC up/down markets, so a price gap looks like an
    arbitrage. It usually is not, for a reason that does not go away with
    better execution: they settle against different numbers. Kalshi uses
    CF Benchmarks' BRTI index and Polymarket uses Chainlink. Two indices
    on the same asset disagree by a small, variable amount, and that
    component of the gap never converges — you can hold both legs to
    expiry and still lose.

    On top of that, Kalshi charges a taker fee on both legs where
    Polymarket charges none, and Kalshi's BTC books are 12-29c wide
    against Polymarket's far tighter ones.

    So this reports three numbers separately — raw gap, cost to cross,
    and what survives — instead of one "edge" that hides the difference.
    A gap only becomes interesting once it exceeds the cost by more than
    plausible index basis, and the basis has never been measured here.
    """

    name = "cross_venue_basis"
    venue = "kalshi+polymarket"
    category = "crypto"
    kind = EdgeKind.SPECULATIVE

    def __init__(self, min_gap: float = 0.02):
        self.min_gap = min_gap

    def scan(self, pairs: list[dict]) -> list[Opportunity]:
        """Compare pre-matched questions.

        Each pair needs `label`, `kalshi_mid`, `kalshi_spread` and
        `poly_mid`. Matching the two venues' questions is the caller's job
        because it depends on strike and expiry alignment that only makes
        sense with both order books in hand.
        """
        found: list[Opportunity] = []
        for pair in pairs:
            k_mid, p_mid = pair.get("kalshi_mid"), pair.get("poly_mid")
            if k_mid is None or p_mid is None:
                continue

            gap = abs(k_mid - p_mid)
            if gap < self.min_gap:
                continue

            # Crossing costs half the spread on each venue plus Kalshi's fee.
            cost = (pair.get("kalshi_spread") or 0.0) / 2.0 + round_trip_cost(k_mid)
            survives = gap - cost

            cheaper = "Kalshi" if k_mid < p_mid else "Polymarket"
            found.append(
                Opportunity(
                    strategy=self.name,
                    venue=self.venue,
                    category=self.category,
                    kind=self.kind,
                    label=pair.get("label", "BTC"),
                    edge=survives,
                    stake_hint=0.0,
                    rationale=(
                        f"{cheaper} cotiza {gap * 100:.1f}¢ más barato; cruzar cuesta "
                        f"{cost * 100:.1f}¢, quedan {survives * 100:.1f}¢."
                    ),
                    evidence=(
                        "NO ES ARBITRAJE. Kalshi liquida con CF Benchmarks y "
                        "Polymarket con Chainlink: parte de la diferencia es base "
                        "entre índices y no converge nunca. Esa base sigue sin medirse "
                        "en este proyecto."
                    ),
                    legs=[
                        {"venue": "kalshi", "mid": k_mid},
                        {"venue": "polymarket", "mid": p_mid},
                    ],
                )
            )

        found.sort(key=lambda o: -o.edge)
        return found
