"""Paper-betting the favourite, scored against what actually happened.

The strategy is deliberately the most naive one available: on every
fixture, back whichever outcome the market already rates most likely.
That is not expected to make money — the backtest on 299 BTC windows
found the opposite, that markets *overprice* confident outcomes and
fading the favourite in the 0.60-0.90 band returned +2-4c per trade.
Running it live on sports is a way to see whether the same bias shows up
here, on a different asset class, with real resolutions.

Each pick is priced at the ask, so one share costs what it would actually
cost to buy: a win returns `1 - ask`, a loss returns `-ask`. Quoting the
last-traded price instead would flatter every number.

Predictions are appended to a JSONL ledger so the record survives a
restart and cannot be quietly re-written after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger()

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# A resolved outcome settles at exactly 1; allow for float noise.
RESOLVED_AT = 0.99


@dataclass
class Prediction:
    event_id: str
    title: str
    url: str
    league: str
    discipline: str
    pick: str                     # outcome we backed
    pick_prob: float              # normalised probability at pick time
    pick_ask: float               # price per share at pick time
    recorded_at: str
    start_time: str | None = None
    source: str = "ia"            # "ia" (automatic) | "manual" (user pick)
    stake: float = 1.0            # dollars committed
    resolved: bool = False
    won: bool | None = None
    winner: str | None = None
    resolved_at: str | None = None

    @property
    def shares(self) -> float:
        """A $1 stake buys 1/price shares, each settling at $1 or nothing."""
        if self.pick_ask <= 0:
            return 0.0
        return self.stake / self.pick_ask

    @property
    def pnl(self) -> float:
        """Profit in dollars; zero while the fixture is still open."""
        if not self.resolved or self.won is None:
            return 0.0
        return (self.shares - self.stake) if self.won else -self.stake

    @property
    def potential_payout(self) -> float:
        """What the stake returns if the pick lands."""
        return self.shares


def _loads(value, default):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return default
    return value if value is not None else default


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _winner_of(event: dict) -> str | None:
    """Which outcome settled at 1, or None if the event has not resolved.

    Both market shapes are handled: a 1X2 fixture resolves the winning
    side's own Yes/No market to 1, while a moneyline resolves one of the
    two competitor tokens inside a single market.
    """
    if not event.get("closed"):
        return None

    teams = _loads(event.get("teams"), [])
    names = {t.get("name") for t in teams if isinstance(t, dict) and t.get("name")}

    for market in event.get("markets") or []:
        outcomes = [str(o) for o in _loads(market.get("outcomes"), [])]
        prices = [_as_float(p) for p in _loads(market.get("outcomePrices"), [])]
        if len(outcomes) != 2 or len(prices) != 2 or None in prices:
            continue

        group = (market.get("groupItemTitle") or "").strip()

        # 1X2: a Yes/No market named after one side (or the draw).
        if [o.lower() for o in outcomes] == ["yes", "no"]:
            is_result_market = group in names or group.lower().startswith("draw")
            if is_result_market and prices[0] >= RESOLVED_AT:
                return "Empate" if group.lower().startswith("draw") else group
            continue

        # Moneyline: the two competitors inside one market.
        if names and {o.strip() for o in outcomes} == names:
            for outcome, price in zip(outcomes, prices):
                if price >= RESOLVED_AT:
                    return outcome

    return None


@dataclass
class PredictionLedger:
    """Append-only record of favourite-backing picks and their outcomes."""

    path: Path
    predictions: dict[str, Prediction] = field(default_factory=dict)

    def load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except ValueError:
                continue
            # Tolerate rows written before a field existed rather than
            # discarding history on every schema change.
            data.pop("pnl", None)
            known = Prediction.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in known}
            try:
                prediction = Prediction(**filtered)
            except TypeError:
                continue
            # Later lines for the same key supersede earlier ones, which is
            # how a pick recorded as pending becomes a resolved one.
            self.predictions[self._key(prediction)] = prediction

    @staticmethod
    def _key(prediction: Prediction) -> str:
        """Automatic and manual picks coexist on the same fixture."""
        return f"{prediction.source}:{prediction.event_id}"

    def _append(self, prediction: Prediction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")

    def record(self, matches: list, min_prob: float = 0.0) -> int:
        """Back the favourite on any fixture not already in the ledger.

        A pick is only taken once per event: re-recording as the price
        drifts would let the ledger cherry-pick its own entry point.
        """
        added = 0
        for match in matches:
            if f"ia:{match.event_id}" in self.predictions:
                continue
            # Backing a result the market has already called is not a
            # forecast, and it would inflate the hit rate for free.
            if getattr(match, "decided", False):
                continue
            favourite = match.favourite()
            if favourite is None:
                continue

            outcome, prob = favourite
            if prob < min_prob or outcome.best_ask is None:
                continue
            # A side already trading at 1 has effectively resolved; there is
            # no bet left to place and it would flatter the hit rate.
            if outcome.best_ask >= RESOLVED_AT:
                continue

            prediction = Prediction(
                event_id=match.event_id,
                title=match.title,
                url=match.url,
                league=match.league,
                discipline=match.discipline,
                pick=outcome.label,
                pick_prob=round(prob, 4),
                pick_ask=round(outcome.best_ask, 4),
                recorded_at=datetime.now(timezone.utc).isoformat(),
                start_time=match.start_time.isoformat() if match.start_time else None,
                source="ia",
            )
            self.predictions[self._key(prediction)] = prediction
            self._append(prediction)
            added += 1
        return added

    def record_manual(
        self,
        event_id: str,
        title: str,
        url: str,
        pick: str,
        pick_prob: float,
        pick_ask: float,
        league: str = "",
        discipline: str = "",
        stake: float = 1.0,
    ) -> Prediction:
        """A pick the user made by hand, priced the same way as an automatic one."""
        prediction = Prediction(
            event_id=event_id,
            title=title,
            url=url,
            league=league,
            discipline=discipline,
            pick=pick,
            pick_prob=round(pick_prob, 4),
            pick_ask=round(pick_ask, 4),
            recorded_at=datetime.now(timezone.utc).isoformat(),
            source="manual",
            stake=stake,
        )
        self.predictions[self._key(prediction)] = prediction
        self._append(prediction)
        return prediction

    def pending(self) -> list[Prediction]:
        return [p for p in self.predictions.values() if not p.resolved]

    async def resolve(self, client: httpx.AsyncClient, batch: int = 20) -> int:
        """Settle any pending picks whose events have since closed."""
        outstanding = self.pending()
        if not outstanding:
            return 0

        settled = 0
        for prediction in outstanding[:batch]:
            try:
                resp = await client.get(
                    GAMMA_EVENTS,
                    params={"id": prediction.event_id},
                    headers=UA_HEADERS,
                    timeout=12,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                log.warning("predictions.resolve_failed", error=str(exc)[:80])
                continue

            if not payload:
                continue
            winner = _winner_of(payload[0])
            if winner is None:
                continue

            prediction.resolved = True
            prediction.winner = winner
            prediction.won = winner.strip().lower() == prediction.pick.strip().lower()
            prediction.resolved_at = datetime.now(timezone.utc).isoformat()
            self._append(prediction)
            settled += 1

        return settled

    def stats(self, source: str | None = None) -> dict:
        picks = [
            p for p in self.predictions.values()
            if source is None or p.source == source
        ]
        resolved = [p for p in picks if p.resolved]
        wins = sum(1 for p in resolved if p.won)
        pnl = sum(p.pnl for p in resolved)
        staked = sum(p.stake for p in resolved)

        # Expected hit rate if the market's prices are honest. Comparing the
        # realised rate against this is the whole point of the exercise.
        expected = sum(p.pick_prob for p in resolved)

        return {
            "total": len(picks),
            "pending": len(picks) - len(resolved),
            "resolved": len(resolved),
            "wins": wins,
            "losses": len(resolved) - wins,
            "hit_rate": round(wins / len(resolved), 4) if resolved else None,
            "expected_hit_rate": (
                round(expected / len(resolved), 4) if resolved else None
            ),
            "staked": round(staked, 2),
            "pnl": round(pnl, 4),
            "pnl_per_trade": round(pnl / len(resolved), 4) if resolved else None,
            "roi": round(pnl / staked, 4) if staked > 0 else None,
            "open_stake": round(sum(p.stake for p in picks if not p.resolved), 2),
        }

    def recent(self, limit: int = 40, source: str | None = None) -> list[dict]:
        picks = [
            p for p in self.predictions.values()
            if source is None or p.source == source
        ]
        ordered = sorted(
            picks, key=lambda p: (p.resolved_at or p.recorded_at), reverse=True
        )
        return [
            {
                **asdict(p),
                "pnl": round(p.pnl, 4),
                "payout": round(p.potential_payout, 4),
            }
            for p in ordered[:limit]
        ]

    def has_manual(self, event_id: str) -> bool:
        return f"manual:{event_id}" in self.predictions
