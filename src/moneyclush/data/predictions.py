"""Paper-betting the favourite, scored against what actually happened.

The strategy is deliberately the most naive one available: on every
fixture, back whichever outcome the market already rates most likely.
That is not expected to make money — the backtest on 299 BTC windows
found the opposite, that markets *overprice* confident outcomes and
fading the favourite in the 0.60-0.90 band returned +2-4c per trade.
Running it live on sports is a way to see whether the same bias shows up
here, on a different asset class, with real resolutions.

Two prices per pick, and they are not interchangeable:

    pick_prob   de-vigged market probability at entry — the hypothesis
    pick_ask    what one share actually cost — the money

Calibration compares realised hit rates against `pick_prob`, never
against `pick_ask`: the ask sits half a spread above what the market
actually believes, so scoring the market's forecast at the ask charges it
the spread and manufactures a deficit of 2-3 points — the same order of
magnitude as the effect being measured.

Predictions are appended to a JSONL ledger so the record survives a
restart and cannot be quietly re-written after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import structlog

from moneyclush import calibration
from moneyclush.data import store

log = structlog.get_logger()

GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# A resolved outcome settles at exactly 1; allow for float noise.
RESOLVED_AT = 0.99

# Polymarket settles a market in two steps: a resolution is proposed to
# UMA, and only later does the whole event flip to `closed`. Waiting for
# the event flag means waiting on every prop market on the card, which on
# a UFC event is hours to days after the fight ended. Either signal on the
# individual result market is enough.
SETTLED_UMA_STATUSES = {"proposed", "resolved", "settled"}

# A pick that cannot be settled must not sit at the head of the queue
# forever. Retries back off geometrically from this base.
RETRY_BASE_SECONDS = 90.0
RETRY_MAX_SECONDS = 6 * 3600.0

# Seven days past kick-off with no settlement means the fixture is never
# going to settle. Keeping it pending would slowly fill the resolve batch
# with corpses until nothing new ever gets checked.
ABANDON_AFTER_HOURS = 168.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass
class Prediction:
    event_id: str
    title: str
    url: str
    league: str
    discipline: str
    pick: str                     # outcome we backed
    pick_prob: float              # de-vigged probability at pick time
    pick_ask: float               # price per share at pick time
    recorded_at: str
    start_time: str | None = None
    source: str = "ia"            # "ia" (automatic) | "manual" (user pick)
    stake: float = 1.0            # dollars committed
    volume_24h: float = 0.0       # book depth at entry, for banding
    resolved: bool = False
    won: bool | None = None
    winner: str | None = None
    resolved_at: str | None = None
    # Resolution bookkeeping. Without it a fixture that never settles is
    # retried at full rate forever and blocks everything behind it.
    attempts: int = 0
    last_attempt_at: str | None = None
    void: bool = False            # market settled paying nobody
    abandoned: bool = False       # gave up; excluded from every statistic

    @property
    def fair_prob(self) -> float:
        """The market's own probability at entry — the calibration baseline."""
        return self.pick_prob

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

    @property
    def settled(self) -> bool:
        """Whether this pick has left the queue, one way or another."""
        return self.resolved or self.void or self.abandoned

    def next_attempt_due(self) -> datetime:
        """When this pick may be checked again.

        Geometric backoff so a fixture stuck in dispute costs one request
        every six hours instead of one every two minutes.
        """
        last = _parse(self.last_attempt_at)
        if last is None or self.attempts <= 0:
            return datetime.min.replace(tzinfo=timezone.utc)
        delay = min(RETRY_BASE_SECONDS * (2 ** (self.attempts - 1)), RETRY_MAX_SECONDS)
        return last + timedelta(seconds=delay)

    def is_stale(self) -> bool:
        """Past the point where settlement is still plausible."""
        reference = _parse(self.start_time) or _parse(self.recorded_at)
        if reference is None:
            return False
        return _now() - reference > timedelta(hours=ABANDON_AFTER_HOURS)


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


def _result_markets(event: dict) -> list[tuple[str, float, bool]]:
    """(outcome label, its price, whether that market has settled).

    Both market shapes are handled: a 1X2 fixture carries one Yes/No
    market per side (plus the draw), while a moneyline holds the two
    competitors as complementary tokens inside a single market. Prop
    markets — totals, method of victory, round betting — share the event
    and must not be mistaken for the result.
    """
    teams = _loads(event.get("teams"), [])
    names = {t.get("name") for t in teams if isinstance(t, dict) and t.get("name")}

    found: list[tuple[str, float, bool]] = []
    for market in event.get("markets") or []:
        outcomes = [str(o) for o in _loads(market.get("outcomes"), [])]
        prices = [_as_float(p) for p in _loads(market.get("outcomePrices"), [])]
        if len(outcomes) != 2 or len(prices) != 2 or None in prices:
            continue

        uma = (market.get("umaResolutionStatus") or "").strip().lower()
        settled = bool(market.get("closed")) or uma in SETTLED_UMA_STATUSES
        group = (market.get("groupItemTitle") or "").strip()

        if [o.lower() for o in outcomes] == ["yes", "no"]:
            if group in names:
                found.append((group, prices[0], settled))
            elif group.lower().startswith("draw"):
                found.append(("Empate", prices[0], settled))
            continue

        if names and {o.strip() for o in outcomes} == names:
            for outcome, price in zip(outcomes, prices):
                found.append((outcome, price, settled))

    return found


def _decide(event: dict) -> tuple[str | None, bool]:
    """(winner, was_voided) for a fixture, from its result markets.

    Deliberately does *not* consult the event-level `closed` flag. That
    flag only flips once UMA has finalised every market on the event,
    which trails the actual result by hours or days — measured against
    live data, 138 of 139 pending picks had `closed=False` while a third
    of them already had a settled result market.

    Equally deliberately, a price of 99c on a market that has *not*
    settled is not treated as a winner. That is the market forecasting,
    not the fixture finishing, and recording it would be scoring the
    ledger against the very prices it is supposed to be testing.
    """
    markets = _result_markets(event)
    settled = [(label, price) for label, price, is_settled in markets if is_settled]
    if not settled:
        return None, False

    for label, price in settled:
        if price >= RESOLVED_AT:
            return label, False

    # Every deciding market settled and none paid out: the fixture was
    # cancelled and the stake comes back. Not a loss, not a win, no data.
    if all(abs(price - 0.5) < 1e-9 for _, price in settled):
        return None, True

    return None, False


@dataclass
class PredictionLedger:
    """Append-only record of favourite-backing picks and their outcomes."""

    path: Path
    predictions: dict[str, Prediction] = field(default_factory=dict)

    @staticmethod
    def _from_dict(data: dict) -> Prediction | None:
        """Tolerate rows written before a field existed rather than
        discarding history on every schema change."""
        data = dict(data)
        for derived in ("pnl", "payout", "pick_mid"):
            data.pop(derived, None)
        known = Prediction.__dataclass_fields__.keys()
        try:
            return Prediction(**{k: v for k, v in data.items() if k in known})
        except TypeError:
            return None

    def load(self) -> None:
        """Read the database, then fold in any legacy JSONL rows once."""
        for data in store.load_predictions():
            prediction = self._from_dict(data)
            if prediction:
                self.predictions[self._key(prediction)] = prediction

        migrated = 0
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except ValueError:
                    continue
                prediction = self._from_dict(data)
                if not prediction:
                    continue
                key = self._key(prediction)
                # Later lines supersede earlier ones, which is how a pick
                # recorded as pending becomes a resolved one.
                existing = self.predictions.get(key)
                if existing is None or (prediction.resolved and not existing.resolved):
                    self.predictions[key] = prediction
                    store.save_prediction(key, asdict(prediction))
                    migrated += 1
        if migrated:
            log.info("predictions.migrated_from_jsonl", rows=migrated)

    @staticmethod
    def _key(prediction: Prediction) -> str:
        """Automatic and manual picks coexist on the same fixture."""
        return f"{prediction.source}:{prediction.event_id}"

    def _persist(self, prediction: Prediction, audit: bool = True) -> None:
        """Write through to the database.

        `audit=False` is for bookkeeping that changes no outcome — retry
        counters and timestamps. Those must survive a restart, or the
        backoff resets and the queue jams again, but appending them to the
        plain-text ledger would bury the actual picks under thousands of
        lines of "checked again, still nothing".
        """
        store.save_prediction(self._key(prediction), asdict(prediction))
        if not audit:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("predictions.jsonl_write_failed", error=str(exc)[:80])

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
                recorded_at=_now().isoformat(),
                start_time=match.start_time.isoformat() if match.start_time else None,
                source="ia",
                volume_24h=round(getattr(match, "volume_24h", 0.0) or 0.0, 2),
            )
            self.predictions[self._key(prediction)] = prediction
            self._persist(prediction)
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
        volume_24h: float = 0.0,
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
            recorded_at=_now().isoformat(),
            source="manual",
            stake=stake,
            volume_24h=round(volume_24h or 0.0, 2),
        )
        self.predictions[self._key(prediction)] = prediction
        self._persist(prediction)
        return prediction

    def pending(self) -> list[Prediction]:
        """Picks still waiting on a result — excludes voids and write-offs."""
        return [p for p in self.predictions.values() if not p.settled]

    def _resolve_order(self) -> list[Prediction]:
        """Pending picks that are due a check, most likely to settle first.

        Two things starve this queue if left alone. Insertion order puts a
        manual pick behind a hundred automatic ones, so it never gets
        reached. And a fixture that will never settle — a cancelled match,
        a disputed resolution — sorts to the very front by age and holds
        its slot in every single batch, forever; the batch is a fixed-size
        prefix, so a handful of those permanently hides everything behind
        them. Filtering on the backoff clock is what breaks that: a stuck
        pick drops to one check every six hours and stops crowding out
        fixtures that are actually finishing.
        """
        now = _now()
        due = [p for p in self.pending() if p.next_attempt_due() <= now]

        def sort_key(p: Prediction):
            return (
                0 if p.source == "manual" else 1,
                p.attempts,                      # least-tried first
                p.start_time or p.recorded_at,
            )

        return sorted(due, key=sort_key)

    async def resolve(self, client: httpx.AsyncClient, batch: int = 40) -> int:
        """Settle any pending picks whose events have since closed."""
        outstanding = self._resolve_order()
        if not outstanding:
            return 0

        settled = 0
        for prediction in outstanding[:batch]:
            prediction.attempts += 1
            prediction.last_attempt_at = _now().isoformat()

            # Give up on fixtures that are never going to settle, so they
            # stop consuming a request every six hours until the heat death
            # of the universe.
            if prediction.is_stale():
                prediction.abandoned = True
                log.info(
                    "predictions.abandoned",
                    event_id=prediction.event_id,
                    attempts=prediction.attempts,
                )
                self._persist(prediction)
                continue

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
                self._persist(prediction, audit=False)
                continue

            if not payload:
                self._persist(prediction, audit=False)
                continue

            winner, was_void = _decide(payload[0])

            if was_void:
                # Stake returned, no information. Counting it as a loss
                # would invent a defeat the market never handed out.
                prediction.void = True
                prediction.resolved_at = _now().isoformat()
                log.info("predictions.void", event_id=prediction.event_id)
                self._persist(prediction)
                continue

            if winner is None:
                self._persist(prediction, audit=False)
                continue

            prediction.resolved = True
            prediction.winner = winner
            prediction.won = winner.strip().lower() == prediction.pick.strip().lower()
            prediction.resolved_at = _now().isoformat()
            self._persist(prediction)
            settled += 1

        return settled

    def _samples(self, picks: list[Prediction]) -> list[calibration.Sample]:
        return [
            calibration.Sample(
                prob=p.fair_prob,
                ask=p.pick_ask,
                won=bool(p.won),
                pnl=p.pnl,
                volume=p.volume_24h,
            )
            for p in picks
            if p.resolved and p.won is not None
        ]

    def stats(self, source: str | None = None) -> dict:
        picks = [
            p for p in self.predictions.values()
            if source is None or p.source == source
        ]
        resolved = [p for p in picks if p.resolved]
        samples = self._samples(resolved)
        overall = calibration.summary(samples)

        pnl = sum(p.pnl for p in resolved)
        staked = sum(p.stake for p in resolved)
        open_picks = [p for p in picks if not p.settled]

        return {
            "total": len(picks),
            "pending": len(open_picks),
            "resolved": len(resolved),
            "void": sum(1 for p in picks if p.void),
            "abandoned": sum(1 for p in picks if p.abandoned),
            "wins": overall["wins"],
            "losses": len(resolved) - overall["wins"],
            "hit_rate": overall["hit_rate"],
            # What the market's own prices said would happen. The gap
            # between this and hit_rate is the entire experiment.
            "expected_hit_rate": overall["expected_hit_rate"],
            "ci_low": overall["ci_low"],
            "ci_high": overall["ci_high"],
            "edge_pts": overall["edge_pts"],
            "z": overall["z"],
            "significant": overall["significant"],
            "verdict": calibration.verdict(overall),
            "staked": round(staked, 2),
            "pnl": round(pnl, 4),
            "pnl_per_trade": round(pnl / len(resolved), 4) if resolved else None,
            "roi": round(pnl / staked, 4) if staked > 0 else None,
            "open_stake": round(sum(p.stake for p in open_picks), 2),
            "calibration": calibration.calibration_rows(samples),
            # Thin books are where stale prices survive; deep ones are
            # already arbitraged. The volume was being fetched and thrown
            # away. Rows written before the field existed carry 0, which
            # is "unknown" rather than "illiquid" — counting them as the
            # latter would stuff the thin band with the entire back
            # catalogue and invent a result there.
            "calibration_volume": calibration.calibration_rows(
                [s for s in samples if s.volume > 0],
                calibration.VOLUME_BANDS,
                by="volume",
            ),
            "volume_unknown": sum(1 for s in samples if s.volume <= 0),
            "curve": self._equity_curve(resolved),
        }

    @staticmethod
    def _equity_curve(resolved: list[Prediction]) -> list[list]:
        """Cumulative PnL over time, for the header sparkline."""
        ordered = sorted(resolved, key=lambda p: p.resolved_at or "")
        curve = []
        running = 0.0
        for p in ordered:
            running += p.pnl
            parsed = _parse(p.resolved_at)
            if parsed is None:
                continue
            curve.append([int(parsed.timestamp() * 1000), round(running, 4)])
        return curve[-300:]

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
