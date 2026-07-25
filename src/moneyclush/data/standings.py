"""League tables from TheSportsDB.

The free tier returns the top five of any table and nothing more, so this
is a leaderboard rather than a full standings page. That limit is surfaced
to the caller rather than hidden, because a table that silently stops at
fifth place looks like a bug.

MLS is a special case: it runs two conferences and the endpoint returns
them interleaved, which is why its ranks repeat (1, 1, 2, 2...). The
conference name is kept alongside each row so the display can say so.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx
import structlog

log = structlog.get_logger()

API = "https://www.thesportsdb.com/api/v1/json/3/lookuptable.php"
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Free-tier keys cap the response; anything beyond this is not available.
ROW_LIMIT = 5


@dataclass(frozen=True)
class LeagueSpec:
    key: str
    label: str
    country: str
    league_id: str
    season: str


LEAGUES: tuple[LeagueSpec, ...] = (
    LeagueSpec("peru", "Liga 1", "Perú", "4688", "2026"),
    LeagueSpec("mx", "Liga MX", "México", "4350", "2026-2027"),
    LeagueSpec("laliga", "LaLiga", "España", "4335", "2026-2027"),
    LeagueSpec("mls", "MLS", "EE.UU.", "4346", "2026"),
)


@dataclass
class TeamRow:
    rank: int
    team: str
    played: int
    win: int
    draw: int
    loss: int
    goal_diff: int
    points: int
    conference: str | None = None
    badge: str | None = None


@dataclass
class Standing:
    key: str
    label: str
    country: str
    season: str
    rows: list[TeamRow] = field(default_factory=list)
    updated: str | None = None
    error: str | None = None

    @property
    def started(self) -> bool:
        """False before a season's first matchday, when every row is zeroed."""
        return any(r.played > 0 for r in self.rows)


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


async def _fetch_one(client: httpx.AsyncClient, spec: LeagueSpec) -> Standing:
    standing = Standing(
        key=spec.key, label=spec.label, country=spec.country, season=spec.season
    )
    try:
        resp = await client.get(
            API,
            params={"l": spec.league_id, "s": spec.season},
            headers=UA_HEADERS,
            timeout=12,
        )
        resp.raise_for_status()
        table = resp.json().get("table") or []
    except Exception as exc:
        log.warning("standings.failed", league=spec.key, error=str(exc)[:80])
        standing.error = str(exc)[:80]
        return standing

    for entry in table[:ROW_LIMIT]:
        standing.rows.append(
            TeamRow(
                rank=_as_int(entry.get("intRank")),
                team=entry.get("strTeam") or "",
                played=_as_int(entry.get("intPlayed")),
                win=_as_int(entry.get("intWin")),
                draw=_as_int(entry.get("intDraw")),
                loss=_as_int(entry.get("intLoss")),
                goal_diff=_as_int(entry.get("intGoalDifference")),
                points=_as_int(entry.get("intPoints")),
                conference=entry.get("strConference") or entry.get("strDivision"),
                badge=entry.get("strBadge"),
            )
        )
    if table:
        standing.updated = table[0].get("dateUpdated")
    return standing


async def fetch_standings(client: httpx.AsyncClient) -> list[Standing]:
    results = await asyncio.gather(
        *(_fetch_one(client, spec) for spec in LEAGUES), return_exceptions=True
    )
    return [r for r in results if isinstance(r, Standing)]


def to_rows(standings: list[Standing]) -> list[dict]:
    return [
        {
            "key": s.key,
            "label": s.label,
            "country": s.country,
            "season": s.season,
            "updated": s.updated,
            "started": s.started,
            "error": s.error,
            "limited_to": ROW_LIMIT,
            "rows": [
                {
                    "rank": r.rank,
                    "team": r.team,
                    "played": r.played,
                    "win": r.win,
                    "draw": r.draw,
                    "loss": r.loss,
                    "goal_diff": r.goal_diff,
                    "points": r.points,
                    "conference": r.conference,
                }
                for r in s.rows
            ],
        }
        for s in standings
    ]
