"""Durable storage for everything the bot accumulates.

Until now the ledger lived in a JSONL file and the paper-trading history
lived in memory, which meant a redeploy erased both. That is fatal for
this project specifically: the favourite-bias question needs on the order
of a thousand settled bets, and at a few hundred a day that is weeks of
uptime. Losing the record on every deploy means never reaching an answer.

SQLite is enough here — one file, no server, transactional writes. On a
platform with an ephemeral filesystem the file still needs to sit on a
mounted volume; `MONEYCLUSH_DB` points at it, and `storage_is_durable()`
reports whether that was actually configured so the UI can warn instead
of quietly accumulating data that will vanish.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

_DEFAULT_PATH = Path("data") / "moneyclush.db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def db_path() -> Path:
    return Path(os.environ.get("MONEYCLUSH_DB") or _DEFAULT_PATH)


def storage_is_durable() -> bool:
    """Whether the database was pointed somewhere deliberate.

    A default path on a container filesystem is wiped on redeploy, so the
    honest answer when nobody set MONEYCLUSH_DB is "no".
    """
    return bool(os.environ.get("MONEYCLUSH_DB"))


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn

    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(path), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    # WAL keeps reads from blocking the background writers.
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA synchronous=NORMAL")
    _init(_conn)
    log.info("store.connected", path=str(path), durable=storage_is_durable())
    return _conn


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            key           TEXT PRIMARY KEY,
            event_id      TEXT NOT NULL,
            source        TEXT NOT NULL,
            payload       TEXT NOT NULL,
            resolved      INTEGER NOT NULL DEFAULT 0,
            recorded_at   TEXT,
            resolved_at   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_pred_resolved ON predictions(resolved);
        CREATE INDEX IF NOT EXISTS idx_pred_source ON predictions(source);

        CREATE TABLE IF NOT EXISTS fav_bets (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            settled_at    INTEGER NOT NULL,
            market        TEXT,
            side          TEXT,
            price         REAL,
            won           INTEGER,
            pnl           REAL
        );
        CREATE INDEX IF NOT EXISTS idx_fav_time ON fav_bets(settled_at);
        """
    )
    conn.commit()


# ---------------------------------------------------------------- predictions

def save_prediction(key: str, data: dict[str, Any]) -> None:
    conn = connect()
    with _lock:
        conn.execute(
            """INSERT INTO predictions
                   (key, event_id, source, payload, resolved, recorded_at, resolved_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET
                   payload=excluded.payload,
                   resolved=excluded.resolved,
                   resolved_at=excluded.resolved_at""",
            (
                key,
                data.get("event_id", ""),
                data.get("source", "ia"),
                json.dumps(data, ensure_ascii=False),
                1 if data.get("resolved") else 0,
                data.get("recorded_at"),
                data.get("resolved_at"),
            ),
        )
        conn.commit()


def load_predictions() -> list[dict[str, Any]]:
    conn = connect()
    rows = conn.execute("SELECT payload FROM predictions").fetchall()
    out = []
    for row in rows:
        try:
            out.append(json.loads(row["payload"]))
        except ValueError:
            continue
    return out


def prediction_count() -> int:
    return connect().execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]


# ------------------------------------------------------------------ fav track

def save_fav_bet(
    settled_at: int, market: str, side: str, price: float, won: bool, pnl: float
) -> None:
    conn = connect()
    with _lock:
        conn.execute(
            """INSERT INTO fav_bets (settled_at, market, side, price, won, pnl)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (settled_at, market, side, price, 1 if won else 0, pnl),
        )
        conn.commit()


def load_fav_history(limit: int = 600) -> tuple[list[list], dict]:
    """Cumulative PnL curve and aggregate stats, oldest first."""
    conn = connect()
    rows = conn.execute(
        "SELECT settled_at, price, won, pnl FROM fav_bets ORDER BY settled_at"
    ).fetchall()

    curve: list[list] = []
    total = 0.0
    wins = 0
    expected = 0.0
    for row in rows:
        total += row["pnl"]
        wins += row["won"]
        expected += row["price"]
        curve.append([row["settled_at"], round(total, 2)])

    n = len(rows)
    stats = {
        "resolved": n,
        "wins": wins,
        "staked": float(n),
        "expected": expected,
    }
    if n:
        stats["hit_rate"] = round(wins / n, 4)
        stats["expected_hit_rate"] = round(expected / n, 4)
        stats["roi"] = round(total / n, 4)
    return curve[-limit:], stats


def fav_pnl() -> float:
    row = connect().execute("SELECT COALESCE(SUM(pnl), 0) AS s FROM fav_bets").fetchone()
    return float(row["s"])
