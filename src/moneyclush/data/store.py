"""Durable storage for everything the bot accumulates.

Until now the ledger lived in a JSONL file and the paper-trading history
lived in memory, which meant a redeploy erased both. That is fatal for
this project specifically: the favourite-bias question needs on the order
of a thousand settled bets, and at a few hundred a day that is weeks of
uptime. Losing the record on every deploy means never reaching an answer.

SQLite is enough here — one file, no server, transactional writes. On a
platform with an ephemeral filesystem the file still needs to sit on a
mounted volume, and `MONEYCLUSH_DB` points at it.

Whether that actually happened is *not* something the environment can be
asked. Setting `MONEYCLUSH_DB=/data/moneyclush.db` without mounting a
volume at `/data` gets you a perfectly writable directory on the
container's own disk: every write succeeds, nothing warns, and the whole
history disappears at the next deploy. Checking that the variable exists
is therefore worse than not checking at all — it silences the alarm
precisely when it matters.

So durability is proven, not assumed. Every boot writes a row with the
platform's deployment id; the storage counts as durable once rows from an
*earlier, different* deployment are found sitting in the database. Until
that has happened the honest answer is "unproven", and where the platform
makes it knowable — a path that is not a mount point — the answer is a
flat "ephemeral".
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import structlog

from moneyclush import calibration

log = structlog.get_logger()

_DEFAULT_PATH = Path("data") / "moneyclush.db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

# Set once at startup by record_boot(); the UI reads it every poll.
_status: dict = {"state": "unknown", "reason": "sin comprobar todavía"}

# Environment variables that mean "this filesystem is not yours to keep".
_PLATFORM_VARS = (
    "RAILWAY_DEPLOYMENT_ID",
    "RAILWAY_SERVICE_ID",
    "FLY_ALLOC_ID",
    "RENDER_INSTANCE_ID",
    "K_REVISION",
    "DYNO",
)


def db_path() -> Path:
    return Path(os.environ.get("MONEYCLUSH_DB") or _DEFAULT_PATH)


def deployment_id() -> str | None:
    """An identifier that changes on every redeploy, or None when local."""
    for var in ("RAILWAY_DEPLOYMENT_ID", "FLY_ALLOC_ID", "RENDER_INSTANCE_ID",
                "K_REVISION", "DYNO", "RAILWAY_GIT_COMMIT_SHA"):
        value = os.environ.get(var)
        if value:
            return f"{var}:{value}"
    return None


def on_ephemeral_platform() -> bool:
    return any(os.environ.get(v) for v in _PLATFORM_VARS)


def _is_mounted(path: Path) -> bool | None:
    """Whether the database's directory is its own mount point.

    On a container this is the difference between a volume and a folder
    that ceases to exist at the next deploy. Returns None where the
    question is not meaningful (local disk).
    """
    try:
        return os.path.ismount(str(path.parent))
    except OSError:
        return None


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
    log.info("store.connected", path=str(path))
    return _conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


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

        -- Bets placed but not yet settled. Held only in memory until now,
        -- so every restart silently dropped whatever was in flight.
        CREATE TABLE IF NOT EXISTS fav_open (
            condition_id  TEXT PRIMARY KEY,
            payload       TEXT NOT NULL
        );

        -- One row per process start. This is what makes durability a
        -- measurement instead of a guess.
        CREATE TABLE IF NOT EXISTS boots (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            booted_at     TEXT NOT NULL,
            deployment_id TEXT
        );

        -- Web Push subscriptions. The endpoint URL is unique per browser
        -- installation and doubles as the primary key; p256dh/auth are the
        -- keys pywebpush needs to encrypt the payload for that browser.
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            endpoint      TEXT PRIMARY KEY,
            p256dh        TEXT NOT NULL,
            auth          TEXT NOT NULL,
            created_at    TEXT NOT NULL
        );
        """
    )
    # The favourite track originally stored only the ask it paid and then
    # scored the market's forecast against that same number, charging the
    # spread to the forecast. The mid is the market's actual estimate.
    if "mid" not in _columns(conn, "fav_bets"):
        conn.execute("ALTER TABLE fav_bets ADD COLUMN mid REAL")
    conn.commit()


# --------------------------------------------------------------- durability

def record_boot() -> dict:
    """Log this start-up and work out whether the data is actually safe."""
    global _status
    conn = connect()
    path = db_path()
    configured = bool(os.environ.get("MONEYCLUSH_DB"))
    current = deployment_id()
    platform = on_ephemeral_platform()

    with _lock:
        prior = conn.execute(
            "SELECT deployment_id FROM boots ORDER BY id"
        ).fetchall()
        conn.execute(
            "INSERT INTO boots (booted_at, deployment_id) VALUES (datetime('now'), ?)",
            (current,),
        )
        conn.commit()

    prior_ids = {r["deployment_id"] for r in prior if r["deployment_id"]}
    survived = sorted(prior_ids - {current}) if current else []
    mounted = _is_mounted(path) if platform else None

    if platform and not configured:
        state = "ephemeral"
        reason = (
            "MONEYCLUSH_DB no está definida: la base vive en el disco del "
            "contenedor y se borrará en el próximo despliegue."
        )
    elif platform and mounted is False:
        state = "ephemeral"
        reason = (
            f"MONEYCLUSH_DB apunta a {path}, pero {path.parent} no es un punto "
            "de montaje — es una carpeta del contenedor. Falta crear el volumen "
            "y montarlo ahí; la variable por sí sola no persiste nada."
        )
    elif survived:
        state = "durable"
        reason = (
            f"verificado: los datos sobrevivieron a {len(survived)} despliegue(s) "
            f"anterior(es), {len(prior)} arranques registrados."
        )
    elif not prior:
        state = "unproven"
        reason = (
            "primer arranque sobre esta base — todavía no se puede demostrar "
            "que sobreviva a un despliegue."
        )
    elif current is None:
        state = "durable"
        reason = f"disco local, {len(prior)} arranques previos sobre el mismo fichero."
    else:
        state = "unproven"
        reason = (
            f"{len(prior)} arranques, todos del mismo despliegue. Se confirmará "
            "en cuanto sobreviva a un redeploy."
        )

    _status = {
        "state": state,
        "reason": reason,
        "path": str(path),
        "configured": configured,
        "mounted": mounted,
        "boots": len(prior) + 1,
        "deployments_survived": len(survived),
    }
    log.info("store.boot", **_status)
    return _status


def storage_status() -> dict:
    return dict(_status)


def storage_is_durable() -> bool:
    """Only true once persistence has actually been demonstrated."""
    return _status.get("state") == "durable"


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

def save_open_fav(condition_id: str, payload: dict[str, Any]) -> None:
    """Remember a bet that has been placed but not yet settled."""
    conn = connect()
    with _lock:
        conn.execute(
            """INSERT INTO fav_open (condition_id, payload) VALUES (?, ?)
               ON CONFLICT(condition_id) DO UPDATE SET payload=excluded.payload""",
            (condition_id, json.dumps(payload, ensure_ascii=False)),
        )
        conn.commit()


def load_open_favs() -> dict[str, dict]:
    conn = connect()
    out: dict[str, dict] = {}
    for row in conn.execute("SELECT condition_id, payload FROM fav_open"):
        try:
            out[row["condition_id"]] = json.loads(row["payload"])
        except ValueError:
            continue
    return out


def drop_open_fav(condition_id: str) -> None:
    conn = connect()
    with _lock:
        conn.execute("DELETE FROM fav_open WHERE condition_id = ?", (condition_id,))
        conn.commit()


def save_fav_bet(
    settled_at: int,
    market: str,
    side: str,
    price: float,
    won: bool,
    pnl: float,
    mid: float | None = None,
) -> None:
    conn = connect()
    with _lock:
        conn.execute(
            """INSERT INTO fav_bets (settled_at, market, side, price, won, pnl, mid)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (settled_at, market, side, price, 1 if won else 0, pnl, mid),
        )
        conn.commit()


def load_fav_history(limit: int = 600) -> tuple[list[list], dict]:
    """Cumulative PnL curve and aggregate stats, oldest first.

    The calibration baseline is the mid, not the ask that was paid: rows
    written before the two were separated fall back to the ask, which
    understates the market's accuracy by half a spread.
    """
    conn = connect()
    rows = conn.execute(
        "SELECT settled_at, price, mid, won, pnl FROM fav_bets ORDER BY settled_at"
    ).fetchall()

    curve: list[list] = []
    total = 0.0
    samples: list[calibration.Sample] = []
    for row in rows:
        total += row["pnl"]
        curve.append([row["settled_at"], round(total, 2)])
        samples.append(
            calibration.Sample(
                prob=row["mid"] if row["mid"] is not None else row["price"],
                ask=row["price"],
                won=bool(row["won"]),
                pnl=row["pnl"],
            )
        )

    overall = calibration.summary(samples)
    stats = {
        "resolved": len(rows),
        "wins": overall["wins"],
        "staked": float(len(rows)),
        "hit_rate": overall["hit_rate"],
        "expected_hit_rate": overall["expected_hit_rate"],
        "ci_low": overall["ci_low"],
        "ci_high": overall["ci_high"],
        "edge_pts": overall["edge_pts"],
        "z": overall["z"],
        "significant": overall["significant"],
        "verdict": calibration.verdict(overall),
        "roi": round(total / len(rows), 4) if rows else None,
        "calibration": calibration.calibration_rows(samples),
    }
    return curve[-limit:], stats


def fav_pnl() -> float:
    row = connect().execute("SELECT COALESCE(SUM(pnl), 0) AS s FROM fav_bets").fetchone()
    return float(row["s"])


# ------------------------------------------------------------ push subscriptions

def save_push_subscription(endpoint: str, p256dh: str, auth: str) -> None:
    conn = connect()
    with _lock:
        conn.execute(
            """INSERT INTO push_subscriptions (endpoint, p256dh, auth, created_at)
               VALUES (?, ?, ?, datetime('now'))
               ON CONFLICT(endpoint) DO UPDATE SET p256dh=excluded.p256dh, auth=excluded.auth""",
            (endpoint, p256dh, auth),
        )
        conn.commit()


def delete_push_subscription(endpoint: str) -> None:
    conn = connect()
    with _lock:
        conn.execute("DELETE FROM push_subscriptions WHERE endpoint = ?", (endpoint,))
        conn.commit()


def load_push_subscriptions() -> list[dict[str, Any]]:
    conn = connect()
    return [dict(r) for r in conn.execute(
        "SELECT endpoint, p256dh, auth FROM push_subscriptions"
    )]
