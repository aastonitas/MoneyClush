"""Safety guardrails for live order placement: limits, a kill switch, and
the single choke point every real order must clear before it is sent.

Nothing calls `check_order_allowed` yet — `ExecutionEngine` still only
produces `OrderPlan` objects for paper/backtest use, and nothing posts a
signed order to Polymarket (see execution/engine.py roadmap items 2-4).
This module exists so that when that wiring happens, there is one place
that decides "is this order actually allowed to go out," instead of the
check being scattered across the strategy and the dashboard and easy to
forget in one of them.

Fail-safe by construction: a brand new checkout, a fresh `data/` directory,
a server that has never run before — none of these should ever look like
"trading enabled." Two independent markers, checked with `or`, not `and`:

- ARMED_FILE must exist for live orders to be considered at all. Its
  absence (the default) blocks everything. Creating it is an explicit,
  separate action from anything the strategy or the dashboard does on its
  own.
- KILL_FILE, if present, blocks everything regardless of ARMED_FILE. This
  is the emergency stop: trivially easy to trigger (touch the file, or hit
  a dashboard button that touches it), deliberately not
  trivially easy to clear (no dashboard "resume" button in the same flow —
  clearing it is a separate, considered action).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
ARMED_FILE = DATA_DIR / "TRADING_ARMED"
KILL_FILE = DATA_DIR / "KILL_SWITCH"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SafetyLimits:
    """Every field is env-overridable so limits can be tuned without a
    code change. Defaults are the ones agreed on 2026-08-01, sized against
    a real balance of ~$14: a single mistake should be recoverable, and a
    bug that fires repeatedly should still be caught by the daily/hourly
    caps before it can matter.
    """

    max_order_usd: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_MAX_ORDER_USD", 1.0)
    )
    max_open_exposure_usd: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_MAX_OPEN_EXPOSURE_USD", 3.0)
    )
    max_daily_loss_usd: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_MAX_DAILY_LOSS_USD", 3.0)
    )
    max_orders_per_hour: int = field(
        default_factory=lambda: _env_int("MONEYCLUSH_MAX_ORDERS_PER_HOUR", 10)
    )
    min_seconds_remaining: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_MIN_SECONDS_REMAINING", 30.0)
    )
    # Never let an order plan spend the last cent of the on-chain balance —
    # leaves room for fee/rounding drift between the read and the fill.
    balance_buffer_usd: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_BALANCE_BUFFER_USD", 0.05)
    )


class KillSwitch:
    """File-backed on purpose: it has to work even if the dashboard
    process is the one that's hung or crash-looping, and a human should be
    able to stop trading with `del data/TRADING_ARMED` from a terminal
    with no dashboard involved at all.
    """

    def __init__(
        self, armed_file: Path = ARMED_FILE, kill_file: Path = KILL_FILE
    ) -> None:
        self.armed_file = armed_file
        self.kill_file = kill_file

    def is_killed(self) -> bool:
        return self.kill_file.exists()

    def is_armed(self) -> bool:
        return self.armed_file.exists() and not self.is_killed()

    def trigger_stop(self, reason: str = "") -> None:
        """The emergency stop. Idempotent, and always safe to call."""
        self.kill_file.parent.mkdir(parents=True, exist_ok=True)
        self.kill_file.write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}".strip() + "\n"
        )

    def clear_stop(self) -> None:
        """Deliberately separate from `trigger_stop` — resuming after an
        emergency stop should never be one click in the same flow that
        triggered it.
        """
        self.kill_file.unlink(missing_ok=True)

    def arm(self) -> None:
        self.armed_file.parent.mkdir(parents=True, exist_ok=True)
        self.armed_file.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    def disarm(self) -> None:
        self.armed_file.unlink(missing_ok=True)

    def status(self) -> dict:
        return {
            "armed": self.is_armed(),
            "killed": self.is_killed(),
            # Live orders happen only when armed and not killed. Spelling
            # this out as its own field means the dashboard never has to
            # re-derive the boolean logic and risk getting it backwards.
            "live_orders_allowed": self.is_armed() and not self.is_killed(),
        }


def check_order_allowed(
    order_usd: float,
    *,
    kill_switch: KillSwitch,
    limits: SafetyLimits,
    on_chain_balance_usd: float | None,
    open_exposure_usd: float,
    daily_realized_pnl_usd: float,
    orders_in_last_hour: int,
    seconds_remaining: float,
) -> tuple[bool, str]:
    """The single choke point a live order must clear before it is sent.

    Returns (allowed, reason). `reason` is always populated, including on
    approval, so a caller can log *why* an order went out, not just that
    it did.
    """
    if kill_switch.is_killed():
        return False, "kill switch engaged"
    if not kill_switch.is_armed():
        return False, "trading not armed"
    if on_chain_balance_usd is None:
        return False, "on-chain balance unavailable — refusing to trade blind"
    if order_usd > limits.max_order_usd:
        return False, f"order ${order_usd:.2f} exceeds max ${limits.max_order_usd:.2f}"
    if order_usd + limits.balance_buffer_usd > on_chain_balance_usd:
        return False, (
            f"order ${order_usd:.2f} + buffer exceeds on-chain balance "
            f"${on_chain_balance_usd:.2f}"
        )
    if open_exposure_usd + order_usd > limits.max_open_exposure_usd:
        return False, (
            f"open exposure ${open_exposure_usd:.2f} + order would exceed max "
            f"${limits.max_open_exposure_usd:.2f}"
        )
    if daily_realized_pnl_usd <= -limits.max_daily_loss_usd:
        return False, (
            f"daily loss ${-daily_realized_pnl_usd:.2f} already at/past max "
            f"${limits.max_daily_loss_usd:.2f}"
        )
    if orders_in_last_hour >= limits.max_orders_per_hour:
        return False, f"already {orders_in_last_hour} orders in the last hour"
    if seconds_remaining < limits.min_seconds_remaining:
        return False, (
            f"only {seconds_remaining:.0f}s left, below min "
            f"{limits.min_seconds_remaining:.0f}s"
        )
    return True, "all checks passed"
