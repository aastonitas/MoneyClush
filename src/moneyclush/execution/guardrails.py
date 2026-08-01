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

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
ARMED_FILE = DATA_DIR / "TRADING_ARMED"
KILL_FILE = DATA_DIR / "KILL_SWITCH"

# Arming lapses after this long. Long enough to cover a watched session,
# short enough that walking away and forgetting is self-correcting rather
# than open-ended.
ARM_DURATION_SECONDS = 2 * 60 * 60


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
    code change. Sizing agreed on 2026-08-01: order size is a *fraction*
    of the live on-chain balance (order_fraction), not a flat dollar
    figure — the same idea as the unused KELLY_FRACTION in .env.example.
    A flat dollar cap does not grow when the account grows and, worse,
    does not shrink after a loss unless someone remembers to lower it by
    hand; a fraction does both automatically. max_order_usd is kept as an
    absolute backstop ceiling only — protection against a misread balance
    or a future bug, not the number sizing is supposed to target.
    """

    order_fraction: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_ORDER_FRACTION", 0.25)
    )
    max_order_usd: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_MAX_ORDER_USD", 10.0)
    )
    # Two order-fractions' worth open at once, not a flat dollar figure,
    # for the same reason order size itself isn't one.
    max_open_exposure_fraction: float = field(
        default_factory=lambda: _env_float("MONEYCLUSH_MAX_OPEN_EXPOSURE_FRACTION", 0.5)
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


def _floor_cents(value: float) -> float:
    """Round down to the cent. Money you might spend must never be
    rounded up — round(3.555, 2) == 3.56 would let two "25%" orders add
    up to more than the 50% exposure cap they're supposed to fit under.
    """
    return math.floor(value * 100) / 100


def compute_order_size_usd(balance_usd: float, limits: SafetyLimits) -> float:
    """What the next order should cost, in dollars — `order_fraction` of
    the live balance, never more than the absolute backstop ceiling.
    """
    return _floor_cents(min(balance_usd * limits.order_fraction, limits.max_order_usd))


def compute_max_exposure_usd(balance_usd: float, limits: SafetyLimits) -> float:
    """How much can be open across all positions at once, in dollars."""
    return _floor_cents(balance_usd * limits.max_open_exposure_fraction)


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

    def armed_expires_at(self) -> float | None:
        """Epoch seconds when the current arming lapses, or None if not
        armed. Unparseable contents count as expired, not as armed —
        a corrupt or hand-edited marker must never read as "go".
        """
        try:
            return float(self.armed_file.read_text().strip().split()[0])
        except (OSError, ValueError, IndexError):
            return None

    def is_armed(self) -> bool:
        """Arming lapses on its own.

        A marker that stays "on" until someone remembers to switch it off
        is the wrong default for something that spends real money: the
        failure mode of forgetting is unbounded. Expiry makes the safe
        state the one you drift into, not the one you must maintain.
        """
        if self.is_killed():
            return False
        expires_at = self.armed_expires_at()
        return expires_at is not None and time.time() < expires_at

    def armed_seconds_left(self) -> float:
        expires_at = self.armed_expires_at()
        if expires_at is None:
            return 0.0
        return max(0.0, expires_at - time.time())

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

    def arm(self, duration_seconds: float = ARM_DURATION_SECONDS) -> float:
        """Arm live trading until `duration_seconds` from now.

        Returns the expiry epoch. Writes the expiry first and the
        human-readable stamp second, so a partially-written file still
        parses to a bounded time rather than to "armed forever".
        """
        expires_at = time.time() + duration_seconds
        self.armed_file.parent.mkdir(parents=True, exist_ok=True)
        self.armed_file.write_text(
            f"{expires_at:.0f} armed_until="
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expires_at))}\n"
        )
        return expires_at

    def disarm(self) -> None:
        self.armed_file.unlink(missing_ok=True)

    def status(self) -> dict:
        return {
            "armed": self.is_armed(),
            "killed": self.is_killed(),
            "armed_seconds_left": round(self.armed_seconds_left()),
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

    max_order_now = compute_order_size_usd(on_chain_balance_usd, limits)
    if order_usd > max_order_now:
        return False, (
            f"order ${order_usd:.2f} exceeds {limits.order_fraction*100:.0f}% of "
            f"balance (${max_order_now:.2f}, backstop ${limits.max_order_usd:.2f})"
        )
    if order_usd + limits.balance_buffer_usd > on_chain_balance_usd:
        return False, (
            f"order ${order_usd:.2f} + buffer exceeds on-chain balance "
            f"${on_chain_balance_usd:.2f}"
        )

    max_exposure_now = compute_max_exposure_usd(on_chain_balance_usd, limits)
    if open_exposure_usd + order_usd > max_exposure_now:
        return False, (
            f"open exposure ${open_exposure_usd:.2f} + order would exceed "
            f"{limits.max_open_exposure_fraction*100:.0f}% of balance "
            f"(${max_exposure_now:.2f})"
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
