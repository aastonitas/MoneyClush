"""Risk management: Kelly sizing, exposure limits, kill switches.

Enforces hard limits that no strategy can override.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from moneyclush.data.models import Position

log = structlog.get_logger()


def fractional_kelly_size(
    win_probability: float,
    entry_price: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Position size as fraction of bankroll using fractional Kelly criterion.

    Returns 0.0 if the edge is negative or inputs are invalid.
    """
    if not (0 < entry_price < 1) or not (0 < win_probability < 1):
        return 0.0

    loss_probability = 1 - win_probability
    net_odds = (1 - entry_price) / entry_price
    if net_odds <= 0:
        return 0.0

    full_kelly = (net_odds * win_probability - loss_probability) / net_odds
    if full_kelly <= 0:
        return 0.0

    return full_kelly * kelly_fraction


@dataclass
class DailyPnLTracker:
    """Tracks cumulative PnL for the current trading day."""

    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    _day_start: str = field(default_factory=lambda: "")

    def record(self, pnl: float) -> None:
        self.realized_pnl += pnl

    def reset_if_new_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._day_start:
            self._day_start = today
            self.realized_pnl = 0.0
            self.unrealized_pnl = 0.0

    @property
    def total(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class KillSwitchState:
    """Tracks conditions that should halt all trading."""

    price_feed_stale: bool = False
    api_latency_high: bool = False
    daily_loss_exceeded: bool = False
    manual_halt: bool = False

    @property
    def triggered(self) -> bool:
        return any([
            self.price_feed_stale,
            self.api_latency_high,
            self.daily_loss_exceeded,
            self.manual_halt,
        ])

    @property
    def reason(self) -> str:
        reasons = []
        if self.price_feed_stale:
            reasons.append("stale_price_feed")
        if self.api_latency_high:
            reasons.append("high_api_latency")
        if self.daily_loss_exceeded:
            reasons.append("daily_loss_limit")
        if self.manual_halt:
            reasons.append("manual_halt")
        return ", ".join(reasons) if reasons else "none"


@dataclass
class RiskManager:
    """Enforces position limits, Kelly sizing, and kill switches."""

    max_bankroll_usd: float = 1000.0
    kelly_fraction: float = 0.25
    max_loss_daily_usd: float = 50.0
    max_position_per_market: float = 200.0
    max_uncovered_shares: int = 500
    max_correlated_positions: int = 3
    kill_switch_stale_data_sec: float = 10.0
    kill_switch_latency_ms: float = 2000.0

    pnl: DailyPnLTracker = field(default_factory=DailyPnLTracker)
    kill_switch: KillSwitchState = field(default_factory=KillSwitchState)
    _active_positions: dict[str, Position] = field(default_factory=dict)

    def compute_position_size(
        self,
        win_probability: float,
        entry_price: float,
        bankroll: float | None = None,
    ) -> float:
        """Compute the dollar amount to risk on a single trade."""
        if bankroll is None:
            bankroll = self.max_bankroll_usd

        fraction = fractional_kelly_size(
            win_probability, entry_price, self.kelly_fraction
        )
        dollar_size = fraction * bankroll
        dollar_size = min(dollar_size, self.max_position_per_market)
        shares = dollar_size / entry_price if entry_price > 0 else 0
        return shares

    def check_order(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
        price_feed_age_ms: float,
        api_latency_ms: float = 0,
    ) -> tuple[bool, str]:
        """Validate a proposed order against all risk limits.

        Returns (approved, reason).
        """
        self.pnl.reset_if_new_day()

        self.kill_switch.price_feed_stale = (
            price_feed_age_ms > self.kill_switch_stale_data_sec * 1000
        )
        self.kill_switch.api_latency_high = (
            api_latency_ms > self.kill_switch_latency_ms
        )
        self.kill_switch.daily_loss_exceeded = (
            self.pnl.total < -self.max_loss_daily_usd
        )

        if self.kill_switch.triggered:
            log.warning("risk.kill_switch", reason=self.kill_switch.reason)
            return False, f"kill_switch: {self.kill_switch.reason}"

        pos = self._active_positions.get(market_id)
        if pos is not None:
            if pos.directional_exposure + size > self.max_uncovered_shares:
                return False, "max_uncovered_shares exceeded"

            if pos.total_invested + size * price > self.max_position_per_market:
                return False, "max_position_per_market exceeded"

        if len(self._active_positions) >= self.max_correlated_positions:
            if market_id not in self._active_positions:
                return False, "max_correlated_positions exceeded"

        return True, "approved"

    def register_position(self, position: Position) -> None:
        self._active_positions[position.market_condition_id] = position

    def remove_position(self, market_id: str) -> None:
        self._active_positions.pop(market_id, None)

    def record_pnl(self, amount: float) -> None:
        self.pnl.record(amount)

    def halt(self) -> None:
        self.kill_switch.manual_halt = True

    def resume(self) -> None:
        self.kill_switch.manual_halt = False
