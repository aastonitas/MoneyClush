"""Temporal Arbitrage Strategy.

Builds a paired position (Up + Down) across two different market states,
targeting a combined cost below $1.00 per pair.

The strategy:
1. When one side is cheap (e.g., Down drops to 29¢ after BTC pumps), buy it
2. Wait for the opposite side to become cheap
3. Build matched pairs at combined cost < $1.00
4. Collect $1.00 at resolution regardless of outcome

Risk: the second leg may never get cheap enough. The strategy manages
this by building in small incremental blocks and tracking the maximum
acceptable price for the second leg at all times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

from moneyclush.data.models import MarketState, OutcomeSide, Position
from moneyclush.pricing.fair_value import FairValueResult
from moneyclush.signals.order_book import combined_pair_cost
from moneyclush.strategies.base import SignalAction, Strategy, TradeSignal

log = structlog.get_logger()


@dataclass
class TemporalArbitrageStrategy(Strategy):
    """Accumulates Up and Down at different times to lock in < $1 pairs."""

    name: str = "temporal_arbitrage"

    max_pair_cost: float = 0.96
    block_size: float = 50.0
    min_edge_per_pair: float = 0.02
    max_single_side_price: float = 0.48
    min_time_remaining_sec: float = 30.0
    execution_cost: float = 0.018  # fees + expected slippage
    _pending_leg: Optional[OutcomeSide] = field(default=None, init=False)

    def evaluate(
        self,
        state: MarketState,
        fair_value: FairValueResult,
        position: Position,
    ) -> Optional[TradeSignal]:
        if state.seconds_remaining < self.min_time_remaining_sec:
            return None

        instant_arb = self._check_instant_arbitrage(state, position)
        if instant_arb is not None:
            return instant_arb

        return self._check_leg_opportunity(state, fair_value, position)

    def _check_instant_arbitrage(
        self,
        state: MarketState,
        position: Position,
    ) -> Optional[TradeSignal]:
        """Check if both sides are simultaneously cheap enough for instant arb."""
        pair_cost = combined_pair_cost(
            state.book_up, state.book_down, self.block_size
        )
        if pair_cost is None:
            return None

        net_profit = 1.0 - pair_cost - 0.012
        if net_profit < self.min_edge_per_pair:
            return None

        cheaper_side, cheaper_price = self._cheaper_side(state)
        if cheaper_price is None:
            return None

        log.info(
            "temporal_arb.instant",
            pair_cost=f"{pair_cost:.4f}",
            net_profit=f"{net_profit:.4f}",
        )

        return TradeSignal(
            action=(
                SignalAction.BUY_UP
                if cheaper_side == OutcomeSide.UP
                else SignalAction.BUY_DOWN
            ),
            side=cheaper_side,
            target_price=cheaper_price,
            target_size=self.block_size,
            edge=net_profit,
            confidence=0.85,
            reason=f"instant_arb: pair_cost={pair_cost:.3f}, profit={net_profit:.3f}",
        )

    def _check_leg_opportunity(
        self,
        state: MarketState,
        fair_value: FairValueResult,
        position: Position,
    ) -> Optional[TradeSignal]:
        """Check if one leg is cheap enough to start/continue building a pair.

        A leg is only worth buying when it is cheap relative to what the
        outcome is actually worth. Buying whichever side happens to sit
        below a fixed price threshold is not an edge: it measures distance
        to that threshold, not mispricing, and will happily pay 46c for an
        outcome the model prices at 33c.
        """
        if not fair_value.valid:
            return None

        ask_up = state.book_up.best_ask
        ask_down = state.book_down.best_ask

        if position.shares_up > position.shares_down:
            needed = OutcomeSide.DOWN
            budget = self.max_pair_cost - position.avg_cost_up - 0.012
            current_ask = ask_down
        elif position.shares_down > position.shares_up:
            needed = OutcomeSide.UP
            budget = self.max_pair_cost - position.avg_cost_down - 0.012
            current_ask = ask_up
        else:
            needed, current_ask = self._cheaper_side(state)
            budget = self.max_single_side_price

        if current_ask is None or current_ask > budget:
            return None

        # The leg must also be underpriced against its own fair value,
        # after fees and slippage.
        fair = (
            fair_value.posterior_up
            if needed == OutcomeSide.UP
            else fair_value.posterior_down
        )
        edge = fair - current_ask - self.execution_cost
        if edge < self.min_edge_per_pair:
            return None

        size = min(
            self.block_size,
            self.block_size - position.directional_exposure,
        )
        if size <= 0:
            return None

        log.info(
            "temporal_arb.leg",
            side=needed.value,
            price=f"{current_ask:.4f}",
            fair=f"{fair:.4f}",
            budget=f"{budget:.4f}",
            edge=f"{edge:.4f}",
        )

        return TradeSignal(
            action=(
                SignalAction.BUY_UP
                if needed == OutcomeSide.UP
                else SignalAction.BUY_DOWN
            ),
            side=needed,
            target_price=current_ask,
            target_size=size,
            edge=edge,
            confidence=0.60,
            reason=f"leg_build: {needed.value}@{current_ask:.3f} (max {max_price:.3f})",
        )

    def should_exit(
        self,
        state: MarketState,
        fair_value: FairValueResult,
        position: Position,
    ) -> Optional[TradeSignal]:
        """Exit if nearing resolution with uncovered inventory."""
        if state.seconds_remaining > self.min_time_remaining_sec:
            return None

        if position.directional_exposure <= 0:
            return None

        exposed_side = position.directional_side
        if exposed_side is None:
            return None

        opposite = (
            OutcomeSide.DOWN
            if exposed_side == OutcomeSide.UP
            else OutcomeSide.UP
        )
        action = (
            SignalAction.SELL_UP
            if exposed_side == OutcomeSide.UP
            else SignalAction.SELL_DOWN
        )

        log.warning(
            "temporal_arb.force_exit",
            exposed_side=exposed_side.value,
            exposure=position.directional_exposure,
            seconds_left=state.seconds_remaining,
        )

        return TradeSignal(
            action=action,
            side=exposed_side,
            target_price=0.0,
            target_size=position.directional_exposure,
            edge=-0.05,
            confidence=0.30,
            reason=f"force_exit: {exposed_side.value} exposure with {state.seconds_remaining:.0f}s left",
        )

    @staticmethod
    def _cheaper_side(
        state: MarketState,
    ) -> tuple[OutcomeSide, Optional[float]]:
        ask_up = state.book_up.best_ask
        ask_down = state.book_down.best_ask
        if ask_up is None and ask_down is None:
            return OutcomeSide.UP, None
        if ask_up is None:
            return OutcomeSide.DOWN, ask_down
        if ask_down is None:
            return OutcomeSide.UP, ask_up
        if ask_up <= ask_down:
            return OutcomeSide.UP, ask_up
        return OutcomeSide.DOWN, ask_down
