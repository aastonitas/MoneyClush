"""Backtesting engine for simulating strategies against historical data.

Replays a sequence of MarketState snapshots through a strategy,
tracking fills, positions, PnL, and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from moneyclush.data.models import MarketState, OutcomeSide, Position
from moneyclush.execution.engine import ExecutionEngine
from moneyclush.pricing.fair_value import FairValueEngine
from moneyclush.risk.manager import RiskManager
from moneyclush.strategies.base import Strategy, SignalAction

log = structlog.get_logger()


@dataclass
class TradeRecord:
    timestamp_ms: int
    market_id: str
    side: str
    action: str
    price: float
    size: float
    edge: float
    reason: str


@dataclass
class BacktestResult:
    total_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pairs_completed: int = 0
    max_drawdown: float = 0.0
    sharpe_estimate: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    pnl_curve: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def summary(self) -> dict:
        return {
            "total_pnl": round(self.total_pnl, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "pairs_completed": self.total_pairs_completed,
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_estimate": round(self.sharpe_estimate, 4),
        }


@dataclass
class Backtester:
    """Replays historical market states through a strategy."""

    strategy: Strategy
    fair_value_engine: FairValueEngine = field(default_factory=FairValueEngine)
    execution_engine: ExecutionEngine = field(default_factory=ExecutionEngine)
    risk_manager: RiskManager = field(default_factory=RiskManager)
    initial_bankroll: float = 1000.0

    def run(self, snapshots: list[MarketState]) -> BacktestResult:
        """Run backtest over a sequence of market snapshots.

        Each snapshot represents the market at one point in time.
        Resolution happens when seconds_remaining reaches 0.
        """
        result = BacktestResult()
        position = Position(market_condition_id="backtest")
        bankroll = self.initial_bankroll
        peak_bankroll = bankroll
        pnl_per_resolution: list[float] = []
        current_market_id = ""

        for i, state in enumerate(snapshots):
            if state.info.condition_id != current_market_id:
                if position.paired_shares > 0 or position.directional_exposure > 0:
                    pnl = self._resolve_position(state, position)
                    bankroll += pnl
                    result.total_pnl += pnl
                    pnl_per_resolution.append(pnl)
                    if pnl > 0:
                        result.winning_trades += 1
                    elif pnl < 0:
                        result.losing_trades += 1
                    result.total_pairs_completed += int(position.paired_shares)

                    peak_bankroll = max(peak_bankroll, bankroll)
                    drawdown = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                    result.max_drawdown = max(result.max_drawdown, drawdown)

                    position = Position(market_condition_id=state.info.condition_id)

                current_market_id = state.info.condition_id

            fv = self.fair_value_engine.evaluate(state)

            exit_signal = self.strategy.should_exit(state, fv, position)
            if exit_signal is not None:
                continue

            signal = self.strategy.evaluate(state, fv, position)
            if signal is None:
                continue

            approved, reason = self.risk_manager.check_order(
                market_id=state.info.condition_id,
                side=signal.side.value,
                size=signal.target_size,
                price=signal.target_price,
                price_feed_age_ms=0,
            )
            if not approved:
                continue

            orders = self.execution_engine.plan_orders(signal, state, position)
            for order in orders:
                book = state.book_up if order.side == OutcomeSide.UP else state.book_down
                fill = self.execution_engine.simulate_fill(order, book)
                if fill.filled:
                    position.add_fill(order.side, fill.fill_price, fill.fill_size)
                    result.total_trades += 1
                    result.trades.append(
                        TradeRecord(
                            timestamp_ms=state.timestamp_ms,
                            market_id=state.info.condition_id,
                            side=order.side.value,
                            action=order.action,
                            price=fill.fill_price,
                            size=fill.fill_size,
                            edge=signal.edge,
                            reason=signal.reason,
                        )
                    )

            result.pnl_curve.append(result.total_pnl)

        if position.paired_shares > 0 or position.directional_exposure > 0:
            if snapshots:
                last = snapshots[-1]
                pnl = self._resolve_position(last, position)
                bankroll += pnl
                result.total_pnl += pnl
                pnl_per_resolution.append(pnl)

        if len(pnl_per_resolution) >= 2:
            mean_pnl = sum(pnl_per_resolution) / len(pnl_per_resolution)
            var = sum((p - mean_pnl) ** 2 for p in pnl_per_resolution) / len(pnl_per_resolution)
            std = var**0.5
            result.sharpe_estimate = mean_pnl / std if std > 0 else 0.0

        return result

    @staticmethod
    def _resolve_position(state: MarketState, position: Position) -> float:
        """Determine PnL when market resolves."""
        if state.btc_above_open is True:
            winner = OutcomeSide.UP
        elif state.btc_above_open is False:
            winner = OutcomeSide.DOWN
        else:
            winner = OutcomeSide.UP

        return position.pnl_if_resolves(winner)
