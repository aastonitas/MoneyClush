"""Main bot loop: data -> signal -> fair value -> edge -> position -> execution -> risk.

This is the production entry point. In paper mode, it connects to real data
but simulates execution. In backtest mode, use the Backtester instead.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from config.settings import Settings, TradingMode, load_settings
from moneyclush.data.models import MarketInfo, OutcomeSide, Position
from moneyclush.data.polymarket_client import PolymarketClient
from moneyclush.data.price_feeds import PriceFeed
from moneyclush.execution.engine import ExecutionEngine
from moneyclush.pricing.fair_value import FairValueEngine
from moneyclush.risk.manager import RiskManager
from moneyclush.strategies.base import SignalAction
from moneyclush.strategies.temporal_arbitrage import TemporalArbitrageStrategy

log = structlog.get_logger()


async def run_bot(settings: Settings | None = None) -> None:
    if settings is None:
        settings = load_settings()

    log.info("bot.starting", mode=settings.mode.value)

    strategy = TemporalArbitrageStrategy()
    fv_engine = FairValueEngine()
    exec_engine = ExecutionEngine()
    risk_mgr = RiskManager(
        max_bankroll_usd=settings.risk.max_bankroll_usd,
        kelly_fraction=settings.risk.kelly_fraction,
        max_loss_daily_usd=settings.risk.max_loss_daily_usd,
        max_position_per_market=settings.risk.max_position_per_market,
        max_uncovered_shares=settings.risk.max_uncovered_shares,
        max_correlated_positions=settings.risk.max_correlated_positions,
    )

    price_feed = PriceFeed()
    price_task = asyncio.create_task(price_feed.start())

    await asyncio.sleep(2)
    if price_feed.latest is None:
        log.info("bot.fetching_initial_price")
        await price_feed.fetch_once()

    client = PolymarketClient(
        base_url=settings.polymarket.clob_url,
        api_key=settings.polymarket.api_key,
        api_secret=settings.polymarket.api_secret,
        api_passphrase=settings.polymarket.api_passphrase,
    )

    positions: dict[str, Position] = {}

    async with client:
        log.info("bot.connected")

        while True:
            try:
                markets = await client.get_markets("btc-up-or-down")
                if not markets:
                    log.warning("bot.no_markets_found")
                    await asyncio.sleep(10)
                    continue

                for market_data in markets[:3]:
                    market_info = MarketInfo(
                        condition_id=market_data.get("condition_id", ""),
                        token_id_up=market_data.get("tokens", [{}])[0].get("token_id", ""),
                        token_id_down=market_data.get("tokens", [{}, {}])[1].get("token_id", "") if len(market_data.get("tokens", [])) > 1 else "",
                        question=market_data.get("question", ""),
                        duration_minutes=5,
                    )

                    if not market_info.token_id_up or not market_info.token_id_down:
                        continue

                    btc = price_feed.price
                    btc_ts = price_feed.latest.timestamp_ms if price_feed.latest else 0

                    state = await client.build_market_state(market_info, btc, btc_ts)

                    if state.seconds_remaining <= 0:
                        continue

                    fv = fv_engine.evaluate(state)

                    pos = positions.get(
                        market_info.condition_id,
                        Position(market_condition_id=market_info.condition_id),
                    )

                    exit_signal = strategy.should_exit(state, fv, pos)
                    if exit_signal is not None:
                        log.warning("bot.force_exit", signal=exit_signal.reason)
                        continue

                    signal = strategy.evaluate(state, fv, pos)
                    if signal is None:
                        continue

                    approved, reason = risk_mgr.check_order(
                        market_id=market_info.condition_id,
                        side=signal.side.value,
                        size=signal.target_size,
                        price=signal.target_price,
                        price_feed_age_ms=price_feed.age_ms,
                    )

                    if not approved:
                        log.info("bot.risk_rejected", reason=reason)
                        continue

                    orders = exec_engine.plan_orders(signal, state, pos)

                    for order in orders:
                        if settings.mode == TradingMode.PAPER:
                            book = state.book_up if order.side == OutcomeSide.UP else state.book_down
                            fill = exec_engine.simulate_fill(order, book)
                            if fill.filled:
                                pos.add_fill(order.side, fill.fill_price, fill.fill_size)
                                log.info(
                                    "bot.paper_fill",
                                    side=order.side.value,
                                    price=f"{fill.fill_price:.4f}",
                                    size=fill.fill_size,
                                    slippage=f"{fill.slippage:.4f}",
                                )
                        elif settings.mode == TradingMode.LIVE:
                            token_id = (
                                market_info.token_id_up
                                if order.side == OutcomeSide.UP
                                else market_info.token_id_down
                            )
                            result = await client.place_limit_order(
                                token_id=token_id,
                                side=order.action,
                                price=order.limit_price,
                                size=order.size,
                                order_type=order.order_type,
                            )
                            log.info("bot.live_order", result=result)

                    positions[market_info.condition_id] = pos
                    risk_mgr.register_position(pos)

                await asyncio.sleep(2)

            except KeyboardInterrupt:
                log.info("bot.shutting_down")
                break
            except Exception as exc:
                log.error("bot.error", error=str(exc))
                await asyncio.sleep(5)

    price_feed.stop()
    price_task.cancel()


if __name__ == "__main__":
    asyncio.run(run_bot())
