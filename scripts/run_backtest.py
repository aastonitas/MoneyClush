"""Run a backtest of the temporal arbitrage strategy with synthetic data.

Usage:
    python scripts/run_backtest.py

Generates realistic BTC Up/Down market scenarios and evaluates performance.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush.data.models import (
    MarketInfo,
    MarketState,
    OrderBookLevel,
    OrderBookSnapshot,
    OutcomeSide,
)
from moneyclush.research.backtester import Backtester
from moneyclush.strategies.temporal_arbitrage import TemporalArbitrageStrategy


def generate_synthetic_market(
    market_id: str,
    opening_price: float,
    num_snapshots: int = 60,
    duration_minutes: int = 5,
) -> list[MarketState]:
    """Generate a sequence of market snapshots simulating a 5m window."""
    snapshots = []
    btc = opening_price
    window_ms = duration_minutes * 60 * 1000
    interval_ms = window_ms // num_snapshots
    open_ts = 1700000000000
    close_ts = open_ts + window_ms

    for i in range(num_snapshots):
        btc += random.gauss(0, opening_price * 0.0003)

        ts = open_ts + i * interval_ms
        above_open = btc > opening_price
        pct_above = (btc - opening_price) / opening_price

        p_up = 0.5 + pct_above * 50
        p_up = max(0.05, min(0.95, p_up))
        p_down = 1 - p_up

        noise = random.gauss(0, 0.03)
        ask_up = max(0.02, min(0.98, p_up + 0.01 + noise))
        bid_up = max(0.01, ask_up - random.uniform(0.01, 0.04))
        ask_down = max(0.02, min(0.98, p_down + 0.01 - noise * 0.5))
        bid_down = max(0.01, ask_down - random.uniform(0.01, 0.04))

        def make_book(bid: float, ask: float) -> OrderBookSnapshot:
            return OrderBookSnapshot(
                bids=[
                    OrderBookLevel(price=round(bid, 4), size=random.uniform(100, 500)),
                    OrderBookLevel(price=round(bid - 0.02, 4), size=random.uniform(200, 800)),
                ],
                asks=[
                    OrderBookLevel(price=round(ask, 4), size=random.uniform(100, 500)),
                    OrderBookLevel(price=round(ask + 0.02, 4), size=random.uniform(200, 800)),
                ],
                timestamp_ms=ts,
            )

        snapshots.append(
            MarketState(
                info=MarketInfo(
                    condition_id=market_id,
                    token_id_up=f"{market_id}-up",
                    token_id_down=f"{market_id}-down",
                    question=f"BTC Up or Down 5m #{market_id}",
                    duration_minutes=duration_minutes,
                    opening_price_btc=opening_price,
                    open_timestamp_ms=open_ts,
                    close_timestamp_ms=close_ts,
                ),
                book_up=make_book(bid_up, ask_up),
                book_down=make_book(bid_down, ask_down),
                recent_trades=[],
                btc_price=btc,
                btc_price_timestamp_ms=ts,
                timestamp_ms=ts,
            )
        )

    return snapshots


def main():
    random.seed(42)

    print("=" * 60)
    print("MoneyClush — Temporal Arbitrage Backtest")
    print("=" * 60)

    all_snapshots: list[MarketState] = []
    num_markets = 50
    btc_base = 67500.0

    print(f"\nGenerating {num_markets} synthetic BTC Up/Down 5m markets...")

    for i in range(num_markets):
        opening = btc_base + random.gauss(0, 200)
        snaps = generate_synthetic_market(
            market_id=f"market-{i:03d}",
            opening_price=opening,
            num_snapshots=60,
        )
        all_snapshots.extend(snaps)

    print(f"Total snapshots: {len(all_snapshots)}")

    strategy = TemporalArbitrageStrategy(
        max_pair_cost=0.96,
        block_size=50,
        min_edge_per_pair=0.02,
        max_single_side_price=0.48,
    )

    backtester = Backtester(
        strategy=strategy,
        initial_bankroll=1000.0,
    )

    print("\nRunning backtest...")
    result = backtester.run(all_snapshots)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    summary = result.summary()
    for key, value in summary.items():
        print(f"  {key:>20}: {value}")

    if result.trades:
        print(f"\n  Sample trades (first 10):")
        for t in result.trades[:10]:
            print(
                f"    {t.market_id} | {t.side:>4} {t.action:>4} "
                f"@ {t.price:.4f} x{t.size:.0f} | edge={t.edge:.4f} | {t.reason}"
            )

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
