"""Collect live market data from Polymarket for research.

Saves order book snapshots, trades, and BTC prices to CSV files
for offline backtesting.

Usage:
    python scripts/collect_data.py --duration 3600
"""

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush.data.polymarket_client import PolymarketClient
from moneyclush.data.price_feeds import PriceFeed


async def collect(duration_sec: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    books_file = output_dir / f"books_{int(time.time())}.csv"
    prices_file = output_dir / f"btc_prices_{int(time.time())}.csv"

    price_feed = PriceFeed()
    price_task = asyncio.create_task(price_feed.start())

    await asyncio.sleep(2)
    if price_feed.latest is None:
        await price_feed.fetch_once()

    client = PolymarketClient()

    books_writer = csv.writer(open(books_file, "w", newline=""))
    books_writer.writerow([
        "timestamp_ms", "market_id", "side",
        "best_bid", "best_ask", "bid_volume", "ask_volume",
    ])

    prices_writer = csv.writer(open(prices_file, "w", newline=""))
    prices_writer.writerow(["timestamp_ms", "btc_price"])

    start = time.time()
    tick = 0

    print(f"Collecting data for {duration_sec}s -> {output_dir}")

    async with client:
        while time.time() - start < duration_sec:
            try:
                markets = await client.get_markets()

                for m in markets[:5]:
                    tokens = m.get("tokens", [])
                    if len(tokens) < 2:
                        continue

                    cid = m.get("condition_id", "")

                    for idx, label in [(0, "UP"), (1, "DOWN")]:
                        tid = tokens[idx].get("token_id", "")
                        if not tid:
                            continue
                        book = await client.get_order_book(tid)
                        books_writer.writerow([
                            int(time.time() * 1000),
                            cid,
                            label,
                            book.best_bid or 0,
                            book.best_ask or 0,
                            book.total_bid_volume(),
                            book.total_ask_volume(),
                        ])

                if price_feed.latest:
                    prices_writer.writerow([
                        price_feed.latest.timestamp_ms,
                        price_feed.price,
                    ])

                tick += 1
                if tick % 10 == 0:
                    elapsed = int(time.time() - start)
                    print(f"  [{elapsed}s] {tick} snapshots collected")

                await asyncio.sleep(5)

            except Exception as exc:
                print(f"  Error: {exc}")
                await asyncio.sleep(10)

    price_feed.stop()
    price_task.cancel()
    print(f"\nData saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Collect Polymarket data")
    parser.add_argument("--duration", type=int, default=3600, help="Seconds to collect")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    asyncio.run(collect(args.duration, Path(args.output)))


if __name__ == "__main__":
    main()
