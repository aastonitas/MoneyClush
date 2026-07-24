"""Download real historical BTC Up/Down windows for backtesting.

Usage:
    python scripts/fetch_history.py --windows 300
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from moneyclush.research.historical import fetch_history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--duration", type=int, default=5, choices=[5, 15])
    ap.add_argument("--asset", type=str, default="btc")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        ROOT / "data" / "history" / f"{args.asset}_{args.duration}m.jsonl"
    )

    windows = asyncio.run(
        fetch_history(
            num_windows=args.windows,
            asset=args.asset,
            duration_min=args.duration,
            out_path=out,
        )
    )

    if not windows:
        print("No se descargaron ventanas.")
        return

    with_book = sum(1 for w in windows if w.up_prices)
    with_btc = sum(1 for w in windows if w.btc_open and w.btc_close)
    up_wins = sum(1 for w in windows if w.winner == "Up")

    print()
    print(f"  ventanas totales:      {len(windows)}")
    print(f"  con precios de mercado:{with_book}")
    print(f"  con precio BTC open/close: {with_btc}")
    print(f"  ganó Up:               {up_wins} ({up_wins/len(windows)*100:.1f}%)")
    print(f"  rango: {windows[0].slug} .. {windows[-1].slug}")


if __name__ == "__main__":
    main()
