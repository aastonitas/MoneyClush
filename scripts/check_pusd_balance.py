"""Read-only pUSD balance check, bypassing py-clob-client's stale collateral address.

py-clob-client 0.34.6 (latest on PyPI as of 2026-02) still hardcodes USDC.e
as the Polygon collateral token; Polymarket has since migrated to its own
pUSD token, so `client.get_balance_allowance()` reads $0 for accounts that
actually hold pUSD. This script reads the pUSD ERC-20 balance directly from
a public Polygon RPC instead — no private key needed, no order signing,
nothing sensitive.

Usage:
    python scripts/check_pusd_balance.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PUSD_CONTRACT = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
PUSD_DECIMALS = 6
PUBLIC_RPC = "https://polygon-bor-rpc.publicnode.com"


def main() -> None:
    load_dotenv(ENV_PATH)
    address = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()
    if not address:
        sys.exit("POLYMARKET_FUNDER_ADDRESS is empty in .env.")

    data = "0x70a08231" + "000000000000000000000000" + address.removeprefix("0x").lower()
    resp = httpx.post(
        PUBLIC_RPC,
        json={
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": PUSD_CONTRACT, "data": data}, "latest"],
            "id": 1,
        },
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        sys.exit(f"RPC error: {result['error']}")

    raw = int(result["result"], 16)
    balance = raw / 10**PUSD_DECIMALS
    print(f"{address}: {balance:.6f} pUSD")


if __name__ == "__main__":
    main()
