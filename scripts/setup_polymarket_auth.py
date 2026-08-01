"""One-time setup: derive Polymarket CLOB V2 API credentials from your wallet.

Uses py-clob-client-v2. The V1 package cannot authenticate or sign orders
against the post-2026-04-28 CLOB — see execution/engine.py item 0.

Fill PRIVATE_KEY (and, for an email/Google Polymarket account,
POLYMARKET_SIGNATURE_TYPE + POLYMARKET_FUNDER_ADDRESS) in .env yourself
first — never paste those into chat. This script only reads .env locally
and writes the derived credentials back into .env; it never prints the
private key, api_secret, or api_passphrase to the terminal.

Usage:
    python scripts/setup_polymarket_auth.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv, set_key
from py_clob_client_v2.client import ClobClient

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
POLYGON_CHAIN_ID = 137


def main() -> None:
    if not ENV_PATH.exists():
        sys.exit(f"{ENV_PATH} not found — copy .env.example to .env first.")
    load_dotenv(ENV_PATH)

    private_key = os.environ.get("PRIVATE_KEY", "").strip()
    if not private_key:
        sys.exit("PRIVATE_KEY is empty in .env — fill it in, then re-run.")

    host = os.environ.get("POLYMARKET_CLOB_URL", "https://clob.polymarket.com")
    sig_type = os.environ.get("POLYMARKET_SIGNATURE_TYPE", "").strip()
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "").strip()

    kwargs: dict = {"host": host, "key": private_key, "chain_id": POLYGON_CHAIN_ID}
    if sig_type:
        kwargs["signature_type"] = int(sig_type)
    if funder:
        kwargs["funder"] = funder

    client = ClobClient(**kwargs)
    # V2 renamed this from create_or_derive_api_creds.
    creds = client.create_or_derive_api_key()

    set_key(str(ENV_PATH), "POLYMARKET_API_KEY", creds.api_key)
    set_key(str(ENV_PATH), "POLYMARKET_API_SECRET", creds.api_secret)
    set_key(str(ENV_PATH), "POLYMARKET_API_PASSPHRASE", creds.api_passphrase)

    print("Derived API credentials written to .env — nothing sensitive printed here.")
    print("POLYMARKET_API_KEY, POLYMARKET_API_SECRET, POLYMARKET_API_PASSPHRASE are set.")


if __name__ == "__main__":
    main()
