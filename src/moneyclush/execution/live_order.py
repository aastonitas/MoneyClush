"""The one and only place `ClobClient.post_order` is ever called.

Kept in its own module, separate from shadow.py, on purpose: "shadow.py
never submits an order" stays true by construction — nobody has to trust
that a shared function got its branches right — and there is exactly one
file in this codebase that a reviewer has to trust not to fire when it
shouldn't. Everything upstream of this module (the strategy, shadow mode,
the dashboard) can be wrong in a hundred small ways and the worst outcome
is a bad log line; this module being wrong is the only way real money
moves.

This function does not decide whether an order is allowed. It takes
`allowed` as a plain bool, produced once by
`guardrails.check_order_allowed()`, and obeys it. That function returns
False for everything unless a human has explicitly created
`data/TRADING_ARMED` — which nothing in this codebase does on its own,
and which an AI assistant operating this repository must never create on
a user's behalf, including for testing this module. Testing here uses a
fake client with a stub `post_order` that never touches the network — the
real `ClobClient.post_order` is exercised for the first time only when a
human arms trading and a real signal fires, not before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import structlog
from py_clob_client_v2.clob_types import OrderArgs
from py_clob_client_v2.order_builder.constants import BUY

log = structlog.get_logger()


@dataclass
class LiveOrderResult:
    ts_ms: int
    side: str
    price: float
    shares: float
    order_usd: float
    submitted: bool
    reason: str
    order_id: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ts": self.ts_ms,
            "side": self.side,
            "price": self.price,
            "shares": self.shares,
            "order_usd": self.order_usd,
            "submitted": self.submitted,
            "reason": self.reason,
            "order_id": self.order_id,
            "error": self.error,
        }


def submit_order_if_allowed(
    *,
    client,
    token_id: str,
    side_label: str,
    price: float,
    shares: float,
    order_usd: float,
    allowed: bool,
    why: str,
) -> LiveOrderResult:
    """Sign and submit a real order — but only when `allowed` is True.

    `allowed`/`why` must come from `guardrails.check_order_allowed()`,
    called by the caller against the exact same order this function is
    about to sign. This function re-derives nothing about eligibility; it
    is a dumb, auditable gate, not a second opinion.
    """
    result = LiveOrderResult(
        ts_ms=int(time.time() * 1000), side=side_label, price=price,
        shares=shares, order_usd=order_usd, submitted=False, reason=why,
    )
    if not allowed:
        return result

    try:
        args = OrderArgs(token_id=token_id, price=round(price, 2), size=shares, side=BUY)
        signed = client.create_order(args)
        resp = client.post_order(signed)
        result.submitted = True
        result.order_id = (
            resp.get("orderID") or resp.get("id") if isinstance(resp, dict) else None
        )
        log.warning(
            "live.order_submitted",
            token_id=token_id[:16], side=side_label, price=price, shares=shares,
        )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:200]
        log.error("live.submit_failed", error=result.error)

    return result
