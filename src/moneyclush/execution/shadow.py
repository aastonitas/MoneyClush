"""Shadow mode: for every real strategy signal, build and sign a real V2
order and run it through the guardrails — but never submit it.

This is the bridge between "the signing works" (verified 2026-08-01
against a live market, see execution/engine.py item 1) and an actual live
engine (roadmap items 2-4, still open). Running this continuously against
real signals, on real prices, proves the whole pipeline end to end before
anything is ever armed. `ClobClient.post_order` / `create_and_post_order`
must never be called from this module — that line is the entire point of
it existing separately from the eventual live engine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import structlog
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, OrderArgs
from py_clob_client_v2.order_builder.constants import BUY

from moneyclush.execution.guardrails import (
    KillSwitch,
    SafetyLimits,
    check_order_allowed,
    compute_order_size_usd,
    order_blockers,
)

log = structlog.get_logger()

# Confirmed live against BTC Up/Down 5m/15m order books on 2026-08-01 —
# see the "5 shares, not $5" finding that forced the switch from a flat
# dollar cap to %-of-balance sizing in guardrails.py.
MIN_ORDER_SHARES = 5


@dataclass
class ShadowResult:
    ts_ms: int
    side: str
    price: float
    shares: float
    order_usd: float
    reason: str
    allowed: bool
    why: str
    signed: bool
    # Would this order have passed everything *except* being armed? The
    # question that matters when deciding whether arming is worth it.
    would_pass_if_armed: bool = False
    other_blockers: str = ""
    maker: Optional[str] = None
    signer: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "ts": self.ts_ms,
            "side": self.side,
            "price": self.price,
            "shares": self.shares,
            "order_usd": self.order_usd,
            "reason": self.reason,
            "allowed": self.allowed,
            "why": self.why,
            "signed": self.signed,
            "would_pass_if_armed": self.would_pass_if_armed,
            "other_blockers": self.other_blockers,
            "maker": self.maker,
            "signer": self.signer,
            "error": self.error,
        }


def build_shadow_client(
    *, private_key: str, funder: str, signature_type: int,
    api_key: str = "", api_secret: str = "", api_passphrase: str = "",
    host: str = "https://clob.polymarket.com", chain_id: int = 137,
) -> ClobClient:
    """Client used for both signing (shadow) and submitting (live).

    The two need different auth levels, which is easy to get wrong in a
    way that only shows up at the worst moment: signing an order needs
    only the private key (L1), so shadow mode works fine without API
    credentials — but submitting needs L2, and its absence surfaced as
    "API Credentials are needed" on the very first armed signal rather
    than at startup. Hence `assert_ready_to_submit` below: the gap is
    now visible before arming, not after.
    """
    client = ClobClient(
        host=host, chain_id=chain_id, key=private_key,
        signature_type=signature_type, funder=funder,
    )
    if api_key and api_secret and api_passphrase:
        client.set_api_creds(
            ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        )
    return client


def assert_ready_to_submit(client: ClobClient) -> tuple[bool, str]:
    """Can this client actually submit an order, or only sign one?

    Returns (ready, reason). Cheap and offline — checks the credentials
    are present, not that the venue will accept them.
    """
    creds = getattr(client, "creds", None)
    if not creds or not getattr(creds, "api_key", ""):
        return False, "sin credenciales de API — el cliente puede firmar pero no enviar"
    return True, "credenciales de API presentes"


def run_shadow_order(
    *,
    client: ClobClient,
    token_id: str,
    side_label: str,
    price: float,
    reason: str,
    kill_switch: KillSwitch,
    limits: SafetyLimits,
    on_chain_balance_usd: Optional[float],
    seconds_remaining: float,
) -> ShadowResult:
    """Size, guardrail-check, and sign one order. Never posts it.

    Sizing and the guardrail check happen even when the balance is
    unavailable or the price is unusable, so the caller always gets a
    result to log — silently skipping would look identical to "nothing
    happened," which is exactly the failure mode this exists to avoid.
    """
    price = round(price, 2)
    target_usd = (
        compute_order_size_usd(on_chain_balance_usd, limits)
        if on_chain_balance_usd is not None else 0.0
    )
    # Floor, not round — the same reasoning as guardrails._floor_cents.
    # Rounding to the nearest share can overshoot the dollar target (12
    # shares @ 30c = $3.60 against a $3.55 target); flooring can only
    # undershoot it or exactly hit the exchange's MIN_ORDER_SHARES floor,
    # which is the correct outcome when even the minimum lot is too big
    # for the target at this price — that case should be rejected by the
    # guardrail below, not silently rounded past it.
    shares = max(MIN_ORDER_SHARES, math.floor(target_usd / price)) if price > 0 else MIN_ORDER_SHARES
    order_usd = round(shares * price, 4)

    # Nothing live has ever opened a position or lost money yet, so these
    # are placeholders — see engine.py roadmap item 4 for where they stop
    # being placeholders.
    common = dict(
        kill_switch=kill_switch,
        limits=limits,
        on_chain_balance_usd=on_chain_balance_usd,
        open_exposure_usd=0.0,
        daily_realized_pnl_usd=0.0,
        orders_in_last_hour=0,
        seconds_remaining=seconds_remaining,
    )
    allowed, why = check_order_allowed(order_usd, **common)

    # Same order, evaluated as if arming were not in question. This is what
    # tells you whether arming would actually have changed the outcome.
    other = order_blockers(order_usd, include_arming=False, **common)

    result = ShadowResult(
        ts_ms=int(time.time() * 1000), side=side_label, price=price,
        shares=shares, order_usd=order_usd, reason=reason,
        allowed=allowed, why=why, signed=False,
        would_pass_if_armed=not other,
        other_blockers="; ".join(other),
    )

    try:
        args = OrderArgs(token_id=token_id, price=price, size=shares, side=BUY)
        signed = client.create_order(args)
        result.signed = True
        result.maker = signed.maker
        result.signer = signed.signer
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"[:160]
        log.warning("shadow.sign_failed", error=result.error)

    return result
