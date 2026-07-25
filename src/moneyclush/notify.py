"""Web Push delivery for the handful of events worth interrupting someone for.

Everything else in this project accumulates quietly for later analysis —
these don't, because they close or matter within seconds to minutes:

    arbitrage    Up+Down pair under $1 after costs — closes in seconds
    edge         net edge past cost on a BTC 5m/15m window — same reason
    storage      persistence fell back to ephemeral — silent data loss
    favourite    a sports match's favourite crossed the alert threshold

VAPID identifies this server to the push service without a server-side
account: the private key signs a short-lived JWT on every send, and the
matching public key is what the browser embeds in the subscription it
creates. That coupling means the private key cannot be rotated for free —
every subscription made against the old public key stops working the
moment the key changes, and every browser has to re-subscribe. Generate
it once, keep it in `VAPID_PRIVATE_KEY`, never regenerate in place.

Without that variable set, a key is generated per-process for local
testing. It works until the next restart, at which point every
subscription silently starts failing (still POSTs fine on this end, the
push service just rejects the auth) until the browser re-subscribes.
"""

from __future__ import annotations

import base64
import os
import threading

import structlog
from py_vapid import Vapid
from pywebpush import WebPushException, webpush
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from moneyclush.data import store

log = structlog.get_logger()

_vapid: Vapid | None = None
_lock = threading.Lock()

# Push services return these when a subscription is gone for good — the
# user revoked permission, uninstalled the browser profile, or it simply
# expired. Retrying forever would mean paying an HTTP round trip for a
# dead endpoint on every single alert, permanently.
_DEAD_SUBSCRIPTION_CODES = {404, 410}


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _load_vapid() -> Vapid:
    """The signing keypair, generating one only if unset.

    `send_to_all` runs on a worker thread per call (dispatched via
    `asyncio.to_thread`), and several pushes can fire out of the same poll
    tick — a burst of sports favourites crossing 75% on the very first
    prediction cycle produced eleven of them at once. Without the lock,
    every one of those threads sees `_vapid is None` before any of them
    finishes, so each generates its *own* keypair: whichever one the
    public-key endpoint happens to hand the browser stops matching the
    key later pushes get signed with, and delivery fails with a 401 that
    looks like nothing is wrong.
    """
    global _vapid
    if _vapid is not None:
        return _vapid

    with _lock:
        if _vapid is not None:
            return _vapid

        v = Vapid()
        pem_env = os.environ.get("VAPID_PRIVATE_KEY")
        if pem_env:
            pem_bytes = pem_env.encode() if isinstance(pem_env, str) else pem_env
            # .env stores can't hold real newlines; tolerate the literal
            # `\n` a shell or Railway's variable editor is likely to produce.
            if b"\\n" in pem_bytes:
                pem_bytes = pem_bytes.replace(b"\\n", b"\n")
            v.from_pem(pem_bytes)
        else:
            v.generate_keys()
            log.warning(
                "notify.vapid_ephemeral",
                msg="VAPID_PRIVATE_KEY no definida — clave generada solo para "
                    "este proceso. Todas las suscripciones dejarán de "
                    "funcionar en el próximo reinicio.",
            )

        _vapid = v
        return _vapid


def public_key_b64url() -> str:
    """The key the browser embeds in `PushManager.subscribe()`."""
    v = _load_vapid()
    raw = v.public_key.public_bytes(
        Encoding.X962, PublicFormat.UncompressedPoint
    )
    return _b64url(raw)


def is_configured() -> bool:
    """Whether push has any subscriber to actually deliver to."""
    return bool(store.load_push_subscriptions())


def send_to_all(title: str, body: str, tag: str | None = None, url: str | None = None) -> int:
    """Deliver one notification to every stored subscription.

    Synchronous and network-bound — callers on the event loop should run
    this via `asyncio.to_thread`, not await it directly. Returns how many
    deliveries succeeded.
    """
    import json as _json

    # Passing the loaded Vapid object rather than its PEM string matters:
    # pywebpush only auto-detects PEM-with-headers when it's handed a file
    # *path*, and treats any other string as raw base64url DER. Handing it
    # a PEM string that way fails to decode — the object sidesteps that
    # ambiguity entirely and is what pywebpush's own examples pass.
    vapid_obj = _load_vapid()
    claims = {"sub": os.environ.get("VAPID_CLAIM_EMAIL", "mailto:admin@example.com")}
    payload = _json.dumps({"title": title, "body": body, "tag": tag, "url": url or "/"})

    delivered = 0
    for sub in store.load_push_subscriptions():
        subscription_info = {
            "endpoint": sub["endpoint"],
            "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
        }
        try:
            webpush(
                subscription_info=subscription_info,
                data=payload,
                vapid_private_key=vapid_obj,
                vapid_claims=dict(claims),
                ttl=300,
            )
            delivered += 1
        except WebPushException as exc:
            code = exc.response.status_code if exc.response is not None else None
            if code in _DEAD_SUBSCRIPTION_CODES:
                store.delete_push_subscription(sub["endpoint"])
                log.info("notify.subscription_expired", endpoint=sub["endpoint"][:60])
            else:
                log.warning("notify.push_failed", error=str(exc)[:160], status=code)
        except Exception as exc:
            log.warning("notify.push_failed", error=str(exc)[:160])

    return delivered
