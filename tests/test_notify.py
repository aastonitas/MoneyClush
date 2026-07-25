"""Tests for the Web Push signing key lifecycle.

The bug these guard against was invisible in a single-threaded run: a
burst of concurrent calls to `_load_vapid` — exactly what happens when a
dozen sports favourites cross 75% on the same prediction-loop tick, each
dispatched to its own worker thread — raced past the "already loaded"
check before any of them finished, and each generated its *own* keypair.
Whichever key the public-key endpoint happened to serve the browser
stopped matching whatever key later pushes were signed with, and every
delivery after the first failed with an auth error that looked like
nothing was wrong.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush import notify  # noqa: E402


def test_concurrent_load_returns_the_same_key():
    notify._vapid = None
    results = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()  # line every thread up so they hit the check together
        results.append(notify._load_vapid())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    assert len({id(v) for v in results}) == 1, (
        "concurrent callers generated distinct keypairs instead of sharing one"
    )


def test_public_key_is_a_valid_uncompressed_point():
    """Browsers reject an applicationServerKey that isn't 65 raw bytes
    starting with 0x04 — the format `PushManager.subscribe` expects."""
    import base64

    notify._vapid = None
    key = notify.public_key_b64url()
    padded = key + "=" * (-len(key) % 4)
    raw = base64.urlsafe_b64decode(padded)
    assert len(raw) == 65
    assert raw[0] == 0x04


def test_public_key_stable_across_calls():
    notify._vapid = None
    first = notify.public_key_b64url()
    second = notify.public_key_b64url()
    assert first == second
