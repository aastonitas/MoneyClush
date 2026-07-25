"""Tests for the two rules that decide whether a pick ever gets counted.

Both bugs these cover were silent: the ledger simply stayed at "pending"
forever and the dashboard kept showing a hit rate computed off whatever
fraction of the sample happened to escape. Nothing errored, so nothing
was noticed until the numbers were checked against the API by hand.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush.data.predictions import Prediction, _decide  # noqa: E402


def _market(group, prices, outcomes=("Yes", "No"), closed=False, uma=None):
    return {
        "groupItemTitle": group,
        "outcomes": json.dumps(list(outcomes)),
        "outcomePrices": json.dumps([str(p) for p in prices]),
        "closed": closed,
        "umaResolutionStatus": uma,
    }


def _event(markets, teams=("Top Esports", "ThunderTalk Gaming"), closed=False):
    return {
        "closed": closed,
        "teams": [{"name": t} for t in teams],
        "markets": markets,
    }


# --------------------------------------------------------------- resolution

def test_settles_on_market_flag_while_event_still_open():
    """The event-level `closed` flag trails the result by hours or days.

    Measured against live Gamma data, 138 of 139 pending picks had
    `closed=False` while a third of them already had a settled result
    market. Gating on the event flag is what stranded them.
    """
    event = _event(
        [_market("Top Esports", [1.0, 0.0], closed=True)],
        closed=False,
    )
    assert _decide(event) == ("Top Esports", False)


def test_settles_on_uma_proposal():
    event = _event(
        [_market("Top Esports", [0.9995, 0.0005], closed=False, uma="proposed")]
    )
    assert _decide(event) == ("Top Esports", False)


def test_does_not_settle_on_price_alone():
    """99c on an unsettled market is the market forecasting, not a result.

    Counting it would score the ledger against the very prices it exists
    to test, and would book a win before the fixture had finished.
    """
    event = _event([_market("Top Esports", [0.9995, 0.0005], closed=False)])
    assert _decide(event) == (None, False)


def test_voided_market_is_not_a_loss():
    """A cancelled fixture settles every side at 0.5 and returns the stake."""
    event = _event(
        [_market("Lazer Cats", [0.5, 0.5], closed=True)],
        teams=("Lazer Cats", "Inner Circle"),
    )
    assert _decide(event) == (None, True)


def test_draw_is_recognised():
    event = _event(
        [
            _market("Montrose FC", [0.0, 1.0], closed=True),
            _market("Draw", [1.0, 0.0], closed=True),
        ],
        teams=("Montrose FC", "Spartans FC"),
    )
    assert _decide(event) == ("Empate", False)


def test_moneyline_shape():
    event = _event(
        [
            _market(
                None, [0.0, 1.0],
                outcomes=("Ismael Bonfim", "Axel Sola"),
                closed=True,
            )
        ],
        teams=("Ismael Bonfim", "Axel Sola"),
    )
    assert _decide(event) == ("Axel Sola", False)


def test_prop_markets_are_ignored():
    """A UFC card carries method-of-victory and round props on the same
    event; settling off one of those would record the wrong winner."""
    event = _event(
        [
            _market("Fight to Go the Distance", [1.0, 0.0], closed=True),
            _market("Fight won by KO", [1.0, 0.0], closed=True),
        ],
        teams=("Rizvan Kuniev", "Tyrell Fortune"),
    )
    assert _decide(event) == (None, False)


# -------------------------------------------------------------- queue health

def _pending(**kw) -> Prediction:
    base = dict(
        event_id="1", title="t", url="", league="", discipline="",
        pick="A", pick_prob=0.6, pick_ask=0.62,
        recorded_at=datetime.now(timezone.utc).isoformat(),
    )
    base.update(kw)
    return Prediction(**base)


def test_untried_pick_is_due_immediately():
    assert _pending().next_attempt_due() <= datetime.now(timezone.utc)


def test_backoff_grows_with_attempts():
    """A fixture that will never settle used to hold its slot in every
    batch forever, hiding everything queued behind it."""
    now = datetime.now(timezone.utc)
    early = _pending(attempts=1, last_attempt_at=now.isoformat())
    late = _pending(attempts=8, last_attempt_at=now.isoformat())
    assert early.next_attempt_due() < late.next_attempt_due()
    assert late.next_attempt_due() > now + timedelta(hours=1)


def test_backoff_is_capped():
    now = datetime.now(timezone.utc)
    stuck = _pending(attempts=40, last_attempt_at=now.isoformat())
    assert stuck.next_attempt_due() <= now + timedelta(hours=6, seconds=1)


def test_stale_fixture_is_abandonable():
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    assert _pending(start_time=old).is_stale()
    assert not _pending(
        start_time=datetime.now(timezone.utc).isoformat()
    ).is_stale()


def test_settled_states_leave_the_queue():
    assert not _pending().settled
    assert _pending(void=True).settled
    assert _pending(abandoned=True).settled
    assert _pending(resolved=True, won=True).settled
