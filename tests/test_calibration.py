"""Tests for the numbers the dashboard draws conclusions from.

The failure these guard against is not a crash — it is a confident green
"+28.2 pts edge" computed off four bets, which is what the table used to
print. Every assertion here is about refusing to claim more than the
sample supports.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moneyclush import calibration  # noqa: E402
from moneyclush.calibration import Sample  # noqa: E402


def _samples(n, prob, ask, wins, volume=0.0):
    return [
        Sample(
            prob=prob, ask=ask, won=i < wins,
            pnl=(1 / ask - 1) if i < wins else -1.0,
            volume=volume,
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------- Wilson

def test_wilson_stays_inside_zero_one():
    """The normal approximation happily returns bounds below 0 or above 1
    at the edges, which is where these rates live."""
    lo, hi = calibration.wilson_interval(10, 10)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = calibration.wilson_interval(0, 5)
    assert 0.0 <= lo <= hi <= 1.0


def test_interval_narrows_with_sample_size():
    small = calibration.wilson_interval(7, 10)
    large = calibration.wilson_interval(700, 1000)
    assert (large[1] - large[0]) < (small[1] - small[0])


# -------------------------------------------------------------------- z / n

def test_z_uses_poisson_binomial_variance():
    """Each pick carries its own price, so the null is a sum of distinct
    Bernoullis. Collapsing to a binomial around the mean understates the
    variance whenever the prices are spread out — which they are."""
    spread = [
        Sample(prob=0.55, ask=0.57, won=True, pnl=0.75),
        Sample(prob=0.95, ask=0.96, won=True, pnl=0.04),
    ]
    z = calibration.calibration_z(spread)
    assert z is not None
    # expected wins 1.5, variance .2475+.0475 = .295 -> z = .5/sqrt(.295)
    assert abs(z - (0.5 / 0.295 ** 0.5)) < 1e-9


def test_large_gap_on_small_sample_is_not_significant():
    """Four bets at 72c that all won: a +28 point gap, and pure noise.

    This is the exact row the old table painted bright green.
    """
    row = calibration.summary(_samples(4, 0.72, 0.74, 4))
    assert row["edge_pts"] > 20
    assert row["significant"] is False
    assert "significativo" not in calibration.verdict(row)


def test_same_gap_on_a_real_sample_is_significant():
    row = calibration.summary(_samples(600, 0.60, 0.62, 420))
    assert row["significant"] is True
    assert "significativo" in calibration.verdict(row)


def test_verdict_refuses_to_read_tiny_samples():
    row = calibration.summary(_samples(12, 0.7, 0.72, 12))
    assert "muestra insuficiente" in calibration.verdict(row)


def test_verdict_estimates_the_sample_still_needed():
    row = calibration.summary(_samples(100, 0.60, 0.62, 66))
    text = calibration.verdict(row)
    assert not row["significant"]
    assert "harían falta" in text


# ------------------------------------------------------- prob vs ask baseline

def test_expected_rate_uses_prob_not_ask():
    """Scoring the market's forecast at the ask charges it half a spread
    and manufactures a deficit the size of the effect being measured."""
    row = calibration.summary(_samples(50, 0.60, 0.65, 30))
    assert row["expected_hit_rate"] == 0.6
    assert row["edge_pts"] == 0.0


def test_pnl_uses_ask():
    row = calibration.summary(_samples(10, 0.60, 0.50, 10))
    assert row["pnl_per_trade"] == 1.0


# ---------------------------------------------------------------- band split

def test_bands_cut_on_probability():
    samples = _samples(5, 0.55, 0.57, 3) + _samples(7, 0.85, 0.87, 6)
    rows = {r["band"]: r for r in calibration.calibration_rows(samples)}
    assert rows["50-60%"]["n"] == 5
    assert rows["80-90%"]["n"] == 7
    assert rows["60-70%"]["n"] == 0


def test_volume_bands():
    samples = _samples(3, 0.6, 0.62, 2, volume=900.0) + _samples(
        4, 0.6, 0.62, 3, volume=120_000.0
    )
    rows = {
        r["band"]: r
        for r in calibration.calibration_rows(
            samples, calibration.VOLUME_BANDS, by="volume"
        )
    }
    assert rows["<$5K"]["n"] == 3
    assert rows[">$50K"]["n"] == 4


def test_empty_input_is_reported_not_crashed():
    row = calibration.summary([])
    assert row["n"] == 0
    assert row["hit_rate"] is None
    assert row["significant"] is False
