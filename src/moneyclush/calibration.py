"""Calibration tables, with the uncertainty that makes them readable.

A hit rate without a sample size is not evidence, it is decoration. The
whole question this project exists to answer — does the market misprice
favourites, and by how much — lives at the 2-4 point level, which is
inside the noise of any sample smaller than a few hundred bets. A table
that paints "+30.0" in green off three observations actively misleads.

So every row carries three things the raw hit rate does not:

    n                how much to believe it at all
    ci_low/ci_high   Wilson interval on the realised rate
    z                how far the realised rate sits from what was priced

`z` is a Poisson-binomial score, not a plain binomial one: each pick had
its own market probability, so the expected number of wins is the sum of
those probabilities and the variance is the sum of p(1-p). That is the
correct null for "the market's prices were honest".

Two prices per pick matter and they are not interchangeable:

    prob   the de-vigged market probability — the hypothesis variable
    ask    what a share actually cost — the PnL variable

Banding on `ask` conflates the two: asks sit half a spread above the
market's real estimate, so the expected hit rate comes out inflated and
every band looks like it underperformed by half a spread. Bands are cut
on `prob`; `ask` is only ever used for money.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Where the BTC backtest put the suspected favourite-overpricing bias:
# concentrated in 0.60-0.90, not spread evenly. A single aggregate number
# hides exactly the thing worth knowing.
PROBABILITY_BANDS = (
    (0.50, 0.60, "50-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90%+"),
)

# Thin books are where stale prices survive; deep ones are already
# arbitraged. If a favourite bias exists anywhere it should show up here
# first, and this split costs nothing — the volume is already fetched.
VOLUME_BANDS = (
    (0.0, 5_000.0, "<$5K"),
    (5_000.0, 50_000.0, "$5-50K"),
    (50_000.0, float("inf"), ">$50K"),
)

# Two-sided 95%. Below this a row is noise and must not be coloured as
# though it were a finding.
SIGNIFICANCE_Z = 1.96


@dataclass(frozen=True)
class Sample:
    """One settled bet, reduced to the four numbers calibration needs."""

    prob: float      # de-vigged market probability at entry
    ask: float       # price paid per share
    won: bool
    pnl: float
    volume: float = 0.0


def wilson_interval(wins: int, n: int, z: float = SIGNIFICANCE_Z) -> tuple[float, float]:
    """95% confidence interval for a proportion, Wilson score method.

    Wilson rather than the normal approximation because the samples here
    are small and the rates are near the edges, where `p ± z*sqrt(pq/n)`
    happily returns bounds outside [0, 1].
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


def calibration_z(samples: list[Sample]) -> float | None:
    """How many standard deviations the realised wins sit from the priced ones.

    Under "the market is honest", the number of wins is Poisson-binomial
    over the individual probabilities: mean = sum(p), var = sum(p(1-p)).
    A plain binomial around the *average* probability would understate
    the variance whenever the prices are spread out, which they are.
    """
    if not samples:
        return None
    expected = sum(s.prob for s in samples)
    variance = sum(s.prob * (1.0 - s.prob) for s in samples)
    if variance <= 0:
        return None
    wins = sum(1 for s in samples if s.won)
    return (wins - expected) / math.sqrt(variance)


def _row(label: str, bucket: list[Sample]) -> dict:
    n = len(bucket)
    if not n:
        return {
            "band": label, "n": 0, "hit_rate": None, "ci_low": None,
            "ci_high": None, "expected_hit_rate": None, "edge_pts": None,
            "z": None, "significant": False, "pnl": None, "pnl_per_trade": None,
        }
    wins = sum(1 for s in bucket if s.won)
    hit = wins / n
    expected = sum(s.prob for s in bucket) / n
    pnl = sum(s.pnl for s in bucket)
    lo, hi = wilson_interval(wins, n)
    z = calibration_z(bucket)
    return {
        "band": label,
        "n": n,
        "hit_rate": round(hit, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "expected_hit_rate": round(expected, 4),
        "edge_pts": round((hit - expected) * 100, 2),
        "z": round(z, 2) if z is not None else None,
        "significant": bool(z is not None and abs(z) >= SIGNIFICANCE_Z),
        "pnl": round(pnl, 4),
        "pnl_per_trade": round(pnl / n, 4),
    }


def calibration_rows(
    samples: list[Sample], bands: tuple = PROBABILITY_BANDS, by: str = "prob"
) -> list[dict]:
    """One row per band, cut on `prob` (default) or `volume`."""
    rows = []
    for lo, hi, label in bands:
        bucket = [s for s in samples if lo <= getattr(s, by) < hi]
        rows.append(_row(label, bucket))
    return rows


def summary(samples: list[Sample]) -> dict:
    """Aggregate hit rate, expectation and PnL with the uncertainty attached."""
    n = len(samples)
    if not n:
        return {
            "n": 0, "wins": 0, "hit_rate": None, "ci_low": None, "ci_high": None,
            "expected_hit_rate": None, "edge_pts": None, "z": None,
            "significant": False, "pnl": 0.0, "pnl_per_trade": None,
        }
    row = _row("total", samples)
    row["wins"] = sum(1 for s in samples if s.won)
    return row


def verdict(row: dict) -> str:
    """Plain-language reading of a calibration row, honest about sample size.

    The failure mode this guards against is reading a 9-point gap off 32
    bets as a discovery. Until `z` clears the bar the only correct
    statement is that nothing has been shown yet.
    """
    n = row.get("n") or 0
    if n < 30:
        return f"muestra insuficiente ({n}) — sin lectura posible"
    z = row.get("z")
    edge = row.get("edge_pts")
    if z is None or edge is None:
        return f"muestra insuficiente ({n}) — sin lectura posible"
    if abs(z) < SIGNIFICANCE_Z:
        needed = math.ceil(n * (SIGNIFICANCE_Z / abs(z)) ** 2) if z else None
        extra = f" · harían falta ~{needed} para concluir" if needed else ""
        return (
            f"{edge:+.1f} pts sobre lo cotizado, z={z:+.2f} — "
            f"indistinguible de ruido{extra}"
        )
    direction = "MÁS" if edge > 0 else "MENOS"
    return (
        f"aciertan {abs(edge):.1f} pts {direction} de lo que cotizaban "
        f"(z={z:+.2f}, n={n}) — significativo al 95%"
    )
