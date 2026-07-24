"""Validate the fair value model against real resolved windows.

The question a backtest must answer for this strategy is not "would it
have made money" — that can be produced by luck or by a bug. It is:

    Is our model better calibrated than the market we intend to trade?

If the market's own price predicts outcomes at least as well as our
model, then every apparent edge is our error, and trading it loses money
at exactly the rate we think we are winning.

Calibration is measured with the Brier score (mean squared error of a
probability forecast; lower is better) and with a reliability table: of
the times a forecast said 30%, did the event happen 30% of the time?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from moneyclush.pricing.fair_value import brownian_probability
from moneyclush.research.historical import HistoricalWindow


@dataclass
class Observation:
    """One point in time inside a resolved window."""

    slug: str
    seconds_remaining: float
    distance_bps: float
    model_p_up: float
    market_p_up: float
    realized_up: int

    @property
    def edge(self) -> float:
        """Model minus market, positive when the model thinks Up is cheap."""
        return self.model_p_up - self.market_p_up


@dataclass
class CalibrationBucket:
    low: float
    high: float
    count: int = 0
    predicted_sum: float = 0.0
    realized_sum: int = 0

    @property
    def predicted(self) -> float:
        return self.predicted_sum / self.count if self.count else 0.0

    @property
    def realized(self) -> float:
        return self.realized_sum / self.count if self.count else 0.0

    @property
    def gap(self) -> float:
        return self.realized - self.predicted


def brier_score(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs)


def calibration_table(
    pairs: list[tuple[float, int]], buckets: int = 10
) -> list[CalibrationBucket]:
    table = [
        CalibrationBucket(low=i / buckets, high=(i + 1) / buckets)
        for i in range(buckets)
    ]
    for p, outcome in pairs:
        idx = min(int(p * buckets), buckets - 1)
        b = table[idx]
        b.count += 1
        b.predicted_sum += p
        b.realized_sum += outcome
    return table


def build_observations(
    windows: list[HistoricalWindow],
    sigma_per_minute: float,
    min_seconds_remaining: float = 30.0,
) -> list[Observation]:
    """Reconstruct model and market forecasts at each minute of each window.

    Only points where both a BTC price and a market price are available
    are kept; nothing is interpolated or assumed.
    """
    obs: list[Observation] = []

    for w in windows:
        if not w.btc_open or w.btc_open <= 0 or not w.btc_path or not w.up_prices:
            continue

        # market price of Up, indexed by minute bucket
        market_by_minute: dict[int, float] = {}
        for pt in w.up_prices:
            market_by_minute[pt.t // 60 * 60] = pt.p

        for ts, btc_price in w.btc_path:
            ts = int(ts)
            seconds_remaining = w.window_end - ts
            if seconds_remaining < min_seconds_remaining:
                continue

            market_p = market_by_minute.get(ts)
            if market_p is None or not (0.0 < market_p < 1.0):
                continue

            distance = (btc_price - w.btc_open) / w.btc_open
            model_p = brownian_probability(
                distance, sigma_per_minute, seconds_remaining
            )

            obs.append(
                Observation(
                    slug=w.slug,
                    seconds_remaining=seconds_remaining,
                    distance_bps=distance * 10000,
                    model_p_up=model_p,
                    market_p_up=market_p,
                    realized_up=w.realized_up,
                )
            )

    return obs


@dataclass
class EdgeBucket:
    threshold: float
    trades: int = 0
    wins: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def pnl_per_trade(self) -> float:
        return self.net_pnl / self.trades if self.trades else 0.0


def simulate_edge_trading(
    observations: list[Observation],
    thresholds: list[float],
    cost_per_trade: float = 0.018,
) -> list[EdgeBucket]:
    """Simulate taking every signal whose edge exceeds each threshold.

    Buys one share of the side the model considers underpriced, at the
    market's own price, and settles it against the real outcome. Costs
    cover fees plus expected slippage.
    """
    buckets = [EdgeBucket(threshold=t) for t in thresholds]

    for o in observations:
        edge_up = o.model_p_up - o.market_p_up
        edge_down = (1 - o.model_p_up) - (1 - o.market_p_up)

        if edge_up >= edge_down:
            edge, entry, won = edge_up, o.market_p_up, o.realized_up == 1
        else:
            edge, entry, won = edge_down, 1 - o.market_p_up, o.realized_up == 0

        if not (0.0 < entry < 1.0):
            continue

        payout = 1.0 if won else 0.0
        gross = payout - entry
        net = gross - cost_per_trade

        for b in buckets:
            if edge >= b.threshold:
                b.trades += 1
                b.wins += int(won)
                b.gross_pnl += gross
                b.net_pnl += net

    return buckets


@dataclass
class ValidationReport:
    windows: int
    observations: int
    model_brier: float
    market_brier: float
    baseline_brier: float
    model_calibration: list[CalibrationBucket] = field(default_factory=list)
    market_calibration: list[CalibrationBucket] = field(default_factory=list)
    edge_buckets: list[EdgeBucket] = field(default_factory=list)
    mean_abs_disagreement: float = 0.0

    @property
    def model_beats_market(self) -> bool:
        return self.model_brier < self.market_brier

    @property
    def skill_vs_market_pct(self) -> float:
        """Positive means the model is more accurate than the market."""
        if not self.market_brier:
            return 0.0
        return (self.market_brier - self.model_brier) / self.market_brier * 100


def validate(
    windows: list[HistoricalWindow],
    sigma_per_minute: float,
    thresholds: list[float] | None = None,
    cost_per_trade: float = 0.018,
) -> ValidationReport:
    thresholds = thresholds or [0.02, 0.05, 0.10, 0.20]
    obs = build_observations(windows, sigma_per_minute)

    model_pairs = [(o.model_p_up, o.realized_up) for o in obs]
    market_pairs = [(o.market_p_up, o.realized_up) for o in obs]
    baseline_pairs = [(0.5, o.realized_up) for o in obs]

    disagreement = (
        sum(abs(o.edge) for o in obs) / len(obs) if obs else 0.0
    )

    return ValidationReport(
        windows=len(windows),
        observations=len(obs),
        model_brier=brier_score(model_pairs),
        market_brier=brier_score(market_pairs),
        baseline_brier=brier_score(baseline_pairs),
        model_calibration=calibration_table(model_pairs),
        market_calibration=calibration_table(market_pairs),
        edge_buckets=simulate_edge_trading(obs, thresholds, cost_per_trade),
        mean_abs_disagreement=disagreement,
    )


@dataclass
class FavouriteBiasResult:
    low: float
    high: float
    windows: int
    avg_entry: float
    win_rate: float
    pnl_per_trade: float
    z_score: float
    p_value: float | None = None


def favourite_bias_test(
    observations: list[Observation],
    band: tuple[float, float] = (0.60, 0.90),
    cost_per_trade: float = 0.018,
    permutations: int = 20000,
    seed: int = 7,
) -> FavouriteBiasResult:
    """Test whether buying the underdog beats its implied probability.

    One trade per window. Observations inside the same window resolve
    together, so treating each minute as independent would overstate the
    sample size several-fold and turn noise into apparent significance.

    The p-value comes from a permutation test: outcomes are re-drawn at
    the market's own implied probability, which is the null hypothesis
    that the market price is correct.
    """
    import random
    from collections import defaultdict

    lo, hi = band
    by_window: dict[str, list[Observation]] = defaultdict(list)
    for o in observations:
        by_window[o.slug].append(o)

    trades: list[tuple[float, float]] = []
    for points in by_window.values():
        for o in sorted(points, key=lambda x: -x.seconds_remaining):
            fav_is_up = o.market_p_up >= 0.5
            fav_price = o.market_p_up if fav_is_up else 1 - o.market_p_up
            if lo <= fav_price < hi:
                entry = 1 - fav_price
                won = (o.realized_up == 0) if fav_is_up else (o.realized_up == 1)
                trades.append((entry, 1.0 if won else 0.0))
                break

    n = len(trades)
    if n < 20:
        return FavouriteBiasResult(lo, hi, n, 0.0, 0.0, 0.0, 0.0, None)

    avg_entry = sum(t[0] for t in trades) / n
    win_rate = sum(t[1] for t in trades) / n
    pnl = sum(t[1] - t[0] - cost_per_trade for t in trades) / n
    se = math.sqrt(avg_entry * (1 - avg_entry) / n)
    z = (win_rate - avg_entry) / se if se > 0 else 0.0

    rng = random.Random(seed)
    hits = 0
    for _ in range(permutations):
        sim = sum(
            (1.0 if rng.random() < entry else 0.0) - entry - cost_per_trade
            for entry, _ in trades
        ) / n
        if sim >= pnl:
            hits += 1

    return FavouriteBiasResult(
        lo, hi, n, avg_entry, win_rate, pnl, z, hits / permutations
    )


def estimate_sigma(windows: list[HistoricalWindow]) -> float:
    """Realized 1-minute volatility across every window's BTC path."""
    rets: list[float] = []
    for w in windows:
        path = [p[1] for p in w.btc_path]
        for i in range(1, len(path)):
            if path[i - 1] > 0:
                rets.append((path[i] - path[i - 1]) / path[i - 1])
    if len(rets) < 30:
        return 0.00026
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    return max(math.sqrt(var), 1e-6)
