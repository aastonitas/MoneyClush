"""Cross-market signals: z-score between related BTC Up/Down markets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def cross_market_zscore(
    current_spread: float,
    average_spread: float,
    spread_std: float,
) -> float:
    """Z-score measuring how far the spread between two related markets
    deviates from its historical average.

    A high absolute z-score suggests one market hasn't adjusted yet.
    """
    if spread_std <= 0:
        return 0.0
    return (current_spread - average_spread) / spread_std


@dataclass
class SpreadTracker:
    """Tracks rolling statistics of the spread between two markets.

    Use to compute z-scores between e.g. 5m and 15m BTC Up/Down markets.
    """

    window_size: int = 200
    _spreads: deque = field(default_factory=lambda: deque(maxlen=200))

    def update(self, price_a: float, price_b: float) -> None:
        self._spreads.append(price_a - price_b)

    @property
    def mean(self) -> float:
        if not self._spreads:
            return 0.0
        return sum(self._spreads) / len(self._spreads)

    @property
    def std(self) -> float:
        if len(self._spreads) < 2:
            return 0.0
        m = self.mean
        variance = sum((s - m) ** 2 for s in self._spreads) / len(self._spreads)
        return variance**0.5

    def zscore(self, current_spread: float) -> float:
        return cross_market_zscore(current_spread, self.mean, self.std)

    @property
    def ready(self) -> bool:
        return len(self._spreads) >= 20
