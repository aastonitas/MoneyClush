"""Measure whether 1-minute direction is predictable, and what it would take.

Written to answer a specific question: could this project generate signals
for 1-minute binary options the way the signal bots advertise? Rather than
argue about it, this measures it on real 1-minute candles.

Two things get computed.

**Hit rate.** Every predictor a 1-minute signal bot typically uses --
momentum, mean reversion, an SMA cross, RSI extremes, streaks, big
candles, volume spikes -- is run over thousands of real minutes and scored
on whether the *next* minute closed up. Wilson intervals, not bare
percentages, because a 51% hit rate on 400 samples means nothing.

**The bar.** Binary options pay less than 1:1 -- typically 80-92% on a win
against a 100% loss -- so the break-even hit rate is 1/(1+payout), which
lands between 52.1% and 55.6%. That is the number every predictor has to
clear, and it is why "better than a coin flip" is not the relevant test.

Run:  python scripts/probe_1min.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import httpx

OKX_HISTORY = "https://www.okx.com/api/v5/market/history-candles"
UA = {"User-Agent": "Mozilla/5.0"}
CACHE = Path(__file__).resolve().parent.parent / "data" / "history" / "btc_1m_probe.json"

# Payout on a winning binary option, as advertised by the usual brokers.
# A loss always costs the full stake, which is what makes the break-even
# hit rate sit well above 50%.
PAYOUTS = (0.92, 0.85, 0.80)


def fetch_candles(pages: int = 20, inst: str = "BTC-USDT") -> list[dict]:
    """Pull 1-minute candles, newest first, paging backwards."""
    rows: dict[int, list] = {}
    after: int | None = None

    with httpx.Client(headers=UA, timeout=25) as client:
        for _ in range(pages):
            params = {"instId": inst, "bar": "1m", "limit": "300"}
            if after:
                params["after"] = str(after)
            data = client.get(OKX_HISTORY, params=params).json().get("data", [])
            if not data:
                break
            for row in data:
                rows[int(row[0])] = row
            after = int(data[-1][0])
            time.sleep(0.12)

    return [
        {"ts": ts, "o": float(r[1]), "h": float(r[2]),
         "l": float(r[3]), "c": float(r[4]), "v": float(r[5])}
        for ts, r in sorted(rows.items())
    ]


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — honest at the sample sizes this produces."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def breakeven(payout: float) -> float:
    """Hit rate needed to break even when a win pays `payout` and a loss costs 1."""
    return 1.0 / (1.0 + payout)


def build_predictors(candles: list[dict]):
    """The standard 1-minute toolkit, each returning True (up), False, or None.

    None means the predictor is not firing on that bar — a signal that only
    triggers sometimes must be scored only on the bars where it triggered,
    or its accuracy gets diluted by minutes it never called.
    """
    c = [x["c"] for x in candles]
    o = [x["o"] for x in candles]
    v = [x["v"] for x in candles]

    def sma(series, i, n):
        return sum(series[i - n + 1: i + 1]) / n

    def rsi(i, period=14):
        gain = loss = 0.0
        for j in range(i - period + 1, i + 1):
            change = c[j] - c[j - 1]
            gain += max(change, 0.0)
            loss += max(-change, 0.0)
        if loss == 0:
            return 100.0
        return 100 - 100 / (1 + (gain / period) / (loss / period))

    def streak(i):
        if c[i] > c[i - 1] and c[i - 1] > c[i - 2]:
            return True
        if c[i] < c[i - 1] and c[i - 1] < c[i - 2]:
            return False
        return None

    def big_body(i):
        return (c[i] > o[i]) if abs(c[i] - o[i]) / c[i] > 0.0008 else None

    def rsi_extreme(i):
        r = rsi(i)
        if r < 30:
            return True
        if r > 70:
            return False
        return None

    def volume_spike(i):
        return (c[i] > o[i]) if v[i] > 2 * sma(v, i, 20) else None

    return {
        "momentum (seguir vela previa)": lambda i: c[i] > c[i - 1],
        "reversión (contra vela previa)": lambda i: c[i] < c[i - 1],
        "cruce SMA 3/10": lambda i: sma(c, i, 3) > sma(c, i, 10),
        "RSI14 sobrecompra/sobreventa": rsi_extreme,
        "racha de 3 velas (seguir)": streak,
        "cuerpo grande (seguir)": big_body,
        "pico de volumen (seguir)": volume_spike,
    }


def score(candles: list[dict], predictor) -> tuple[int, int]:
    """Hits and total calls, skipping flat minutes where there is no answer."""
    c = [x["c"] for x in candles]
    hits = n = 0

    for i in range(25, len(c) - 1):
        try:
            call = predictor(i)
        except Exception:
            call = None
        if call is None or c[i + 1] == c[i]:
            continue
        n += 1
        hits += (call == (c[i + 1] > c[i]))

    return hits, n


def main() -> int:
    if CACHE.exists():
        candles = json.loads(CACHE.read_text())
        print(f"usando caché: {CACHE.name}")
    else:
        candles = fetch_candles()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(candles))

    if len(candles) < 500:
        print("muestra insuficiente")
        return 1

    span_h = (candles[-1]["ts"] - candles[0]["ts"]) / 3_600_000
    print(f"{len(candles)} velas de 1m · {span_h:.0f} horas de BTC real\n")

    print("UMBRALES DE RENTABILIDAD (opciones binarias)")
    for payout in PAYOUTS:
        print(f"  pago {payout * 100:.0f}%  ->  hay que acertar "
              f"{breakeven(payout) * 100:.2f}% solo para no perder")

    bar = breakeven(max(PAYOUTS))     # the most generous payout = lowest bar
    print(f"\nUsamos el umbral más benévolo: {bar * 100:.2f}%\n")

    header = f"{'PREDICTOR':32} {'N':>5} {'ACIERTO':>8} {'IC 95%':>16}  VEREDICTO"
    print(header)
    print("-" * len(header))

    for name, predictor in build_predictors(candles).items():
        hits, n = score(candles, predictor)
        if n < 100:
            print(f"{name:32} {n:5}   muestra insuficiente")
            continue

        lo, hi = wilson(hits, n)
        acc = hits / n
        if lo > bar:
            verdict = "SUPERA EL UMBRAL"
        elif hi < bar:
            verdict = "no llega"
        else:
            verdict = "indistinguible del azar"

        print(f"{name:32} {n:5} {acc * 100:7.2f}% "
              f"[{lo * 100:5.1f}%,{hi * 100:5.1f}%]  {verdict}")

    # What the best of them would actually cost to trade.
    best_name, best = "", 0.0
    for name, predictor in build_predictors(candles).items():
        hits, n = score(candles, predictor)
        if n >= 100 and hits / n > best:
            best_name, best = name, hits / n

    print(f"\nMejor predictor: {best_name} a {best * 100:.2f}%")
    print("Valor esperado por operación, apostando $10:")
    for payout in PAYOUTS:
        ev = best * payout - (1 - best)
        print(f"  pago {payout * 100:.0f}%  ->  {ev * 100:+.2f}% = ${ev * 10:+.3f} "
              f"· 100 operaciones = ${ev * 1000:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
