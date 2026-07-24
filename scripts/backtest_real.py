"""Validate the fair value model against real resolved Polymarket windows.

Usage:
    python scripts/fetch_history.py --windows 300
    python scripts/backtest_real.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from moneyclush.research.historical import load_history
from moneyclush.research.validation import (
    build_observations,
    estimate_sigma,
    favourite_bias_test,
    validate,
)

BAR = "=" * 72


def pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default="")
    ap.add_argument("--cost", type=float, default=0.018)
    args = ap.parse_args()

    path = Path(args.file) if args.file else ROOT / "data" / "history" / "btc_5m.jsonl"
    if not path.exists():
        print(f"No existe {path}. Ejecuta primero: python scripts/fetch_history.py")
        return

    windows = load_history(path)
    sigma = estimate_sigma(windows)

    print(BAR)
    print("MoneyClush — Validación con datos reales de Polymarket")
    print(BAR)
    print(f"  ventanas resueltas : {len(windows)}")
    print(f"  sigma BTC 1m       : {sigma*10000:.2f} bps (medida del propio histórico)")

    report = validate(windows, sigma, cost_per_trade=args.cost)
    print(f"  observaciones      : {report.observations} (puntos minuto a minuto)")

    # ---------------------------------------------------------------- brier
    print()
    print(BAR)
    print("PRECISIÓN — Brier score (más bajo = mejor)")
    print(BAR)
    print(f"  moneda al aire (0.50) : {report.baseline_brier:.4f}")
    print(f"  precio del mercado    : {report.market_brier:.4f}")
    print(f"  nuestro modelo        : {report.model_brier:.4f}")
    print()
    if report.model_beats_market:
        print(f"  -> El modelo es {report.skill_vs_market_pct:+.1f}% MÁS preciso que el mercado.")
    else:
        print(f"  -> El modelo es {abs(report.skill_vs_market_pct):.1f}% MENOS preciso que el mercado.")
        print("     Cualquier 'edge' contra este mercado es error nuestro.")
    print(f"  desacuerdo medio modelo vs mercado: {report.mean_abs_disagreement*100:.1f} puntos")

    # ---------------------------------------------------- calibration tables
    for title, table in [
        ("CALIBRACIÓN DEL MODELO", report.model_calibration),
        ("CALIBRACIÓN DEL MERCADO", report.market_calibration),
    ]:
        print()
        print(BAR)
        print(f"{title} — de las veces que dijo X%, ¿cuántas ganó Up?")
        print(BAR)
        print(f"  {'rango':>12} {'n':>6} {'predicho':>10} {'real':>8} {'error':>8}")
        for b in table:
            if b.count < 5:
                continue
            flag = "  <<<" if abs(b.gap) > 0.15 else ""
            print(
                f"  {b.low:.1f}-{b.high:.1f}".rjust(14)
                + f"{b.count:>6}{pct(b.predicted):>11}{pct(b.realized):>9}"
                + f"{b.gap*100:>+7.1f}{flag}"
            )

    # -------------------------------------------------------- edge simulation
    print()
    print(BAR)
    print("SIMULACIÓN — operar cada señal por encima de cada umbral de edge")
    print(BAR)
    print(f"  costes aplicados: {args.cost*100:.1f}% por operación")
    print()
    print(f"  {'umbral':>8}{'trades':>9}{'win rate':>11}{'PnL bruto':>12}"
          f"{'PnL neto':>11}{'por trade':>12}")
    for b in report.edge_buckets:
        print(
            f"  {b.threshold*100:>6.0f}%{b.trades:>9}{pct(b.win_rate):>11}"
            f"{b.gross_pnl:>+12.2f}{b.net_pnl:>+11.2f}{b.pnl_per_trade:>+12.4f}"
        )

    # --------------------------------------------------- favourite bias test
    print()
    print(BAR)
    print("SESGO DEL FAVORITO — comprar el underdog cuando el favorito está caro")
    print(BAR)
    obs = build_observations(windows, sigma)
    print(f"  {'banda':>12}{'ventanas':>10}{'entrada':>9}{'gana':>8}"
          f"{'PnL/trade':>12}{'z':>7}{'p-value':>10}")
    for band in [(0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.60, 0.90)]:
        r = favourite_bias_test(obs, band, cost_per_trade=args.cost, permutations=8000)
        if r.windows < 20:
            print(f"  {band[0]:.2f}-{band[1]:.2f}{r.windows:>10}   muestra insuficiente")
            continue
        pv = f"{r.p_value:.4f}" if r.p_value is not None else "—"
        print(
            f"  {r.low:.2f}-{r.high:.2f}{r.windows:>10}{r.avg_entry:>9.3f}"
            f"{r.win_rate:>8.1%}{r.pnl_per_trade:>+12.4f}{r.z_score:>7.2f}{pv:>10}"
        )
    print()
    print("  Una operación por ventana: los minutos de una misma ventana resuelven")
    print("  juntos y contarlos por separado inflaría la muestra varias veces.")
    print("  p-value por permutación contra la hipótesis de que el mercado acierta.")

    # ------------------------------------------------------------- verdict
    print()
    print(BAR)
    print("VEREDICTO")
    print(BAR)
    profitable = [b for b in report.edge_buckets if b.net_pnl > 0 and b.trades >= 30]
    if report.model_beats_market and profitable:
        best = max(profitable, key=lambda b: b.pnl_per_trade)
        print(f"  El modelo supera al mercado y el umbral de {best.threshold*100:.0f}%")
        print(f"  habría dado {best.pnl_per_trade:+.4f}$ por operación en {best.trades} trades.")
        print("  Sigue siendo backtest: no incluye latencia ni fills parciales.")
    else:
        print("  El modelo NO supera al mercado en este histórico.")
        print("  Operar sus discrepancias pierde dinero: son error del modelo,")
        print("  no ineficiencia del mercado. El sistema hace bien en no operar.")
        print()
        print("  Vías que sí podrían tener edge, en orden de coste/beneficio:")
        print("   1. Arbitraje Up+Down < $1 — no depende de acertar el fair value")
        print("   2. Mercados de 15m con baja liquidez, mucho peor arbitrados")
        print("   3. Latencia: WebSocket del CLOB en vez de polling REST")
        print("   4. Chainlink como fuente de precio (es con lo que resuelve)")
    print(BAR)


if __name__ == "__main__":
    main()
