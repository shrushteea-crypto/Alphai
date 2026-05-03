"""
backtest.py — Bitcoin GBM-Student-t Price Range Backtester
============================================================
Standalone script: fetches historical data, runs a rolling backtest,
saves results to backtest_results.jsonl, and prints a summary.

Run with:
    python backtest.py
"""

import json

import numpy as np

from alphai.config import (
    BACKTEST_BARS,
    BACKTEST_CONFIDENCE,
    BACKTEST_FILE,
    HORIZON,
    N_SIMS_BT,
    TOTAL_BARS,
    WINDOW,
)
from alphai.data import fetch_candles
from alphai.metrics import winkler_score
from alphai.model import predict_range

# ── Step 1: Fetch historical candles ───────────────────────────────────────
print(f"Fetching {TOTAL_BARS} hourly candles from Binance...")
df     = fetch_candles(TOTAL_BARS)
closes = df["close"].values
print(f"Got {len(closes)} bars. Most recent close: ${closes[-1]:,.2f}")

# ── Step 2: Compute log-returns ────────────────────────────────────────────
log_returns = np.diff(np.log(closes))  # length = len(closes) - 1

# ── Step 3: Rolling backtest loop ─────────────────────────────────────────
print(f"\nRunning backtest on {BACKTEST_BARS} bars (rolling window = {WINDOW})...")
print("This may take ~30 seconds (15,000 simulations × 720 bars)...")

start_idx = TOTAL_BARS - BACKTEST_BARS

results: list[dict] = []
hits = 0

for i in range(BACKTEST_BARS - 1):
    bar_idx = start_idx + i

    # Use only data strictly before bar_idx+1 (no lookahead!)
    available_returns = log_returns[:bar_idx]
    window_returns    = available_returns[-WINDOW:]

    if len(window_returns) < 2:
        continue

    current_price = closes[bar_idx]
    actual_price  = closes[bar_idx + 1]

    forecast = predict_range(
        window_returns=window_returns,
        current_price=current_price,
        confidence=BACKTEST_CONFIDENCE,
        n_sims=N_SIMS_BT,
        horizon=HORIZON,
    )

    covered = bool(forecast.low <= actual_price <= forecast.high)
    if covered:
        hits += 1

    ws = winkler_score(
        low=forecast.low,
        high=forecast.high,
        actual=actual_price,
        alpha=1.0 - BACKTEST_CONFIDENCE,
    )

    results.append({
        "bar_index":      bar_idx,
        "current_price":  round(current_price,  4),
        "actual_price":   round(actual_price,   4),
        "predicted_low":  round(forecast.low,   4),
        "predicted_high": round(forecast.high,  4),
        "covered":        covered,
        "width":          round(forecast.width, 4),
        "winkler_score":  round(ws,             4),
    })

# ── Step 4: Save results ───────────────────────────────────────────────────
with open(BACKTEST_FILE, "w") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")
print(f"\nSaved {len(results)} results → {BACKTEST_FILE}")

# ── Step 5: Print summary ──────────────────────────────────────────────────
valid       = [r for r in results if r["width"] > 0]
n           = len(valid)
coverage    = hits / n if n else 0.0
avg_width   = float(np.mean([r["width"]        for r in valid]))
avg_winkler = float(np.mean([r["winkler_score"] for r in valid]))

print("\n" + "=" * 45)
print("         BACKTEST SUMMARY RESULTS")
print("=" * 45)
print(f"  Bars tested       : {n}")
print(f"  Coverage rate     : {coverage:.4f}  (target ≥ {BACKTEST_CONFIDENCE})")
print(f"  Avg range width   : ${avg_width:,.2f}")
print(f"  Mean Winkler score: {avg_winkler:,.2f}  (lower = better)")
print("=" * 45)

if coverage >= BACKTEST_CONFIDENCE:
    print("✅  Coverage target MET!")
else:
    print(f"⚠️   Coverage below target by {BACKTEST_CONFIDENCE - coverage:.4f}")
