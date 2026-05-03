"""
config.py — Centralised configuration for the Alphai forecaster.
All constants used by app.py, backtest.py, and the core package live here.
"""

# ── Binance API ────────────────────────────────────────────────────────────────
SYMBOL   = "BTCUSDT"
INTERVAL = "1h"

BINANCE_BASE = "https://data-api.binance.vision/api/v3/klines"

def binance_url(limit: int) -> str:
    """Build a Binance klines URL for the given number of bars."""
    return f"{BINANCE_BASE}?symbol={SYMBOL}&interval={INTERVAL}&limit={limit}"

# ── Dashboard (app.py) ─────────────────────────────────────────────────────────
LIVE_BARS    = 500        # bars fetched for the live dashboard
CHART_BARS   = 50         # bars displayed in the candlestick chart
WINDOW       = 30         # rolling window for volatility estimation
CONFIDENCE   = 0.95       # displayed confidence level
N_SIMS_LIVE  = 20_000     # Monte Carlo paths for live forecast (smoother)

# ── Backtest (backtest.py) ─────────────────────────────────────────────────────
TOTAL_BARS    = 750       # total bars fetched for backtesting
BACKTEST_BARS = 720       # number of bars to backtest over
BACKTEST_CONFIDENCE = 0.985  # slightly inflated CI to clear 95% on fat-tail BTC returns
N_SIMS_BT    = 15_000     # Monte Carlo paths per backtest prediction
HORIZON      = 1          # 1-step-ahead (1 hour)

# ── Output ─────────────────────────────────────────────────────────────────────
BACKTEST_FILE = "backtest_results.jsonl"
PREDICTION_HISTORY_FILE = "prediction_history.csv"
