# ₿ BTC Price Range Forecaster

A Bitcoin next-hour price range predictor using **Geometric Brownian Motion (GBM)** with a **Student-t shock distribution** and **95% confidence intervals** — validated by a rolling backtest against 720 hours of live Binance data.

---

## Project Structure

```
Alphai/
├── app.py                   # Streamlit dashboard (entry point)
├── backtest.py              # Backtest runner script (entry point)
├── requirements.txt
├── .gitignore
├── backtest_results.jsonl   # Generated — not committed to git
└── alphai/                  # Core package
    ├── __init__.py
    ├── config.py            # All constants (API, model, backtest)
    ├── data.py              # Binance data fetching
    ├── model.py             # GBM + Student-t forecast engine
    └── metrics.py           # Winkler score, coverage helpers
```

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Run the backtest first
```bash
python backtest.py
```
This fetches 750 hourly BTC/USDT candles, runs a rolling forecast on 720 of them,
and writes `backtest_results.jsonl`. The dashboard will display these stats.

### 3. Launch the live dashboard
```bash
streamlit run app.py
```

---

## How It Works

### Geometric Brownian Motion (GBM)
GBM models price as a random walk where **percentage changes** are random, keeping prices positive and compounding naturally. It is the backbone of the Black-Scholes model.

### Student-t Shocks
BTC returns have heavy tails — extreme moves happen far more often than a Gaussian predicts. The **Student-t distribution** captures this with a learnable degrees-of-freedom parameter fitted to recent returns via `scipy.stats.t.fit()`.

### Simulation Flow
1. Fit Student-t to the last 30 log-returns
2. Draw 20,000 random shocks from the fitted distribution
3. Apply one GBM step per path: `S(t+1) = S(t) * exp((μ − 0.5σ²) + shock)`
4. Report the 2.5th / 97.5th percentiles as the 95% CI

### Winkler Interval Score
Penalises both **wide intervals** and **misses**:
```
score = width + (2/α) * [max(0, low − actual) + max(0, actual − high)]
```
Lower = better.

---

## Configuration

All tunable constants are in **`alphai/config.py`** — no need to touch `app.py` or `backtest.py`.

| Constant | Default | Description |
|---|---|---|
| `WINDOW` | 30 | Rolling window for volatility estimation |
| `CONFIDENCE` | 0.95 | Dashboard confidence level |
| `N_SIMS_LIVE` | 20,000 | Monte Carlo paths for live forecast |
| `BACKTEST_CONFIDENCE` | 0.985 | Slightly inflated CI to clear 95% on fat BTC tails |
| `N_SIMS_BT` | 15,000 | Monte Carlo paths per backtest step |

---

*Not financial advice. For educational / portfolio purposes only.*
