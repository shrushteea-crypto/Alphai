"""
app.py — Bitcoin GBM Price Range Forecaster · Streamlit Dashboard
==================================================================
Entry-point for the live dashboard.
"""

import json
import os
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphai.config import (
    BACKTEST_FILE,
    CHART_BARS,
    CONFIDENCE,
    LIVE_BARS,
    N_SIMS_LIVE,
    WINDOW,
)
from alphai.data import fetch_candles, save_live_prediction, load_live_predictions, update_prediction_actuals
from alphai.model import predict_range

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BTC Price Range Forecaster",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
  }

  .main .block-container { 
    padding: 1.5rem 1rem !important; 
    max-width: 1400px; 
  }

  .hero-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    font-weight: 400;
  }
  hr.accent-rule {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, #f7931a, transparent);
    margin: 1rem 0 2rem 0;
  }

  .section-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-left: 2px solid #f7931a;
    padding-left: 0.6rem;
    margin: 2rem 0 1rem 0;
  }

  .metric-card {
    background-color: #0d1117;
    border: 1px solid #1a1a2e;
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    transition: all 0.2s ease-in-out;
    height: 100%;
  }
  .metric-card:hover {
    border-color: #f7931a66;
    box-shadow: 0 4px 12px rgba(247, 147, 26, 0.15);
  }
  .metric-card.green-glow:hover {
    border-color: #00d4aa66;
    box-shadow: 0 4px 12px rgba(0, 212, 170, 0.15);
  }
  .metric-label {
    font-size: 0.65rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.1;
  }
  .metric-value.orange { color: #f7931a; }
  .metric-value.green  { color: #00d4aa; }
  .metric-value.red    { color: #ef4444; }
  .metric-value.yellow { color: #eab308; }
  .metric-value.purple { color: #818cf8; }
  .metric-value.blue   { color: #60a5fa; }
  
  .metric-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.4rem;
  }
  .insight-text {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid #1a1a2e;
  }

  .timer-badge {
    display: inline-block;
    background: rgba(247, 147, 26, 0.1);
    color: #f7931a;
    border: 1px solid rgba(247, 147, 26, 0.3);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    font-size: 0.7rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  [data-testid="stMetric"], 
  [data-testid="stPlotlyChart"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  
  .footer {
    text-align: center;
    color: #475569;
    font-size: 0.75rem;
    margin-top: 3rem;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  [data-testid="stToolbar"] { display: none; }
  [data-testid="stDecoration"] { display: none; }
  [data-testid="manage-app-button"] { display: none; }
  iframe { display: none !important; } 
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def _fetch_candles_cached() -> pd.DataFrame:
    return fetch_candles(LIVE_BARS)

@st.cache_data
def _load_backtest(filepath: str) -> pd.DataFrame | None:
    if not os.path.exists(filepath): return None
    rows = [json.loads(line) for line in open(filepath) if line.strip()]
    return pd.DataFrame(rows) if rows else None

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER & TIME AWARENESS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-header">₿ BTC Price Range Forecaster</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Geometric Brownian Motion · Student-t Distribution · '
    '95% Confidence Interval · Live Binance Data</div>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="accent-rule">', unsafe_allow_html=True)

with st.spinner("Fetching live BTC data from Binance…"):
    try:
        df = _fetch_candles_cached()
        data_ok = True
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        data_ok = False

if not data_ok:
    st.stop()

# Time logic
current_time = datetime.now(timezone.utc)
latest_candle_time = df["timestamp"].iloc[-1].tz_localize(timezone.utc)
next_hour = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
next_candle_time = latest_candle_time + pd.Timedelta(hours=1)
time_remaining = next_candle_time - current_time
mins_remaining = max(0, int(time_remaining.total_seconds() // 60))

# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST METRICS (Moved to Top)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Model Performance (Historical)</div>', unsafe_allow_html=True)

bt = _load_backtest(BACKTEST_FILE)
if bt is not None:
    n_bars   = len(bt)
    coverage = bt["covered"].mean()
    avg_w    = bt["width"].mean()
    avg_ws   = bt["winkler_score"].mean()

    # Coverage insights
    if 0.94 <= coverage <= 0.96:
        cov_color, cov_class = "#00d4aa", "green"
        cov_insight = f"Model well-calibrated — {coverage*100:.1f}% of actual prices fell within predicted range."
    elif 0.92 <= coverage < 0.94 or 0.96 < coverage <= 0.98:
        cov_color, cov_class = "#eab308", "yellow"
        cov_insight = f"Model well-calibrated — {coverage*100:.1f}% of actual prices fell within predicted range."
    else:
        cov_color, cov_class = "#ef4444", "red"
        cov_insight = f"Model off target — {coverage*100:.1f}% of actual prices fell within predicted range."

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Target 95% Coverage Rate</div>
          <div class="metric-value {cov_class}">{coverage:.4f}</div>
          <div class="metric-sub">Over last {n_bars} hours</div>
          <div class="insight-text">{cov_insight}</div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Mean Range Width</div>
          <div class="metric-value purple">${avg_w:,.0f}</div>
          <div class="metric-sub">Average width of 95% CI</div>
          <div class="insight-text">Narrower intervals indicate a more confident, sharper model.</div>
        </div>
        """, unsafe_allow_html=True)

    with b3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Mean Winkler Score</div>
          <div class="metric-value blue">{avg_ws:,.0f}</div>
          <div class="metric-sub">Accuracy + Sharpness penalty</div>
          <div class="insight-text">Lower is better. Heavily penalizes predictions that miss the actual price.</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("📂 No backtest data found. Run `python backtest.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTE LOG-RETURNS, VOLATILITY REGIME & FORECAST
# ══════════════════════════════════════════════════════════════════════════════
closes      = df["close"].values
log_returns = np.diff(np.log(closes))
window_ret  = log_returns[-WINDOW:]
current_px  = closes[-1]

# Volatility Regime logic
import scipy.stats as stats
historical_vols = pd.Series(log_returns).rolling(WINDOW).std().dropna()
current_vol = historical_vols.iloc[-1]
vol_rank = stats.percentileofscore(historical_vols, current_vol) if len(historical_vols) > 0 else 50

if vol_rank > 75:
    vol_regime, vol_color = "HIGH", "orange"
    vol_insight = "Market currently in high volatility regime → wider prediction band expected."
elif vol_rank < 25:
    vol_regime, vol_color = "LOW", "blue"
    vol_insight = "Market currently in low volatility regime → tighter prediction band expected."
else:
    vol_regime, vol_color = "MEDIUM", "green"
    vol_insight = "Market showing normal volatility levels."

forecast = predict_range(
    window_returns=window_ret,
    current_price=current_px,
    confidence=CONFIDENCE,
    n_sims=N_SIMS_LIVE,
)

low, high, mid, sigma_ann = forecast.low, forecast.high, forecast.mean, forecast.sigma_ann
range_width = forecast.width
pct_width   = range_width / current_px * 100

# Save live prediction
save_live_prediction(str(next_hour), current_px, low, high, mid)

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE SNAPSHOT & INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Live Prediction</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="timer-badge">Valid for next {mins_remaining} mins</div>
      <div class="metric-label">Current BTC Price</div>
      <div class="metric-value">${current_px:,.0f}</div>
      <div class="metric-sub">As of {latest_candle_time.strftime('%H:%M UTC')}</div>
      <div class="insight-text">Data synced successfully. Forecasting +1 hour.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card green-glow">
      <div class="metric-label">Predicted Range · {next_hour.strftime('%H:%M UTC')}</div>
      <div class="metric-value green" style="font-size:1.6rem;">
        ${low:,.0f} – ${high:,.0f}
      </div>
      <div class="metric-sub">95% Confidence Interval</div>
      <div class="insight-text">
        <strong>Expected Move:</strong> ±${forecast.expected_move_usd:,.0f} (±{forecast.expected_move_pct:.2f}%)
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Volatility Regime</div>
      <div class="metric-value {vol_color}">{vol_regime}</div>
      <div class="metric-sub">{sigma_ann*100:.1f}% Ann. Volatility ({vol_rank:.0f}th Percentile)</div>
      <div class="insight-text">{vol_insight}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CHARTING
# ══════════════════════════════════════════════════════════════════════════════
chart_df    = df.tail(CHART_BARS).copy()
forecast_ts = [df["timestamp"].iloc[-1], next_hour]

fig = go.Figure()

# Candlestick bars
fig.add_trace(go.Candlestick(
    x=chart_df["timestamp"],
    open=chart_df["open"],
    high=chart_df["high"],
    low=chart_df["low"],
    close=chart_df["close"],
    name="BTC/USDT",
    increasing_line_color="#f7931a",
    decreasing_line_color="#ef4444",
    increasing_fillcolor="rgba(247,147,26,0.8)",
    decreasing_fillcolor="rgba(239,68,68,0.8)",
    line=dict(width=1.5),
))

# Forecast ribbon
fig.add_trace(go.Scatter(
    x=forecast_ts, y=[high, high],
    mode="lines",
    line=dict(color="rgba(129,140,248,0.0)"),
    showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=forecast_ts, y=[low, low],
    mode="lines",
    line=dict(color="rgba(129,140,248,0.0)"),
    fill="tonexty",
    fillcolor="rgba(129,140,248,0.2)",
    name="95% Forecast Interval",
    hoverinfo="skip",
))

# Mean-forecast dotted line
fig.add_trace(go.Scatter(
    x=forecast_ts, y=[current_px, mid],
    mode="lines+markers",
    line=dict(color="#818cf8", width=2, dash="dot"),
    marker=dict(size=6, color="#818cf8"),
    name=f"Expected Mean ${mid:,.0f}",
))

fig.add_annotation(
    x=next_hour, y=high, text=f"  ${high:,.0f} (Upper)",
    showarrow=False, font=dict(color="#818cf8", size=11, family="Space Grotesk"), xanchor="left",
)
fig.add_annotation(
    x=next_hour, y=low, text=f"  ${low:,.0f} (Lower)",
    showarrow=False, font=dict(color="#818cf8", size=11, family="Space Grotesk"), xanchor="left",
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1117",
    font=dict(family="Space Grotesk", color="#94a3b8", size=12),
    height=450,
    margin=dict(l=10, r=70, t=10, b=10),
    xaxis=dict(showgrid=False, rangeslider=dict(visible=False), color="#64748b", linecolor="#1a1a2e"),
    yaxis=dict(showgrid=True, gridcolor="#1a1a2e", tickprefix="$", tickformat=",.0f", color="#64748b", linecolor="#1a1a2e", side="right"),
    legend=dict(bgcolor="rgba(13,17,23,0.8)", bordercolor="#1a1a2e", borderwidth=1, font=dict(size=11), yanchor="top", y=0.99, xanchor="left", x=0.01),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#0d1117", bordercolor="#1a1a2e", font=dict(color="#f1f5f9")),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Recent Predictions</div>', unsafe_allow_html=True)

live_df = load_live_predictions()
if live_df is not None and not live_df.empty:
    
    # Fill actuals
    live_df = update_prediction_actuals(live_df, df)
    
    # Sort descending
    live_df = live_df.sort_values(by="timestamp", ascending=False).head(20).copy()
    
    def check_hit(row):
        if pd.isna(row.get("actual_close")): return "⏳ Pending"
        if row["predicted_low"] <= row["actual_close"] <= row["predicted_high"]:
            return "✅ Hit"
        return "❌ Miss"
        
    live_df["Status"] = live_df.apply(check_hit, axis=1)
    
    # Format prices as $X,XXX
    for col in ["current_price", "predicted_low", "predicted_high", "predicted_mean", "actual_close"]:
        if col in live_df.columns:
            live_df[col] = live_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")

    # Format for display
    display_df = live_df.rename(columns={
        "timestamp": "Target Time (UTC)",
        "current_price": "Start Price",
        "predicted_low": "Pred Low",
        "predicted_high": "Pred High",
        "predicted_mean": "Pred Mean",
        "actual_close": "Actual Close"
    })
    display_df["Target Time (UTC)"] = display_df["Target Time (UTC)"].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        display_df[["Target Time (UTC)", "Start Price", "Pred Low", "Pred High", "Actual Close", "Status"]],
        use_container_width=True,
        hide_index=True
    )
    
    # Quick stat
    resolved = live_df[live_df["actual_close"] != ""]
    if not resolved.empty:
        recent_hits = (resolved["Status"] == "✅ Hit").sum()
        recent_acc = recent_hits / len(resolved) * 100
        st.markdown(f"<div style='color: #64748b; font-size: 0.8rem; margin-top: 0.5rem;'>"
                    f"Recent Accuracy (Last {len(resolved)} completed): <strong>{recent_acc:.1f}%</strong></div>", 
                    unsafe_allow_html=True)
else:
    st.info("No live predictions saved yet. Data will populate after the next hourly refresh.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "BTC Price Range Forecaster · GBM + Student-t · "
    "Data: Binance Vision API · Not financial advice"
    "</div>",
    unsafe_allow_html=True,
)