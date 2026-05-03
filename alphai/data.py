"""
data.py — Binance public API data fetching.
"""

import os
import csv
import pandas as pd
import requests

from alphai.config import binance_url, PREDICTION_HISTORY_FILE


def fetch_candles(limit: int) -> pd.DataFrame:
    """
    Fetch the latest `limit` hourly BTC/USDT candles from Binance public API.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, open, high, low, close, volume
    """
    url = binance_url(limit)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close",
        "volume", "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]].copy()


def fetch_closes(limit: int) -> list[float]:
    """
    Convenience helper that returns only the close prices as a plain list.
    Used by the backtest script.
    """
    df = fetch_candles(limit)
    return df["close"].tolist()


def save_live_prediction(timestamp: str, current_price: float, low: float, high: float, mean: float):
    """Save a live prediction to the CSV file."""
    file_exists = os.path.exists(PREDICTION_HISTORY_FILE)
    if file_exists:
        try:
            df = pd.read_csv(PREDICTION_HISTORY_FILE)
            if timestamp in df['timestamp'].astype(str).values:
                return # Already exists
        except Exception:
            pass # File might be empty
    
    with open(PREDICTION_HISTORY_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(PREDICTION_HISTORY_FILE) == 0:
            writer.writerow(["timestamp", "current_price", "predicted_low", "predicted_high", "predicted_mean", "actual_close"])
        writer.writerow([timestamp, round(current_price, 2), round(low, 2), round(high, 2), round(mean, 2), ""])

def load_live_predictions() -> pd.DataFrame | None:
    """Load live predictions."""
    if not os.path.exists(PREDICTION_HISTORY_FILE) or os.path.getsize(PREDICTION_HISTORY_FILE) == 0:
        return None
    try:
        df = pd.read_csv(PREDICTION_HISTORY_FILE)
        if df.empty: return None
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return None

def update_prediction_actuals(df_live: pd.DataFrame, df_binance: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing actual_close values if they exist in the recent Binance data."""
    if df_live is None or df_live.empty:
        return df_live
        
    updated = False
    for idx, row in df_live.iterrows():
        if pd.isna(row.get('actual_close')):
            # Find in binance df
            b_row = df_binance[df_binance['timestamp'] == row['timestamp']]
            if not b_row.empty:
                actual_close = b_row['close'].values[0]
                df_live.at[idx, 'actual_close'] = actual_close
                updated = True
                
    if updated:
        df_live.to_csv(PREDICTION_HISTORY_FILE, index=False)
        
    return df_live
