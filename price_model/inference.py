# price_model/inference.py

import os
import time
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import requests
import joblib
import tensorflow as tf

from .config import (
    BINANCE_BASE,
    BINANCE_SYMBOLS,
    INTERVAL,
    INTERVAL_SEC,
    PROCESSED_DIR,
    MODEL_DIR,
    SEQ_LEN,
)

# Binance interval string (e.g. "5m")
INTERVAL_STR = INTERVAL

# Cache for models and scalers so we don't reload on every inference
_model_cache: Dict[str, tf.keras.Model] = {}
_scaler_cache: Dict[str, object] = {}


def _load_model_and_scaler(asset: str):
    """
    Load (or reuse cached) GRU model and scaler for the given asset symbol,
    where asset is "BTC", "ETH", "TAO".
    """
    if asset in _model_cache:
        return _model_cache[asset], _scaler_cache[asset]

    model_path_best = os.path.join(MODEL_DIR, f"{asset.lower()}_gru_best.h5")
    model_path_final = os.path.join(MODEL_DIR, f"{asset.lower()}_gru_final.h5")

    if os.path.exists(model_path_best):
        model_path = model_path_best
    elif os.path.exists(model_path_final):
        model_path = model_path_final
    else:
        raise RuntimeError(f"Model for {asset} not found. Checked:\n" f"  {model_path_best}\n" f"  {model_path_final}")

    scaler_path = os.path.join(PROCESSED_DIR, f"{asset.lower()}_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler not found for {asset}: {scaler_path}")

    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    _model_cache[asset] = model
    _scaler_cache[asset] = scaler
    return model, scaler


def _binance_latest_window(asset: str, lookback_steps: int = SEQ_LEN) -> pd.DataFrame:
    """
    Fetch the latest `lookback_steps` candles for the asset from Binance spot.

    We simply request a limited number of latest klines (no startTime), which
    Binance returns in ascending order, most recent last.
    """
    if asset not in BINANCE_SYMBOLS:
        raise ValueError(f"Unknown asset for Binance: {asset}")

    symbol = BINANCE_SYMBOLS[asset]
    limit = max(lookback_steps + 10, 100)  # small safety margin

    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": INTERVAL_STR,
        "limit": limit,
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    klines = r.json()

    if not klines:
        raise RuntimeError(f"No kline data returned from Binance for {asset}")

    rows = []
    for k in klines:
        open_time = int(k[0])
        open_ = float(k[1])
        high = float(k[2])
        low = float(k[3])
        close = float(k[4])
        volume = float(k[5])
        rows.append((open_time, open_, high, low, close, volume))

    df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

    # Deduplicate & keep chronological order
    df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms").reset_index(drop=True)

    # Add seconds timestamp + datetime
    df["timestamp"] = (df["timestamp_ms"] // 1000).astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    if len(df) < lookback_steps:
        raise RuntimeError(
            f"Not enough recent candles for {asset}: " f"len(df)={len(df)} < lookback_steps={lookback_steps}"
        )

    # Return just the last `lookback_steps` rows
    return df.tail(lookback_steps).reset_index(drop=True)


def predict_1h_price(asset: str) -> Tuple[float, float]:
    """
    Predict price 1 hour ahead (12 x 5m steps) for the given asset using
    the trained GRU model.

    Args:
        asset: "BTC", "ETH", or "TAO" (upper-case).

    Returns:
        (current_price, predicted_price_1h)
    """
    asset = asset.upper()
    model, scaler = _load_model_and_scaler(asset)

    # Fetch latest candles from Binance
    df = _binance_latest_window(asset, lookback_steps=SEQ_LEN)

    # Feature engineering must mirror `preprocess.make_supervised`
    df["log_close"] = np.log(df["close"])
    df["ret_1"] = df["log_close"].diff()
    df["ret_3"] = df["log_close"].diff(3)
    df["ret_6"] = df["log_close"].diff(6)
    df["vol_log"] = np.log1p(df["volume"])
    df = df.dropna().reset_index(drop=True)

    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Not enough rows after feature dropna for {asset}: " f"{len(df)} < SEQ_LEN={SEQ_LEN}")

    feature_cols = ["log_close", "ret_1", "ret_3", "ret_6", "vol_log"]
    feat = df[feature_cols].values
    feat_scaled = scaler.transform(feat)

    # Take last SEQ_LEN rows as the input window
    x = feat_scaled[-SEQ_LEN:]
    x = np.expand_dims(x, axis=0)  # (1, seq_len, n_features)

    # Model outputs relative change for 1h ahead
    rel_change = float(model.predict(x, verbose=0)[0, 0])

    current_price = float(df["close"].iloc[-1])
    predicted_price = current_price * (1.0 + rel_change)

    return current_price, predicted_price
