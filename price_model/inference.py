# price_model/inference.py

import os
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import requests
import joblib
import tensorflow as tf

from .config import (
    GATE_BASE,
    ASSETS,
    INTERVAL,
    SEQ_LEN,
    PROCESSED_DIR,
    MODEL_DIR,
)

# Cache for models and scalers so we don't reload on every inference
_model_cache: Dict[str, tf.keras.Model] = {}
_scaler_cache: Dict[str, object] = {}


def _load_model_and_scaler(asset: str):
    """
    Load (or reuse cached) GRU model and scaler for the given asset symbol,
    where asset is 'BTC', 'ETH', or 'TAO'.
    """
    asset = asset.upper()
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

    # Avoid deserializing training-time metrics (mse, etc.)
    model = tf.keras.models.load_model(model_path, compile=False)

    _model_cache[asset] = model
    _scaler_cache[asset] = scaler
    return model, scaler


def _gate_latest_window(asset: str, lookback_steps: int = SEQ_LEN) -> pd.DataFrame:
    """
    Fetch the latest `lookback_steps` candles for the asset from Gate.io spot.
    """
    asset = asset.lower()
    if asset.upper() not in ASSETS:
        raise ValueError(f"Unknown asset for Gate.io: {asset}")

    symbol = ASSETS[asset.upper()]["symbol"]

    # Ask for *significantly* more than SEQ_LEN as safety margin, because
    # feature engineering (diffs) will drop some rows.
    limit = max(lookback_steps * 2, lookback_steps + 40, 120)

    url = f"{GATE_BASE}/spot/candlesticks"
    params = {
        "currency_pair": symbol,
        "interval": INTERVAL,
        "limit": limit,
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if not data:
        raise RuntimeError(f"No candlestick data from Gate.io for {asset} ({symbol})")

    rows = []
    # Gate returns newest first; we'll sort ascending by timestamp
    for row in data:
        # row schema: [t, v, c, h, l, o, ...]
        ts = int(row[0])
        vol = float(row[1])
        close = float(row[2])
        high = float(row[3])
        low = float(row[4])
        open_ = float(row[5])
        rows.append((ts, open_, high, low, close, vol))

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    if len(df) < lookback_steps:
        raise RuntimeError(
            f"Not enough recent candles for {asset}: " f"len(df)={len(df)} < lookback_steps={lookback_steps}"
        )

    # Only keep the last lookback_steps rows (for raw candles)
    df = df.tail(lookback_steps).reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def predict_1h_price(asset: str) -> Tuple[float, float]:
    asset = asset.upper()
    model, scaler = _load_model_and_scaler(asset)

    # Fetch latest candles from Gate.io
    df = _gate_latest_window(asset, lookback_steps=SEQ_LEN)

    # Feature engineering
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

    x = feat_scaled[-SEQ_LEN:]
    x = np.expand_dims(x, axis=0)

    rel_change = float(model.predict(x, verbose=0)[0, 0])

    current_price = float(df["close"].iloc[-1])
    predicted_price = current_price * (1.0 + rel_change)

    return current_price, predicted_price
