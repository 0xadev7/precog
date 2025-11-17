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


def _gate_latest_window(asset: str, min_rows: int = SEQ_LEN + 20) -> pd.DataFrame:
    """
    Fetch the latest candles for the asset from Gate.io spot.

    We request significantly more than SEQ_LEN candles so that after
    feature engineering and dropna() we still have >= SEQ_LEN rows.
    """
    asset_lower = asset.lower()
    asset_upper = asset.upper()
    if asset_upper not in ASSETS:
        raise ValueError(f"Unknown asset for Gate.io: {asset}")

    symbol = ASSETS[asset_upper]["symbol"]

    # Ask for plenty of candles to survive diff() and dropna()
    limit = max(min_rows * 2, SEQ_LEN * 3, 180)

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

    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Not enough recent candles for {asset}: len(df)={len(df)} < SEQ_LEN={SEQ_LEN}")

    # Do NOT cut to SEQ_LEN here; we keep more and trim after feature engineering
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def _build_feature_window(df: pd.DataFrame, asset: str) -> Tuple[np.ndarray, float]:
    """
    Apply feature engineering mirroring training, and produce the
    last SEQ_LEN feature rows.

    If we end up with slightly fewer than SEQ_LEN rows, pad the
    sequence at the front with the earliest row to reach SEQ_LEN.
    """
    # Feature engineering must mirror preprocess.make_supervised
    df = df.copy()
    df["log_close"] = np.log(df["close"])
    df["ret_1"] = df["log_close"].diff()
    df["ret_3"] = df["log_close"].diff(3)
    df["ret_6"] = df["log_close"].diff(6)
    df["vol_log"] = np.log1p(df["volume"])
    df = df.dropna().reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"No rows left after feature engineering for {asset}")

    # Use the last SEQ_LEN rows if possible; otherwise pad
    if len(df) >= SEQ_LEN:
        df_win = df.tail(SEQ_LEN).reset_index(drop=True)
    else:
        # Pad at the front with the earliest row to reach SEQ_LEN
        missing = SEQ_LEN - len(df)
        first_row = df.iloc[0:1].copy()
        pad_df = pd.concat([first_row] * missing, ignore_index=True)
        df_win = pd.concat([pad_df, df], ignore_index=True)

    # Current price is the last close in the (possibly padded) window
    current_price = float(df_win["close"].iloc[-1])

    feature_cols = ["log_close", "ret_1", "ret_3", "ret_6", "vol_log"]
    feat = df_win[feature_cols].values.astype(np.float32)

    return feat, current_price


def predict_1h_price(asset: str) -> Tuple[float, float]:
    """
    Predict price 1 hour ahead (12 x 5m steps) for the given asset using
    the trained GRU model.

    Args:
        asset: 'BTC', 'ETH', or 'TAO' (case-insensitive).

    Returns:
        (current_price, predicted_price_1h)
    """
    asset = asset.upper()
    model, scaler = _load_model_and_scaler(asset)

    # Fetch latest candles from Gate.io
    df_raw = _gate_latest_window(asset, min_rows=SEQ_LEN + 20)

    # Build feature window and get current price
    feat, current_price = _build_feature_window(df_raw, asset)

    # Scale features with training scaler
    feat_scaled = scaler.transform(feat)

    # Shape (1, seq_len, n_features)
    x = np.expand_dims(feat_scaled, axis=0)

    # Model outputs relative change for 1h ahead
    rel_change = float(model.predict(x, verbose=0)[0, 0])

    predicted_price = current_price * (1.0 + rel_change)

    return current_price, predicted_price
