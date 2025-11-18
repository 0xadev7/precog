# price_model/inference.py

import os
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
    SEQ_LEN,
    HORIZON_STEPS,
    PROCESSED_DIR,
    MODEL_DIR,
    INTERVAL_SEC,
)

# ---------------------------------------------------------------------
# Caches (so repeated calls stay well under your 12s budget)
# ---------------------------------------------------------------------

_MODEL_CACHE: Dict[str, tf.keras.Model] = {}
_SCALER_CACHE: Dict[str, object] = {}


def _normalize_asset_key(asset: str) -> str:
    """
    Map various incoming asset strings to our canonical keys.
    """
    a = asset.upper()
    # Handle TAO_BITTENSOR etc.
    if a in ("TAO_BITTENSOR", "TAO-BITTENSOR"):
        return "TAO"
    return a


def _load_model_and_scaler(asset: str) -> Tuple[tf.keras.Model, object]:
    key = _normalize_asset_key(asset)
    if key not in _MODEL_CACHE:
        model_path = os.path.join(MODEL_DIR, f"{key.lower()}_gru_final.h5")
        scaler_path = os.path.join(PROCESSED_DIR, f"{key.lower()}_scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model for {key}: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing scaler for {key}: {scaler_path}")

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        _MODEL_CACHE[key] = model
        _SCALER_CACHE[key] = scaler

    return _MODEL_CACHE[key], _SCALER_CACHE[key]


# ---------------------------------------------------------------------
# Binance fetch & feature engineering
# ---------------------------------------------------------------------

# Must match FEATURE_COLS in preprocess.py
FEATURE_COLS = [
    "log_close",
    "ret_1",
    "ret_3",
    "ret_6",
    "vol_log",
    "hl_spread",
    "oc_change",
    "rolling_vol_1h",
    "rolling_ret_1h",
    "rolling_vol_4h",
    "rolling_ret_4h",
    "ma_ratio",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def _fetch_latest_klines(asset: str, limit: int = 500) -> pd.DataFrame:
    key = _normalize_asset_key(asset)
    if key not in BINANCE_SYMBOLS:
        raise ValueError(f"Unknown asset for Binance: {asset}")

    symbol = BINANCE_SYMBOLS[key]
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError(f"No kline data from Binance for {symbol}")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["timestamp_ms"] = df["open_time"].astype("int64")
    df["timestamp"] = (df["timestamp_ms"] // 1000).astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df["log_close"] = np.log(df["close"].astype(float))
    df["ret_1"] = df["log_close"].diff(1)
    df["ret_3"] = df["log_close"].diff(3)
    df["ret_6"] = df["log_close"].diff(6)

    df["vol_log"] = np.log(df["volume"].astype(float) + 1e-8)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_change"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    df["rolling_vol_1h"] = df["log_close"].rolling(12).std()
    df["rolling_ret_1h"] = df["log_close"].diff(12)
    df["rolling_vol_4h"] = df["log_close"].rolling(48).std()
    df["rolling_ret_4h"] = df["log_close"].diff(48)

    df["ma_fast"] = df["close"].rolling(12).mean()
    df["ma_slow"] = df["close"].rolling(36).mean()
    df["ma_ratio"] = df["ma_fast"] / df["ma_slow"].replace(0, np.nan) - 1.0

    dt = pd.to_datetime(df["datetime"], utc=True)
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    return df


def _build_feature_window(df_raw: pd.DataFrame, asset: str) -> Tuple[np.ndarray, float]:
    """
    Build the last SEQ_LEN feature window and return (features, current_price).

    features shape: (SEQ_LEN, n_features)
    """
    df = _build_features(df_raw)

    # Drop rows with NaNs in any feature or in close
    df = df.dropna(subset=FEATURE_COLS + ["close"]).reset_index(drop=True)
    if len(df) < SEQ_LEN:
        raise RuntimeError(f"Not enough rows after feature engineering for {asset}: " f"{len(df)} < SEQ_LEN={SEQ_LEN}")

    # Latest window
    df_win = df.iloc[-SEQ_LEN:]
    feat = df_win[FEATURE_COLS].to_numpy(dtype=np.float32)

    current_price = float(df_win["close"].iloc[-1])

    return feat, current_price


# ---------------------------------------------------------------------
# Public API: used by gru_miner.predict_1h_price
# ---------------------------------------------------------------------


def predict_1h_price(asset: str) -> Tuple[float, float]:
    """
    Predict 1h-ahead price for `asset` using Binance public data
    and the trained GRU model.

    Returns:
        (current_price, predicted_price_1h)
    """
    asset_key = _normalize_asset_key(asset)
    model, scaler = _load_model_and_scaler(asset_key)

    # Fetch recent history; limit big enough to cover SEQ_LEN + roll windows
    df_raw = _fetch_latest_klines(asset_key, limit=max(SEQ_LEN + 60, 200))

    feat, current_price = _build_feature_window(df_raw, asset_key)
    feat_scaled = scaler.transform(feat)

    # Shape (1, SEQ_LEN, n_features)
    x = np.expand_dims(feat_scaled, axis=0)

    # Model outputs relative change (future/current - 1)
    rel_change = float(model.predict(x, verbose=0)[0, 0])
    predicted_price = current_price * (1.0 + rel_change)

    return current_price, predicted_price
