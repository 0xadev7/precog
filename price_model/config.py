# price_model/config.py

import os

# ---- Binance spot config ----
GATE_BASE = "https://api.gateio.ws/api/v4"
ASSETS = {
    "BTC": {"symbol": "BTC_USDT"},
    "ETH": {"symbol": "ETH_USDT"},
    "TAO": {"symbol": "TAO_USDT"},  # adjust if TAO symbol differs
}

# Candle interval
INTERVAL = "5m"  # 5-minute candles

INTERVAL_SEC_MAP = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}
INTERVAL_SEC = INTERVAL_SEC_MAP[INTERVAL]

# ---- ML / dataset config ----
LOOKBACK_DAYS = 90  # history for training
SEQ_LEN = 60  # input sequence length (60 * 5m = 5h)
HORIZON_STEPS = 12  # prediction horizon (12 * 5m = 1h)

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))

RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "models")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
