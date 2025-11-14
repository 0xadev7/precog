import asyncio
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import bittensor as bt

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import to_datetime, to_str

# ------------------------------------------------------------
# Optional ML & NLP dependencies (graceful fallback if missing)
# ------------------------------------------------------------
try:
    from sklearn.linear_model import RidgeCV

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from xgboost import XGBRegressor

    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    VADER_OK = True
except Exception:
    VADER_OK = False

# -----------------------------
# Configuration and constants
# -----------------------------
BINANCE_BASE = "https://api.binance.com"
REDDIT_BASE = "https://www.reddit.com"
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"

USER_AGENT = os.environ.get("MINER_UA", "Eric-BTC-Miner/2.0 (+https://github.com)")

# Forecast horizon in minutes for the supervised target
HORIZON_MIN = int(os.environ.get("HORIZON_MIN", "60"))
# How many days of 1-min training data
LOOKBACK_DAYS = float(os.environ.get("LOOKBACK_DAYS", "2"))
# Cap for sentiment drift as a multiple of hourly vol
SENTIMENT_IMPACT_CAP = float(os.environ.get("SENTIMENT_IMPACT_CAP", "0.25"))

# Derivatives config
USE_DERIVATIVES = os.environ.get("USE_DERIVATIVES", "1") == "1"
MAX_OI_MARKETS = int(os.environ.get("MAX_OI_MARKETS", "5"))  # markets per asset
DERIV_LOOKBACK_DAYS = float(os.environ.get("DERIV_LOOKBACK_DAYS", "3"))


# -----------------------------
# Technical indicator utilities
# -----------------------------
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = _ema(series, fast) - _ema(series, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bbands(series: pd.Series, window: int = 20, ndev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window=window, min_periods=window).mean()
    sd = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + ndev * sd
    lower = ma - ndev * sd
    return lower, ma, upper


def realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    return returns.rolling(window=window, min_periods=max(5, window // 3)).std(ddof=0)


# -----------------------------
# External data fetchers (free)
# -----------------------------
def binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000) -> List[List]:
    """Fetch klines from Binance (no API key needed)."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    out: List[List] = []
    cur = start_ms
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            bt.logging.warning(f"Binance klines HTTP {r.status_code}: {r.text[:120]}")
            break
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        last_open_time = batch[-1][0]
        next_start = last_open_time + 1
        if next_start >= end_ms or len(batch) < limit:
            break
        cur = next_start
        time.sleep(0.2)
    return out


def fetch_reddit_posts(subreddits: List[str], limit: int = 100) -> List[str]:
    """Fetch recent Reddit posts (titles + selftext) for crypto subs."""
    headers = {"User-Agent": USER_AGENT}
    texts: List[str] = []
    subs_joined = "+".join(subreddits)
    url = f"{REDDIT_BASE}/r/{subs_joined}/new.json"
    try:
        r = requests.get(url, params={"limit": min(limit, 100)}, headers=headers, timeout=10)
        if r.status_code != 200:
            bt.logging.warning(f"Reddit HTTP {r.status_code}: {r.text[:120]}")
            return texts
        data = r.json()
        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            title = post.get("title", "") or ""
            body = post.get("selftext", "") or ""
            txt = (title + " " + body).strip()
            if txt:
                texts.append(txt)
    except Exception as e:
        bt.logging.warning(f"Reddit fetch error: {e}")
    return texts


def fetch_stocktwits_messages(symbol: str = "BTC.X", limit: int = 30) -> List[str]:
    """Fetch recent messages from Stocktwits public REST (no key)."""
    url = f"{STOCKTWITS_BASE}/streams/symbol/{symbol}.json"
    try:
        r = requests.get(url, params={"limit": min(limit, 30)}, headers={"User-Agent": USER_AGENT}, timeout=10)
        if r.status_code != 200:
            bt.logging.warning(f"Stocktwits HTTP {r.status_code}: {r.text[:120]}")
            return []
        data = r.json()
        out = []
        for msg in data.get("messages", []):
            body = msg.get("body", "")
            if isinstance(body, str) and body:
                out.append(body)
        return out
    except Exception as e:
        bt.logging.warning(f"Stocktwits fetch error: {e}")
        return []


# -----------------------------
# Sentiment (VADER w/ fallback)
# -----------------------------
_POS_WORDS = {
    "bull",
    "bullish",
    "moon",
    "pump",
    "surge",
    "breakout",
    "rip",
    "rocket",
    "ath",
    "undervalued",
    "support",
    "rally",
    "green",
    "accumulate",
    "buy",
    "long",
}

_NEG_WORDS = {
    "bear",
    "bearish",
    "dump",
    "crash",
    "plunge",
    "nuke",
    "rekt",
    "resistance",
    "red",
    "sell",
    "short",
    "liquidation",
    "fear",
    "uncertain",
    "doubt",
    "fud",
}


def _lexicon_sentiment(texts: List[str]) -> float:
    if not texts:
        return 0.0
    score = 0
    count = 0
    for t in texts:
        t_low = t.lower()
        s = 0
        for w in _POS_WORDS:
            if w in t_low:
                s += 1
        for w in _NEG_WORDS:
            if w in t_low:
                s -= 1
        if s != 0:
            score += s
            count += 1
    if count == 0:
        return 0.0
    return float(np.tanh(score / max(1, count)))


def compute_social_sentiment() -> Dict[str, float]:
    """Return {"score": [-1,1], "n_posts": int} combining Reddit + Stocktwits."""
    subreddits = ["Bitcoin", "CryptoCurrency", "CryptoMarkets"]
    reddit_texts = fetch_reddit_posts(subreddits, limit=100)
    st_texts = fetch_stocktwits_messages("BTC.X", limit=30)
    texts = reddit_texts + st_texts

    if VADER_OK:
        try:
            analyzer = SentimentIntensityAnalyzer()
            if not texts:
                score = 0.0
            else:
                comp = [analyzer.polarity_scores(t).get("compound", 0.0) for t in texts]
                score = float(np.median(comp))
        except Exception as e:
            bt.logging.warning(f"VADER error, falling back to lexicon: {e}")
            score = _lexicon_sentiment(texts)
    else:
        score = _lexicon_sentiment(texts)

    return {"score": float(np.clip(score, -1, 1)), "n_posts": int(len(texts))}


# -----------------------------
# Derivatives via CoinMetrics
# -----------------------------
def get_derivatives_markets(cm: CMData, base: str = "btc", quote: str = "usd") -> List[str]:
    """Pull a small set of active futures markets for the asset from CoinMetrics."""
    try:
        catalog = cm.get_open_interest_catalog(base=base, quote=quote, market_type="future")
        if catalog.empty or "market" not in catalog.columns:
            return []
        # Prefer most recently active markets (max_time near end)
        catalog = catalog.sort_values("max_time", ascending=False).reset_index(drop=True)
        markets = catalog["market"].head(MAX_OI_MARKETS).tolist()
        return markets
    except Exception as e:
        bt.logging.warning(f"Failed to fetch OI catalog for {base}-{quote}: {e}")
        return []


def get_derivatives_timeseries(
    cm: CMData, base_asset: str, start: pd.Timestamp, end: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """Aggregate open interest and funding rates across futures markets."""
    if not USE_DERIVATIVES:
        return None

    base = base_asset.lower().replace("_bittensor", "")
    markets = get_derivatives_markets(cm, base=base, quote="usd")
    if not markets:
        return None

    start_time = to_str(start - pd.Timedelta(days=DERIV_LOOKBACK_DAYS))
    end_time = to_str(end)

    oi_df = pd.DataFrame()
    fr_df = pd.DataFrame()

    try:
        oi_df = cm.get_market_open_interest(
            markets,
            page_size=10000,
            parallelize=False,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        bt.logging.warning(f"Open interest fetch failed for {base}: {e}")

    try:
        fr_df = cm.get_market_funding_rates(
            markets,
            page_size=10000,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        bt.logging.warning(f"Funding rates fetch failed for {base}: {e}")

    if (oi_df is None or oi_df.empty) and (fr_df is None or fr_df.empty):
        return None

    frames = []

    if oi_df is not None and not oi_df.empty:
        df = oi_df.copy()
        if "time" not in df.columns:
            bt.logging.warning("OI dataframe missing time column")
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            # Sum value_usd across markets
            val_col = "value_usd" if "value_usd" in df.columns else None
            if val_col is not None:
                agg = df.groupby("time")[val_col].sum().to_frame("oi_value_usd")
                frames.append(agg)

    if fr_df is not None and not fr_df.empty:
        df = fr_df.copy()
        if "time" not in df.columns:
            bt.logging.warning("Funding dataframe missing time column")
        else:
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            # Try to infer the funding rate column
            funding_col = None
            for c in df.columns:
                if "fund" in c.lower() and df[c].dtype != "O":
                    funding_col = c
                    break
            if funding_col is not None:
                agg = df.groupby("time")[funding_col].mean().to_frame("funding_rate")
                frames.append(agg)

    if not frames:
        return None

    deriv = pd.concat(frames, axis=1).sort_index()
    # Drop duplicates indices (if any) by keeping last
    deriv = deriv[~deriv.index.duplicated(keep="last")]
    return deriv


# -----------------------------
# Feature engineering
# -----------------------------
def build_minute_features(price_df: pd.DataFrame, deriv_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    price_df: 1-min OHLC with datetime index and at least 'close'.
    deriv_df: optional derivatives features with datetime index (will be resampled -> 1T).
    """
    df = price_df.copy().sort_index()
    out = pd.DataFrame(index=df.index)
    out["close"] = df["close"]

    # Returns
    out["ret_1m"] = out["close"].pct_change().fillna(0.0)
    out["ret_5m"] = out["close"].pct_change(5).fillna(0.0)
    out["ret_15m"] = out["close"].pct_change(15).fillna(0.0)

    # Vol
    out["vol_30m"] = realized_vol(out["ret_1m"], window=30).bfill().fillna(0.0)
    out["vol_60m"] = realized_vol(out["ret_1m"], window=60).bfill().fillna(0.0)

    # Trend / momentum
    out["ema_7"] = _ema(out["close"], 7)
    out["ema_21"] = _ema(out["close"], 21)
    out["ema_ratio"] = (out["ema_7"] / (out["ema_21"] + 1e-9)) - 1.0

    # RSI, MACD
    out["rsi_14"] = rsi(out["close"], 14).bfill().fillna(50.0)
    macd_line, macd_signal, macd_hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_sig"] = macd_signal
    out["macd_hist"] = macd_hist

    # Bollinger band z-score
    lower, ma, upper = bbands(out["close"], 20, 2.0)
    out["bb_z"] = (out["close"] - ma) / (upper - lower + 1e-9)

    # Derivatives features (optional)
    if deriv_df is not None and not deriv_df.empty:
        # Resample derivatives to 1-min and forward fill
        deriv_resampled = deriv_df.sort_index().resample("1T").ffill()
        out = out.join(deriv_resampled, how="left")

        if "oi_value_usd" in out.columns:
            out["oi_log"] = np.log(out["oi_value_usd"].clip(lower=1.0))
            out["oi_log_roc_60m"] = out["oi_log"].diff(60)
            out["oi_log_roc_240m"] = out["oi_log"].diff(240)
            rolling_mean = out["oi_log"].rolling(60).mean()
            rolling_std = out["oi_log"].rolling(60).std(ddof=0)
            out["oi_z"] = (out["oi_log"] - rolling_mean) / (rolling_std + 1e-9)

        if "funding_rate" in out.columns:
            out["funding_norm"] = out["funding_rate"]
            out["funding_roc_24h"] = out["funding_rate"].diff(24 * 60)

    # Clean
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.bfill().fillna(method="ffill").dropna()
    return out


def make_supervised(features: pd.DataFrame, horizon_min: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised dataset where target is log-return over the next horizon.
    """
    close = features["close"].copy()
    target = np.log(close.shift(-horizon_min) / (close + 1e-9))
    X = features.drop(columns=["close"]).copy()
    X, y = X.iloc[:-horizon_min], target.iloc[:-horizon_min]
    y = y.fillna(0.0)
    return X, y


# -----------------------------
# Prediction interval (vol-based)
# -----------------------------
def calculate_prediction_interval(
    point_estimate: float, historical_prices: pd.Series, asset: str = "btc"
) -> Tuple[float, float]:
    """
    Volatility-based 99% prediction interval scaled to the forecast horizon.
    """
    try:
        if historical_prices.empty or len(historical_prices) < 60:
            margin = point_estimate * 0.10
            return point_estimate - margin, point_estimate + margin

        rets = historical_prices.pct_change().dropna()
        if rets.empty:
            margin = point_estimate * 0.10
            return point_estimate - margin, point_estimate + margin

        vol_60m = rets.rolling(60).std().iloc[-1]
        if pd.isna(vol_60m) or not np.isfinite(vol_60m):
            vol_60m = rets.std()

        if not np.isfinite(vol_60m) or vol_60m <= 0:
            margin = point_estimate * 0.10
            return point_estimate - margin, point_estimate + margin

        vol_60m = float(vol_60m)

        z = 2.58  # ~99%
        scaled_vol = vol_60m * math.sqrt(max(1, HORIZON_MIN) / 60.0)
        margin = point_estimate * z * scaled_vol

        max_margin = point_estimate * 0.30
        min_margin = point_estimate * 0.02
        margin = max(min_margin, min(margin, max_margin))

        lower = point_estimate - margin
        upper = point_estimate + margin
        bt.logging.debug(f"{asset}: hourly_vol={vol_60m:.4f}, margin=${margin:.2f}")
        return lower, upper
    except Exception as e:
        bt.logging.error(f"Error calculating interval for {asset}: {e}")
        margin = point_estimate * 0.15
        return point_estimate - margin, point_estimate + margin


# -----------------------------
# Price history via CM (primary) + Binance (fallback)
# -----------------------------
def get_price_history(asset: str, start: pd.Timestamp, end: pd.Timestamp, cm: Optional[CMData]) -> pd.DataFrame:
    """
    Returns 1-min OHLC dataframe for the asset in USD.
    Primary: CoinMetrics ReferenceRate (1s -> 1T OHLC).
    Fallback: Binance spot pair (USDT).
    """
    asset = asset.lower()

    # CoinMetrics ReferenceRate (close-only -> synthetic OHLC)
    try:
        if cm is not None:
            df = cm.get_CM_ReferenceRate(
                assets=[asset],
                start=to_str(start),
                end=to_str(end),
                end_inclusive=True,
                frequency="1s",
                page_size=10000,
                parallelize=False,
                use_cache=False,
            )
            if not df.empty and {"time", "ReferenceRateUSD"}.issubset(df.columns):
                df = (
                    df[df["asset"] == asset][["time", "ReferenceRateUSD"]]
                    .rename(columns={"ReferenceRateUSD": "close"})
                    .copy()
                )
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time").sort_index()
                ohlc = df["close"].resample("1min").ohlc()
                ohlc = ohlc.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
                ohlc = ohlc.dropna()
                if not ohlc.empty:
                    return ohlc
    except Exception as e:
        bt.logging.warning(f"CoinMetrics price fetch failed for {asset}, will try Binance: {e}")

    # Binance fallback
    symbol_map = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "tao_bittensor": "TAOUSDT",
    }
    symbol = symbol_map.get(asset, f"{asset.upper()}USDT")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    kl = binance_klines(symbol, "1m", start_ms, end_ms, limit=1000)
    if not kl:
        raise RuntimeError(f"No price history from CM or Binance for {asset}")
    df = pd.DataFrame(
        kl,
        columns=[
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "closeTime",
            "qav",
            "num",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df = df.set_index("time")[["open", "high", "low", "close"]].sort_index()
    return df


# -----------------------------
# Core ML model
# -----------------------------
def _fit_predict_return(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Train a small but strong model and return latest prediction (float log-return).
    Priority: XGBoost (if installed) -> RidgeCV -> numpy ridge.
    """
    mask = ~(X.isna().any(axis=1) | y.isna())
    Xc = X[mask]
    yc = y[mask]

    if len(Xc) < 300:
        # Not enough history; assume flat
        return 0.0

    x_train = Xc.iloc[:-1].values
    y_train = yc.iloc[:-1].values
    x_last = Xc.iloc[-1:].values

    # 1) Gradient boosted trees (optional)
    if XGB_OK:
        try:
            model = XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=1,
            )
            model.fit(x_train, y_train)
            pred = float(model.predict(x_last)[0])
            return pred
        except Exception as e:
            bt.logging.warning(f"XGBRegressor failed, falling back to RidgeCV: {e}")

    # 2) RidgeCV (if sklearn available)
    if SKLEARN_OK:
        try:
            model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], store_cv_values=False)
            model.fit(x_train, y_train)
            pred = float(model.predict(x_last)[0])
            return pred
        except Exception as e:
            bt.logging.warning(f"RidgeCV failed, falling back to numpy ridge: {e}")

    # 3) Manual ridge via numpy
    lam = 1.0
    XtX = x_train.T @ x_train
    XtX_reg = XtX + lam * np.eye(XtX.shape[0])
    w = np.linalg.pinv(XtX_reg) @ x_train.T @ y_train
    pred = float(x_last @ w)
    return pred


def _apply_sentiment_adjustment(pred_return: float, vol_1h: float, sentiment: Dict[str, float]) -> float:
    s = float(np.clip(sentiment.get("score", 0.0), -1.0, 1.0))
    impact = float(np.clip(s * vol_1h, -SENTIMENT_IMPACT_CAP * vol_1h, SENTIMENT_IMPACT_CAP * vol_1h))
    return pred_return + impact


# ------------------------------------------------------------
# Public API: async forward (signature unchanged)
# ------------------------------------------------------------
async def forward_async(synapse: Challenge, cm: CMData) -> Challenge:
    total_start = time.perf_counter()

    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [a.lower() for a in raw_assets]

    bt.logging.info(
        f"👈 Received prediction request from: {getattr(synapse.dendrite, 'hotkey', 'unknown')} "
        f"for {assets} at timestamp: {synapse.timestamp}"
    )

    ts_provided = to_datetime(synapse.timestamp)
    end_ts = pd.to_datetime(ts_provided, utc=True)
    start_ts = end_ts - pd.Timedelta(days=LOOKBACK_DAYS)

    predictions: Dict[str, float] = {}
    intervals: Dict[str, List[float]] = {}

    # Compute BTC-focused sentiment once per forward
    sentiment = compute_social_sentiment()
    bt.logging.info(
        f"Social sentiment: score={sentiment['score']:.3f} from {sentiment['n_posts']} posts "
        f"(VADER={'on' if VADER_OK else 'off'})"
    )

    for asset in assets:
        try:
            # 1) Price history
            px = get_price_history(asset, start_ts, end_ts, cm)
            if px.empty or "close" not in px.columns:
                raise RuntimeError("Price history empty")

            # 2) Derivatives time series (Open Interest + Funding)
            deriv_df = None
            if USE_DERIVATIVES and cm is not None and hasattr(cm, "client"):
                deriv_df = get_derivatives_timeseries(cm, asset, start_ts, end_ts)

            # 3) Features + supervised dataset
            feats = build_minute_features(px, deriv_df=deriv_df)
            X, y = make_supervised(feats, horizon_min=HORIZON_MIN)

            # 4) Technical/derivatives model -> next-horizon log return
            pred_ret = _fit_predict_return(X, y)

            # 5) Sentiment adjustment
            vol_1h = feats["ret_1m"].rolling(60).std().iloc[-1]
            if pd.isna(vol_1h) or not np.isfinite(vol_1h):
                vol_1h = feats["ret_1m"].std()
            if not np.isfinite(vol_1h) or vol_1h <= 0:
                vol_1h = 0.0
            vol_1h = float(vol_1h)

            pred_ret_adj = _apply_sentiment_adjustment(pred_ret, vol_1h, sentiment)

            # 6) Convert to price prediction
            last_price = float(feats["close"].iloc[-1])
            point_estimate = float(last_price * math.exp(pred_ret_adj))

            # 7) Interval
            interval = calculate_prediction_interval(point_estimate, feats["close"], asset)

            predictions[asset] = point_estimate
            intervals[asset] = [float(interval[0]), float(interval[1])]

            bt.logging.info(
                f"{asset.upper()} last=${last_price:,.2f} pred_ret={pred_ret_adj:.5f} -> "
                f"pred=${point_estimate:,.2f} [{interval[0]:,.2f}, {interval[1]:,.2f}]"
            )
        except Exception as e:
            bt.logging.error(f"{asset}: prediction error -> {e}")
            # Fallback: nowcast with wide band if we have any price
            try:
                if "close" in locals().get("feats", pd.DataFrame()).columns:
                    last_price = float(feats["close"].iloc[-1])  # type: ignore
                elif "px" in locals() and not px.empty:  # type: ignore
                    last_price = float(px["close"].iloc[-1])  # type: ignore
                else:
                    last_price = 0.0
            except Exception:
                last_price = 0.0

            if last_price <= 0:
                continue

            lo, hi = calculate_prediction_interval(
                last_price,
                feats["close"] if "feats" in locals() and "close" in feats.columns else pd.Series(dtype=float),
                asset,
            )
            predictions[asset] = last_price
            intervals[asset] = [float(lo), float(hi)]

    synapse.predictions = predictions
    synapse.intervals = intervals

    elapsed = time.perf_counter() - total_start
    bt.logging.debug(f"⏱️ forward() completed in {elapsed:.3f}s")

    if synapse.predictions:
        bt.logging.success(f"Predictions ready for: {list(predictions.keys())}")
    else:
        bt.logging.warning("No predictions generated.")
    return synapse


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    return await forward_async(synapse, cm)
