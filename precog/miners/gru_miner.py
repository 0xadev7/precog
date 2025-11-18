import time
from typing import Tuple, Optional

import bittensor as bt
import pandas as pd
import numpy as np
import httpx

from precog.protocol import Challenge
from precog.utils.cm_data import CMData  # kept for type-compat in forward/forward_async

# 🔮 GRU model inference (you need price_model on PYTHONPATH)
from price_model.inference import predict_1h_price

BINANCE_BASE_URL = "https://api.binance.com"


def _asset_to_binance_symbol(asset: str) -> str:
    """
    Map internal asset name to Binance symbol (USDT pairs by default).
    """
    a = asset.lower()
    mapping = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "etc": "ETCUSDT",
        "tao": "TAOUSDT",  # if not listed, fetch will fail and we handle it
    }
    return mapping.get(a, f"{asset.upper()}USDT")


async def _fetch_binance_klines(
    symbol: str,
    interval: str = "5m",
    limit: int = 200,
    timeout: float = 10.0,
) -> pd.Series:
    """
    Fetch recent Binance klines and return a Series of close prices.

    Args:
        symbol: e.g. 'BTCUSDT'
        interval: Binance kline interval (default '5m')
        limit: number of candles to fetch (default 200 -> ~16.6h for 5m)
        timeout: request timeout in seconds

    Returns:
        pd.Series of close prices indexed by close time (UTC).
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    if not data:
        raise ValueError(f"Empty kline response from Binance for {symbol}")

    closes = [float(candle[4]) for candle in data]  # close price
    close_times = [int(candle[6]) for candle in data]  # close time in ms

    close_index = pd.to_datetime(close_times, unit="ms", utc=True)
    return pd.Series(closes, index=close_index)


def _asset_caps(asset: str) -> Tuple[float, float]:
    """
    Asset-specific max hourly move caps (for both the point estimate guard
    and interval width).
    Returns (min_pct, max_pct) as fractions.
    """
    asset_lower = asset.lower()
    if "tao" in asset_lower:
        # TAO is more volatile
        return 0.03, 0.40  # 3%–40%
    if asset_lower in ("btc", "eth", "etc"):
        return 0.01, 0.25  # 1%–25%
    return 0.015, 0.30  # default 1.5%–30%


def _bar_minutes_from_series(prices: pd.Series) -> Optional[int]:
    """
    Infer bar length (in minutes) from the index of the kline series.
    Assumes uniform spacing.
    """
    try:
        if len(prices.index) < 2:
            return None
        deltas = prices.index.to_series().diff().dropna()
        # Take median delta to be robust
        median_delta = deltas.median()
        bar_minutes = int(round(median_delta.total_seconds() / 60.0))
        return bar_minutes if bar_minutes > 0 else None
    except Exception:
        return None


def calculate_prediction_interval(
    point_estimate: float,
    historical_prices: Optional[pd.Series],
    asset: str = "btc",
    bar_minutes: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Calculate a 1-hour prediction interval around the point estimate using
    Binance kline data.

    Approach:
    - Work with returns between consecutive klines.
    - Estimate per-bar volatility (sigma_bar).
    - Scale to 1h volatility: sigma_1h = sigma_bar * sqrt(60 / bar_minutes).
    - Use ~95% normal-equivalent band, then clamp using asset-aware caps.

    Args:
        point_estimate (float): GRU-predicted or fallback 1h-ahead price.
        historical_prices (pd.Series | None): Recent Binance close prices.
        asset (str): Asset name.
        bar_minutes (int | None): Bar length in minutes (e.g. 5 for 5m).
            If None, we will try to infer from the index; if that fails,
            default to 5.

    Returns:
        (float, float): (lower_bound, upper_bound)
    """
    try:
        if historical_prices is None or historical_prices.empty:
            bt.logging.warning(f"{asset}: No Binance historical data for interval; using fallback ±15%.")
            margin = point_estimate * 0.15
            return point_estimate - margin, point_estimate + margin

        # Infer bar_minutes if not provided
        if bar_minutes is None:
            bar_minutes = _bar_minutes_from_series(historical_prices) or 5

        # Require a minimum of ~20 bars to have a stable volatility estimate
        if len(historical_prices) < 20:
            bt.logging.warning(
                f"{asset}: Too few Binance points ({len(historical_prices)}) " "for interval; using fallback ±12%."
            )
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 1) Returns (percentage changes)
        returns = historical_prices.pct_change().dropna()
        if returns.empty:
            bt.logging.warning(f"{asset}: Returns series empty; using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 2) Remove extreme outliers (beyond 4 std devs)
        r_std = returns.std()
        if r_std <= 0 or np.isnan(r_std):
            bt.logging.warning(f"{asset}: Non-positive returns std ({r_std}); using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        r_mean = returns.mean()
        outlier_mask = (returns - r_mean).abs() <= 4 * r_std
        clean_returns = returns[outlier_mask]

        if len(clean_returns) < 10:
            bt.logging.warning(f"{asset}: Too few clean points ({len(clean_returns)}); using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 3) Per-bar volatility and scaling to 1-hour
        sigma_bar = float(clean_returns.std())
        if sigma_bar <= 0 or np.isnan(sigma_bar):
            bt.logging.warning(f"{asset}: Non-positive per-bar volatility ({sigma_bar}); using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # How many bars in one hour?
        bars_per_hour = max(1, int(round(60.0 / float(bar_minutes))))
        sigma_1h = sigma_bar * np.sqrt(bars_per_hour)

        if sigma_1h <= 0 or np.isnan(sigma_1h):
            bt.logging.warning(f"{asset}: Non-positive sigma_1h ({sigma_1h}); using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 4) Convert to a confidence band.
        #    Use ~95% interval: z ≈ 1.96 for the 1h return.
        z = 1.96
        raw_margin_pct = sigma_1h * z  # in *return* space

        min_pct, max_pct = _asset_caps(asset)
        margin_pct = max(min_pct, min(raw_margin_pct, max_pct))
        margin = point_estimate * margin_pct

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        bt.logging.debug(
            f"{asset}: bar_minutes={bar_minutes}, bars_per_hour={bars_per_hour}, "
            f"sigma_bar={sigma_bar:.5f}, sigma_1h={sigma_1h:.5f}, "
            f"raw_margin_pct={raw_margin_pct:.4f}, clamped_margin_pct={margin_pct:.4f}, "
            f"margin=${margin:.2f}"
        )

        return lower_bound, upper_bound

    except Exception as e:
        bt.logging.error(f"Error calculating interval for {asset}: {e}")
        # Emergency fallback: ±15%
        margin = point_estimate * 0.15
        return point_estimate - margin, point_estimate + margin


def _guard_point_estimate_with_caps(
    point_estimate: float,
    anchor_price: Optional[float],
    asset: str,
) -> float:
    """
    Guardrail for the GRU point estimate using asset-specific max hourly
    move caps. This improves robustness / accuracy by preventing the model
    from outputting insane 1h jumps that are not supported by recent market
    behaviour.

    Args:
        point_estimate: Raw GRU 1h prediction.
        anchor_price: Prefer Binance spot; fall back to GRU spot.
        asset: Asset name.

    Returns:
        Possibly-clamped point estimate.
    """
    if anchor_price is None or anchor_price <= 0:
        return point_estimate

    min_pct, max_pct = _asset_caps(asset)
    max_move = max_pct  # we use the upper cap for both up/down

    raw_ret = (point_estimate - anchor_price) / anchor_price
    abs_ret = abs(raw_ret)

    if abs_ret <= max_move:
        return point_estimate

    clamped_ret = np.sign(raw_ret) * max_move
    adjusted = anchor_price * (1.0 + clamped_ret)

    bt.logging.info(
        f"{asset}: Point estimate guard applied. "
        f"raw_ret={raw_ret:.3f} (>{max_move:.3f}) -> "
        f"clamped_ret={clamped_ret:.3f}. "
        f"anchor=${anchor_price:.2f}, raw_pred=${point_estimate:.2f}, "
        f"adjusted_pred=${adjusted:.2f}"
    )
    return float(adjusted)


async def forward_async(synapse: Challenge, cm: CMData) -> Challenge:
    """
    Async forward: now relies solely on Binance spot/klines for
    current price & volatility; no CM data is used anymore.
    """
    _ = cm  # CMData is unused; kept only for signature compatibility

    total_start_time = time.perf_counter()

    # Get list of assets to predict and ensure lowercase
    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in raw_assets]

    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} "
        f"for {assets} at timestamp: {synapse.timestamp}"
    )

    predictions = {}
    intervals = {}

    for asset in assets:
        asset_lower = asset.lower()
        symbol = _asset_to_binance_symbol(asset_lower)
        historical_prices: Optional[pd.Series] = None
        spot_price: Optional[float] = None

        # --- Fetch recent Binance kline data (5m) ---
        try:
            historical_prices = await _fetch_binance_klines(
                symbol=symbol,
                interval="5m",
                limit=200,  # ~16.6h of history
            )
            if historical_prices is None or historical_prices.empty:
                bt.logging.warning(f"{asset}: Binance returned empty kline data for {symbol}.")
            else:
                spot_price = float(historical_prices.iloc[-1])
                bt.logging.debug(f"{asset}: Latest Binance spot (from 5m klines) for {symbol} = ${spot_price:.2f}")
        except Exception as e:
            bt.logging.error(f"{asset}: Error fetching Binance klines for {symbol} ({e}).")

        # --- GRU 1h prediction ---
        point_estimate: Optional[float] = None
        anchor_price: Optional[float] = None

        try:
            # GRU expects asset symbol; assume upper-case like "BTC"
            asset_symbol = asset.upper()
            if asset_symbol == "TAO_BITTENSOR":
                asset_symbol = "TAO"

            current_price_gru, predicted_price_1h = predict_1h_price(asset_symbol)
            point_estimate = float(predicted_price_1h)

            # Prefer Binance spot as anchor; otherwise GRU's current price
            anchor_price = float(spot_price) if spot_price is not None else float(current_price_gru)

            extra_spot = f", Binance spot=${spot_price:.2f}" if spot_price is not None else ""
            bt.logging.info(
                f"{asset}: Raw GRU 1h prediction=${point_estimate:.2f} "
                f"(GRU spot=${float(current_price_gru):.2f}{extra_spot})"
            )

            # Apply volatility-based guard to the point estimate
            guarded_estimate = _guard_point_estimate_with_caps(point_estimate, anchor_price, asset)
            point_estimate = guarded_estimate

        except Exception as e:
            bt.logging.error(f"{asset}: GRU inference failed ({e}).")
            if spot_price is not None:
                point_estimate = spot_price
                anchor_price = spot_price
                bt.logging.info(f"{asset}: Falling back to Binance spot as point estimate " f"${spot_price:.2f}")
            else:
                bt.logging.warning(f"{asset}: No GRU prediction and no Binance spot; " "skipping asset.")
                continue

        # --- Interval around the chosen point estimate (GRU or Binance fallback) ---
        bar_minutes = _bar_minutes_from_series(historical_prices) if historical_prices is not None else 5
        interval = calculate_prediction_interval(
            point_estimate,
            historical_prices,
            asset,
            bar_minutes=bar_minutes,
        )

        predictions[asset] = point_estimate
        intervals[asset] = list(interval)

        bt.logging.info(
            f"{asset}: Final prediction=${point_estimate:.2f} | "
            f"Interval=[${interval[0]:.2f}, ${interval[1]:.2f}] "
            f"(bar_minutes={bar_minutes})"
        )

    synapse.predictions = predictions
    synapse.intervals = intervals

    total_time = time.perf_counter() - total_start_time
    bt.logging.debug(f"⏱️ Total forward call took: {total_time:.3f} seconds")

    if synapse.predictions:
        bt.logging.success(f"Predictions complete for {list(predictions.keys())}")
    else:
        bt.logging.info("No predictions for this request.")

    return synapse


async def forward(synapse: Challenge, cm: CMData) -> Challenge:
    """Async forward function for handling predictions."""
    return await forward_async(synapse, cm)
