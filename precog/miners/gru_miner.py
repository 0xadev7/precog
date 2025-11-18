import time
from typing import Tuple

import bittensor as bt
import pandas as pd
import numpy as np

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str

# 🔮 GRU model inference (you need price_model on PYTHONPATH)
from price_model.inference import predict_1h_price


def calculate_prediction_interval(
    point_estimate: float, historical_prices: pd.Series, asset: str = "btc"
) -> Tuple[float, float]:
    """
    Calculate a 1-hour prediction interval around the GRU point estimate.

    We use realized 1h volatility from high-frequency CM ReferenceRate data
    (typically 1-second prices), then translate that into a percentage band.

    Args:
        point_estimate (float): GRU-predicted 1h-ahead price (or CM fallback).
        historical_prices (pd.Series): High-frequency price series for the
            recent window (e.g. last hour from CM).
        asset (str): Asset name for logging / asset-specific caps.

    Returns:
        (float, float): (lower_bound, upper_bound)
    """
    try:
        if historical_prices is None or historical_prices.empty:
            bt.logging.warning(f"{asset}: No historical data for interval; using fallback ±15%.")
            margin = point_estimate * 0.15
            return point_estimate - margin, point_estimate + margin

        # Require at least a few minutes of data (e.g. ~5 minutes at 1s frequency).
        if len(historical_prices) < 300:
            bt.logging.warning(f"{asset}: Too few points ({len(historical_prices)}) for interval; using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 1) High-frequency returns (percentage changes)
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

        if len(clean_returns) < 100:
            bt.logging.warning(f"{asset}: Too few clean points ({len(clean_returns)}); using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 3) Realized 1-hour volatility:
        #    sqrt(sum(r_t^2)) over the last hour ≈ std of 1h return.
        realized_var = float((clean_returns**2).sum())
        sigma_1h = float(np.sqrt(realized_var))

        if sigma_1h <= 0 or np.isnan(sigma_1h):
            bt.logging.warning(f"{asset}: Non-positive realized volatility ({sigma_1h}); using fallback ±12%.")
            margin = point_estimate * 0.12
            return point_estimate - margin, point_estimate + margin

        # 4) Convert to a confidence band.
        #    Use ~95% interval: z ≈ 1.96. This is for the 1h return.
        z = 1.96
        raw_margin_pct = sigma_1h * z  # in *return* space

        # Asset-aware min / max caps to avoid overly tight/wide bands.
        asset_lower = asset.lower()
        if "tao" in asset_lower:
            # TAO is usually more volatile
            min_pct, max_pct = 0.03, 0.40  # 3%–40%
        elif asset_lower in ("btc", "eth", "etc"):
            min_pct, max_pct = 0.01, 0.25  # 1%–25%
        else:
            min_pct, max_pct = 0.015, 0.30  # default 1.5%–30%

        margin_pct = max(min_pct, min(raw_margin_pct, max_pct))
        margin = point_estimate * margin_pct

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        bt.logging.debug(
            f"{asset}: sigma_1h={sigma_1h:.4f}, raw_margin_pct={raw_margin_pct:.4f}, "
            f"clamped_margin_pct={margin_pct:.4f}, margin=${margin:.2f}"
        )

        return lower_bound, upper_bound

    except Exception as e:
        bt.logging.error(f"Error calculating interval for {asset}: {e}")
        # Emergency fallback: ±15%
        margin = point_estimate * 0.15
        return point_estimate - margin, point_estimate + margin


async def forward_async(synapse: Challenge, cm: CMData) -> Challenge:
    total_start_time = time.perf_counter()

    # Get list of assets to predict and ensure lowercase
    raw_assets = synapse.assets if hasattr(synapse, "assets") else ["btc"]
    assets = [asset.lower() for asset in raw_assets]

    bt.logging.info(
        f"👈 Received prediction request from: {synapse.dendrite.hotkey} "
        f"for {assets} at timestamp: {synapse.timestamp}"
    )

    # Timestamps for CM data fetch (used for volatility + fallback)
    provided_timestamp = to_datetime(synapse.timestamp)
    start_timestamp = get_before(synapse.timestamp, hours=1, minutes=0, seconds=0)  # 1h window for volatility

    # Fetch ALL data in a single CM call (still used for interval + logging)
    all_data = cm.get_CM_ReferenceRate(
        assets=assets,
        start=to_str(start_timestamp),
        end=to_str(provided_timestamp),
        frequency="1s",
    )

    predictions = {}
    intervals = {}

    if not all_data.empty:
        for asset in assets:
            asset_data = all_data[all_data["asset"] == asset]

            if asset_data.empty:
                bt.logging.warning(f"No CM data for {asset} in response")
                continue

            # Latest CM reference price (spot)
            spot_price = float(asset_data["ReferenceRateUSD"].iloc[-1])
            historical_prices = asset_data["ReferenceRateUSD"]

            # --- GRU 1h prediction ---
            try:
                # GRU expects asset symbol; assume upper-case like "BTC"
                asset_symbol = asset.upper()
                if asset_symbol == "TAO_BITTENSOR":
                    asset_symbol = "TAO"

                current_price_gru, predicted_price_1h = predict_1h_price(asset_symbol)

                bt.logging.info(
                    f"{asset}: GRU 1h prediction=${predicted_price_1h:.2f} "
                    f"(GRU spot=${current_price_gru:.2f}, CM spot=${spot_price:.2f})"
                )

                point_estimate = float(predicted_price_1h)

            except Exception as e:
                # Fallback to CM spot price if GRU fails
                bt.logging.error(f"{asset}: GRU inference failed ({e}), falling back to CM spot.")
                point_estimate = spot_price

            # Interval around the chosen point estimate (GRU or fallback)
            interval = calculate_prediction_interval(point_estimate, historical_prices, asset)

            predictions[asset] = point_estimate
            intervals[asset] = list(interval)

            bt.logging.info(
                f"{asset}: " f"Prediction=${point_estimate:.2f} | " f"Interval=[${interval[0]:.2f}, ${interval[1]:.2f}]"
            )
    else:
        bt.logging.warning("CM returned empty data frame; no volatility info available.")
        # In this rare case we can still try GRU-only predictions as a last resort
        for asset in assets:
            try:
                _, predicted_price_1h = predict_1h_price(asset.upper())
                point_estimate = float(predicted_price_1h)
                # No CM history -> simple ±15% interval
                margin = point_estimate * 0.15
                lower, upper = point_estimate - margin, point_estimate + margin
                predictions[asset] = point_estimate
                intervals[asset] = [lower, upper]

                bt.logging.info(f"{asset}: GRU-only prediction=${point_estimate:.2f} " f"(no CM data, interval ±15%)")
            except Exception as e:
                bt.logging.error(f"{asset}: GRU inference failed and CM data empty ({e}); " f"no prediction produced.")

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
