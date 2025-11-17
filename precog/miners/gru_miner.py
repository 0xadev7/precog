import time
from typing import Tuple

import bittensor as bt
import pandas as pd

from precog.protocol import Challenge
from precog.utils.cm_data import CMData
from precog.utils.timestamp import get_before, to_datetime, to_str

# 🔮 GRU model inference (you need price_model on PYTHONPATH)
from price_model.inference import predict_1h_price


def calculate_prediction_interval(
    point_estimate: float, historical_prices: pd.Series, asset: str = "btc"
) -> Tuple[float, float]:
    """Calculate prediction interval using historical volatility from provided data.

    Args:
        point_estimate (float): The center of the prediction interval (predicted price).
        historical_prices (pd.Series): Historical price data for volatility calculation.
        asset (str): The asset name for logging.

    Returns:
        (float, float): (lower_bound, upper_bound)
    """
    try:
        if historical_prices.empty or len(historical_prices) < 100:
            bt.logging.warning(f"Insufficient data for {asset}, using fallback interval")
            # Fallback: ±10% of point estimate
            margin = point_estimate * 0.10
            return point_estimate - margin, point_estimate + margin

        # Calculate returns (percentage changes)
        hourly_returns = historical_prices.pct_change().dropna()

        # Remove extreme outliers (beyond 3 std devs) to get realistic volatility
        returns_std = hourly_returns.std()
        returns_mean = hourly_returns.mean()
        outlier_mask = abs(hourly_returns - returns_mean) <= 3 * returns_std
        clean_returns = hourly_returns[outlier_mask]

        if len(clean_returns) < 12:
            bt.logging.warning(f"Too few clean data points for {asset}, using fallback")
            margin = point_estimate * 0.10  # Increased from 5%
            return point_estimate - margin, point_estimate + margin

        # Use standard deviation of returns for 1-hour prediction
        hourly_vol = float(clean_returns.std())

        # 2.58 standard deviations ≈ 99% confidence interval
        margin = point_estimate * hourly_vol * 2.58

        # Cap interval width
        max_margin = point_estimate * 0.30  # ±30%
        min_margin = point_estimate * 0.02  # ±2%
        margin = max(min_margin, min(margin, max_margin))

        lower_bound = point_estimate - margin
        upper_bound = point_estimate + margin

        bt.logging.debug(f"{asset}: hourly_vol={hourly_vol:.4f}, margin=${margin:.2f}")

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
