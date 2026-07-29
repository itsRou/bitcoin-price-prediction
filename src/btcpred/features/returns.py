"""Return, rolling-moment, and volatility-estimator features from OHLC data."""

from __future__ import annotations

import numpy as np
import pandas as pd

RETURN_LAGS = tuple(range(1, 15))
ROLLING_WINDOWS = (7, 14, 30, 90)
ROC_WINDOWS = (7, 14)
ZSCORE_WINDOW = 30
VOL_WINDOW = 14
_LOG_2 = np.log(2)


def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-return, rolling-moment, and volatility features.

    Args:
        df: DataFrame with open/high/low/close columns, UTC-indexed.

    Returns:
        A DataFrame of return-derived feature columns aligned to `df.index`.
    """
    out = pd.DataFrame(index=df.index)
    log_close = np.log(df["close"])
    log_return_1 = log_close.diff()

    for lag in RETURN_LAGS:
        out[f"log_return_lag_{lag}"] = log_close.diff(lag)

    for window in ROLLING_WINDOWS:
        rolling = log_return_1.rolling(window)
        out[f"return_mean_{window}"] = rolling.mean()
        out[f"return_std_{window}"] = rolling.std()
        out[f"return_skew_{window}"] = rolling.skew()
        out[f"return_kurt_{window}"] = rolling.kurt()

    out[f"realized_vol_{VOL_WINDOW}"] = log_return_1.rolling(VOL_WINDOW).std() * np.sqrt(365)
    out[f"parkinson_vol_{VOL_WINDOW}"] = _parkinson_vol(df, VOL_WINDOW)
    out[f"garman_klass_vol_{VOL_WINDOW}"] = _garman_klass_vol(df, VOL_WINDOW)

    for window in ROC_WINDOWS:
        out[f"roc_{window}"] = df["close"].pct_change(window)

    rolling_price = df["close"].rolling(ZSCORE_WINDOW)
    out[f"price_zscore_{ZSCORE_WINDOW}"] = (
        df["close"] - rolling_price.mean()
    ) / rolling_price.std()

    return out


def _parkinson_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Parkinson volatility estimator using the high-low range."""
    log_hl_sq = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(log_hl_sq.rolling(window).mean() / (4 * _LOG_2))


def _garman_klass_vol(df: pd.DataFrame, window: int) -> pd.Series:
    """Garman-Klass volatility estimator using open/high/low/close."""
    log_hl_sq = np.log(df["high"] / df["low"]) ** 2
    log_co_sq = np.log(df["close"] / df["open"]) ** 2
    daily_var = 0.5 * log_hl_sq - (2 * _LOG_2 - 1) * log_co_sq
    return np.sqrt(daily_var.rolling(window).mean())
