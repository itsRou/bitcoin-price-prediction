"""Cleaning utilities: gap-filling, outlier handling, and UTC normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd

_FREQ_BY_TIMEFRAME = {"1h": "1h", "1d": "1D"}


def normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the index is tz-aware UTC, localizing naive indexes as UTC."""
    if df.index.tz is None:
        return df.tz_localize("UTC")
    return df.tz_convert("UTC")


def fill_missing_candles(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Reindex OHLCV data onto a regular grid, forward-filling gaps and flagging them.

    Args:
        df: OHLCV DataFrame with columns open/high/low/close/volume, UTC-indexed.
        timeframe: One of the keys in `_FREQ_BY_TIMEFRAME` ("1h", "1d").

    Returns:
        Reindexed DataFrame with an added boolean "was_filled" column.
    """
    freq = _FREQ_BY_TIMEFRAME[timeframe]
    full_index = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    reindexed = df.reindex(full_index)
    was_filled = reindexed["close"].isna()
    price_cols = ["open", "high", "low", "close"]
    reindexed[price_cols] = reindexed[price_cols].ffill()
    reindexed["volume"] = reindexed["volume"].fillna(0.0)
    reindexed["was_filled"] = was_filled
    return reindexed


def clip_outliers(df: pd.DataFrame, column: str, z_thresh: float = 8.0) -> pd.DataFrame:
    """Replace extreme single-step log-return outliers (likely bad ticks) via forward-fill.

    Args:
        df: DataFrame containing `column`.
        column: Name of the price column to check for spikes.
        z_thresh: Absolute z-score of the log return above which a point is an outlier.

    Returns:
        A copy of `df` with outlier values in `column` forward-filled.
    """
    log_returns = np.log(df[column] / df[column].shift(1))
    z_scores = (log_returns - log_returns.mean()) / log_returns.std()
    is_outlier = z_scores.abs() > z_thresh
    cleaned = df.copy()
    cleaned.loc[is_outlier, column] = np.nan
    cleaned[column] = cleaned[column].ffill()
    return cleaned
