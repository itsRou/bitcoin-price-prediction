"""Regime features: drawdown, trend regime, halving cycle, Hurst exponent, volatility regime."""

from __future__ import annotations

import numpy as np
import pandas as pd

BULL_BEAR_MA_WINDOW = 200
HURST_WINDOW = 100
HURST_LAGS = range(2, 20)
VOL_WINDOW = 30
N_VOLATILITY_REGIMES = 3
VOLATILITY_REGIME_MIN_PERIODS = 30

# BTC block-reward halving dates (UTC), including the next scheduled one.
HALVING_DATES = pd.DatetimeIndex(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"], tz="UTC")


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown, trend-regime, halving-cycle, Hurst, and volatility-cluster features.

    Args:
        df: DataFrame with a "close" column, UTC-indexed.

    Returns:
        A DataFrame of regime feature columns aligned to `df.index`.
    """
    out = pd.DataFrame(index=df.index)

    running_max = df["close"].cummax()
    out["drawdown_from_ath"] = df["close"] / running_max - 1

    sma = df["close"].rolling(BULL_BEAR_MA_WINDOW).mean()
    out["bull_regime_200ma"] = (df["close"] > sma).astype(float)

    out["days_since_halving"] = _days_since_halving(df.index)

    log_price = np.log(df["close"])
    out["hurst_exponent"] = log_price.rolling(HURST_WINDOW).apply(_hurst_exponent, raw=True)

    volatility = log_price.diff().rolling(VOL_WINDOW).std()
    out["volatility_regime"] = _volatility_regime(volatility)

    return out


def _days_since_halving(index: pd.DatetimeIndex) -> pd.Series:
    """Vectorized days-since-most-recent-halving, NaN before the first known halving."""
    halving_ns = HALVING_DATES.values.astype("datetime64[ns]").astype(np.int64)
    index_ns = index.values.astype("datetime64[ns]").astype(np.int64)
    positions = np.searchsorted(halving_ns, index_ns, side="right") - 1

    days = np.full(len(index), np.nan)
    valid = positions >= 0
    seconds_per_day = 86_400 * 1_000_000_000
    days[valid] = (index_ns[valid] - halving_ns[positions[valid]]) / seconds_per_day
    return pd.Series(days, index=index)


def _hurst_exponent(values: np.ndarray) -> float:
    """Estimate the Hurst exponent via log-log scaling of lagged-difference std devs."""
    tau = np.array([np.std(values[lag:] - values[:-lag]) for lag in HURST_LAGS])
    valid = tau > 0
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(np.array(list(HURST_LAGS))[valid]), np.log(tau[valid]), 1)
    return float(slope)


def _classify_into_expanding_tercile(window: np.ndarray) -> float:
    """Bucket the window's last value against quantile thresholds of that same window."""
    current = window[-1]
    if np.isnan(current):
        return np.nan
    quantile_edges = [i / N_VOLATILITY_REGIMES for i in range(1, N_VOLATILITY_REGIMES)]
    thresholds = np.quantile(window[~np.isnan(window)], quantile_edges)
    return float(np.searchsorted(thresholds, current))


def _volatility_regime(volatility: pd.Series) -> pd.Series:
    """Classify each bar's volatility into regimes using only its own expanding history.

    Unlike a global KMeans fit (which would let regime boundaries depend on the entire
    series, including future volatility), this uses an expanding window so the regime
    label at time t is determined solely by data available up to and including t.
    """
    return volatility.expanding(min_periods=VOLATILITY_REGIME_MIN_PERIODS).apply(
        _classify_into_expanding_tercile, raw=True
    )
