"""Tests for return, rolling-moment, and volatility-estimator features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.features.returns import compute_return_features


def test_log_return_lag_1_matches_manual_calculation(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_return_features(synthetic_ohlcv)
    expected = np.log(synthetic_ohlcv["close"]).diff()

    pd.testing.assert_series_equal(result["log_return_lag_1"], expected, check_names=False)


def test_rolling_stats_columns_present_for_all_windows(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_return_features(synthetic_ohlcv)

    for window in (7, 14, 30, 90):
        for stat in ("mean", "std", "skew", "kurt"):
            assert f"return_{stat}_{window}" in result.columns


def test_volatility_estimators_are_non_negative(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_return_features(synthetic_ohlcv)

    for col in ("realized_vol_14", "parkinson_vol_14", "garman_klass_vol_14"):
        assert (result[col].dropna() >= 0).all()


def test_price_zscore_is_centered_near_zero_on_random_walk(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_return_features(synthetic_ohlcv)
    zscore = result["price_zscore_30"].dropna()

    assert abs(zscore.mean()) < 3.0
