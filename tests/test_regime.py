"""Tests for regime and market-structure features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.features.regime import compute_regime_features


def test_drawdown_from_ath_is_never_positive(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_regime_features(synthetic_ohlcv)

    assert (result["drawdown_from_ath"] <= 1e-9).all()


def test_bull_regime_flag_is_binary(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_regime_features(synthetic_ohlcv)

    assert set(result["bull_regime_200ma"].dropna().unique()).issubset({0.0, 1.0})


def test_days_since_halving_matches_known_date() -> None:
    idx = pd.to_datetime(["2020-05-11", "2020-05-21", "2024-04-25"], utc=True)
    df = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)

    result = compute_regime_features(df)

    assert result["days_since_halving"].iloc[0] == 0
    assert result["days_since_halving"].iloc[1] == 10
    assert result["days_since_halving"].iloc[2] == 5


def test_days_since_halving_is_nan_before_first_halving() -> None:
    idx = pd.to_datetime(["2010-01-01"], utc=True)
    df = pd.DataFrame({"close": [100.0]}, index=idx)

    result = compute_regime_features(df)

    assert np.isnan(result["days_since_halving"].iloc[0])


def test_volatility_regime_labels_are_in_expected_range(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_regime_features(synthetic_ohlcv)
    labels = result["volatility_regime"].dropna().unique()

    assert set(labels).issubset({0.0, 1.0, 2.0})
