"""Tests for pandas-ta-based technical indicator features."""

from __future__ import annotations

import pandas as pd

from btcpred.features.technical import compute_technical_features


def test_compute_technical_features_shape_and_index(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_technical_features(synthetic_ohlcv)

    assert result.index.equals(synthetic_ohlcv.index)
    assert len(result.columns) > 15


def test_rsi_stays_within_bounds(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_technical_features(synthetic_ohlcv)
    rsi = result["rsi_14"].dropna()

    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_ma_crosses_are_binary(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_technical_features(synthetic_ohlcv)

    for col in ("ema_cross_9_21", "ema_cross_21_50", "sma_cross_50_200"):
        assert set(result[col].dropna().unique()).issubset({0, 1})


def test_bb_bandwidth_is_non_negative(synthetic_ohlcv: pd.DataFrame) -> None:
    result = compute_technical_features(synthetic_ohlcv)

    assert (result["bb_bandwidth"].dropna() >= 0).all()
