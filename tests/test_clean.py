"""Tests for OHLCV cleaning utilities."""

from __future__ import annotations

import pandas as pd
from btcpred.data.clean import clip_outliers, fill_missing_candles, normalize_timezone


def test_normalize_timezone_localizes_naive_index() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
    result = normalize_timezone(df)
    assert str(result.index.tz) == "UTC"


def test_normalize_timezone_converts_existing_tz() -> None:
    df = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern"),
    )
    result = normalize_timezone(df)
    assert str(result.index.tz) == "UTC"


def test_fill_missing_candles_flags_gaps() -> None:
    idx = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"], utc=True)
    df = pd.DataFrame(
        {
            "open": [1.0, 3.0],
            "high": [1.1, 3.1],
            "low": [0.9, 2.9],
            "close": [1.0, 3.0],
            "volume": [10.0, 30.0],
        },
        index=idx,
    )
    result = fill_missing_candles(df, "1h")

    assert len(result) == 3
    assert result["was_filled"].sum() == 1
    assert result.iloc[1]["close"] == 1.0


def test_clip_outliers_removes_extreme_spike() -> None:
    values = [100.0] * 20 + [10000.0] + [100.0] * 5
    df = pd.DataFrame(
        {"close": values}, index=pd.date_range("2024-01-01", periods=len(values), freq="D")
    )
    result = clip_outliers(df, "close", z_thresh=3.0)

    assert result["close"].max() < 200.0
