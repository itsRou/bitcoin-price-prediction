"""Tests for exogenous (macro/on-chain/sentiment) features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.features.exog import compute_exog_features


def _make_exog_df() -> pd.DataFrame:
    n = 150
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    close = 20_000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    sp500_close = 4_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    hash_rate_value = np.linspace(100.0, 200.0, n)
    sentiment_value = rng.uniform(0, 100, n)
    return pd.DataFrame(
        {
            "close": close,
            "sp500_close": sp500_close,
            "hash_rate_value": hash_rate_value,
            "sentiment_value": sentiment_value,
        },
        index=idx,
    )


def test_missing_close_column_raises() -> None:
    df = pd.DataFrame({"sp500_close": [1.0, 2.0]})
    try:
        compute_exog_features(df)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_corr_column_is_bounded() -> None:
    result = compute_exog_features(_make_exog_df())
    corr = result["corr_btc_sp500_close"].dropna()

    assert (corr >= -1.0001).all()
    assert (corr <= 1.0001).all()


def test_onchain_and_sentiment_columns_present() -> None:
    result = compute_exog_features(_make_exog_df())

    assert "hash_rate_zscore" in result.columns
    assert "fear_greed_delta" in result.columns
    assert "fear_greed_zscore" in result.columns
