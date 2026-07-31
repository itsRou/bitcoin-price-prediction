"""Tests for Tier-1 statistical time-series models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcpred.models.statistical import (
    AutoArimaRegressor,
    GARCHVolatilityForecaster,
    HoltWintersRegressor,
    ProphetRegressor,
    VARRegressor,
)


def _make_series(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(0, 0.01, n), index=idx)
    X = pd.DataFrame({"exog1": rng.normal(0, 1, n)}, index=idx)
    return X, y


def test_auto_arima_forecasts_test_length() -> None:
    X, y = _make_series()
    model = AutoArimaRegressor().fit(X.iloc[:100], y.iloc[:100])
    preds = model.predict(X.iloc[100:])

    assert len(preds) == len(X.iloc[100:])
    assert np.all(np.isfinite(preds))


def test_holt_winters_forecasts_test_length() -> None:
    X, y = _make_series()
    model = HoltWintersRegressor().fit(X.iloc[:100], y.iloc[:100])
    preds = model.predict(X.iloc[100:])

    assert len(preds) == len(X.iloc[100:])
    assert np.all(np.isfinite(preds))


def test_prophet_forecasts_test_length() -> None:
    X, y = _make_series()
    model = ProphetRegressor().fit(X.iloc[:100], y.iloc[:100])
    preds = model.predict(X.iloc[100:])

    assert len(preds) == len(X.iloc[100:])
    assert np.all(np.isfinite(preds))


def test_var_forecasts_test_length() -> None:
    X, y = _make_series()
    model = VARRegressor(exog_columns=("exog1",)).fit(X.iloc[:100], y.iloc[:100])
    preds = model.predict(X.iloc[100:])

    assert len(preds) == len(X.iloc[100:])
    assert np.all(np.isfinite(preds))


def test_garch_forecasts_positive_volatility() -> None:
    X, y = _make_series()
    model = GARCHVolatilityForecaster().fit(X.iloc[:100], y.iloc[:100])
    preds = model.predict(X.iloc[100:])

    assert len(preds) == len(X.iloc[100:])
    assert np.all(preds > 0)
