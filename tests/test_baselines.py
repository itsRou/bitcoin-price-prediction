"""Tests for the Tier-0 baseline predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.models.baselines import (
    BuyAndHoldClassifier,
    BuyAndHoldRegressor,
    LagOneLinearRegressor,
    MeanReturnRegressor,
    NaiveFlatClassifier,
    NaiveZeroRegressor,
)


def _make_xy() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"log_return_lag_1": [0.01, -0.02, 0.03, -0.01, 0.02]})
    y = pd.Series([0.02, -0.015, 0.025, -0.02, 0.01])
    return X, y


def test_naive_zero_regressor_always_predicts_zero() -> None:
    X, y = _make_xy()
    model = NaiveZeroRegressor().fit(X, y)

    assert np.array_equal(model.predict(X), np.zeros(len(X)))


def test_mean_return_regressor_predicts_training_mean() -> None:
    X, y = _make_xy()
    model = MeanReturnRegressor().fit(X, y)

    assert np.allclose(model.predict(X), y.mean())


def test_buy_and_hold_regressor_predicts_fixed_positive_return() -> None:
    X, y = _make_xy()
    model = BuyAndHoldRegressor(assumed_return=0.002).fit(X, y)

    assert np.allclose(model.predict(X), 0.002)


def test_lag_one_linear_regressor_fits_a_line() -> None:
    X = pd.DataFrame({"log_return_lag_1": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([2.0, 4.0, 6.0, 8.0])
    model = LagOneLinearRegressor().fit(X, y)

    preds = model.predict(X)
    assert np.allclose(preds, y.to_numpy(), atol=1e-6)


def test_naive_flat_classifier_always_predicts_flat() -> None:
    X, y = _make_xy()
    model = NaiveFlatClassifier().fit(X, y)

    assert np.array_equal(model.predict(X), np.zeros(len(X)))


def test_buy_and_hold_classifier_always_predicts_up() -> None:
    X, y = _make_xy()
    model = BuyAndHoldClassifier().fit(X, y)

    assert np.array_equal(model.predict(X), np.ones(len(X)))
