"""Tests for the gradient boosting wrappers with chronological early stopping."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcpred.models.boosting import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    LightGBMClassifier,
    LightGBMRegressor,
    XGBoostClassifier,
    XGBoostRegressor,
)


def _make_regression_xy(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(0.4 * X["f1"] - 0.2 * X["f2"] + rng.normal(scale=0.05, size=n))
    return X, y


def _make_classification_xy(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((X["f1"] > 0).astype(int).to_numpy())
    return X, y


def test_xgboost_regressor_fits_with_early_stopping() -> None:
    X, y = _make_regression_xy()
    model = XGBoostRegressor(early_stopping_rounds=5).fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))


def test_xgboost_classifier_fits_with_early_stopping() -> None:
    X, y = _make_classification_xy()
    model = XGBoostClassifier(early_stopping_rounds=5).fit(X, y)
    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})


def test_lightgbm_regressor_fits_with_early_stopping() -> None:
    X, y = _make_regression_xy()
    model = LightGBMRegressor(early_stopping_rounds=5).fit(X, y)
    preds = model.predict(X)

    assert np.all(np.isfinite(preds))


def test_lightgbm_classifier_fits_with_early_stopping() -> None:
    X, y = _make_classification_xy()
    model = LightGBMClassifier(early_stopping_rounds=5).fit(X, y)
    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})


def test_catboost_regressor_fits_with_early_stopping() -> None:
    X, y = _make_regression_xy()
    model = CatBoostRegressorWrapper(early_stopping_rounds=5).fit(X, y)
    preds = model.predict(X)

    assert np.all(np.isfinite(preds))


def test_catboost_classifier_fits_with_early_stopping() -> None:
    X, y = _make_classification_xy()
    model = CatBoostClassifierWrapper(early_stopping_rounds=5).fit(X, y)
    preds = model.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})
