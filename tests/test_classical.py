"""Tests for the Tier-2 classical ML model factories."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.models.classical import get_classification_models, get_regression_models


def _make_regression_xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(0.5 * X["f1"] - 0.3 * X["f2"] + rng.normal(scale=0.1, size=n))
    return X, y


def _make_classification_xy(n: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(rng.integers(-1, 2, size=n))
    return X, y


def test_all_regression_models_fit_and_predict() -> None:
    X, y = _make_regression_xy()
    for name, factory in get_regression_models().items():
        model = factory().fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X), name
        assert np.all(np.isfinite(preds)), name


def test_all_classification_models_fit_and_predict() -> None:
    X, y = _make_classification_xy()
    for name, factory in get_classification_models().items():
        model = factory().fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X), name
