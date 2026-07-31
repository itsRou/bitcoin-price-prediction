"""Tests for ensemble strategies: weighted top-k, OOF stacking, regime-conditional."""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcpred.models.baselines import MeanReturnRegressor, NaiveZeroRegressor
from btcpred.models.classical import get_regression_models
from btcpred.models.ensemble import (
    RegimeConditionalEnsemble,
    StackingEnsemble,
    WeightedAverageEnsemble,
)
from btcpred.validation.splitters import PurgedWalkForwardSplit


def _make_xy(n: int = 150) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(5)
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(0.5 * X["f1"] + rng.normal(scale=0.05, size=n))
    return X, y


def test_weighted_average_ensemble_drops_the_weakest_models() -> None:
    X, y = _make_xy()
    classical = get_regression_models()
    factories = {
        "naive_zero": NaiveZeroRegressor,
        "mean_return": MeanReturnRegressor,
        "linear": classical["linear"],
        "ridge": classical["ridge"],
    }

    model = WeightedAverageEnsemble(factories, top_k=2).fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert len(model.selected_names_) == 2
    assert "naive_zero" not in model.selected_names_


def test_stacking_ensemble_trains_on_oof_predictions_and_predicts() -> None:
    X, y = _make_xy(200)
    classical = get_regression_models()
    factories = {
        "linear": classical["linear"],
        "ridge": classical["ridge"],
        "naive_zero": NaiveZeroRegressor,
    }
    splitter = PurgedWalkForwardSplit(n_splits=4, purge=1, embargo=1)

    model = StackingEnsemble(factories, splitter=splitter).fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds))
    assert set(model.base_model_names_) == set(factories)


def test_regime_conditional_ensemble_learns_opposite_relationships_per_regime() -> None:
    rng = np.random.default_rng(6)
    n = 200
    regime = rng.integers(0, 2, n)
    f1 = rng.normal(size=n)
    y = np.where(regime == 0, f1, -f1) + rng.normal(scale=0.01, size=n)
    X = pd.DataFrame({"f1": f1, "volatility_regime": regime})
    y = pd.Series(y)

    linear_factory = get_regression_models()["linear"]
    model = RegimeConditionalEnsemble(linear_factory, regime_column="volatility_regime").fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert np.corrcoef(preds, y)[0, 1] > 0.9
