"""Tests for SHAP-based feature importance on tree models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from btcpred.models.importance import compute_shap_importance, explain_single_prediction


def test_compute_shap_importance_ranks_the_informative_feature_first() -> None:
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame({"informative": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = 5.0 * X["informative"] + 0.01 * rng.normal(size=n)

    model = RandomForestRegressor(random_state=42, n_estimators=50).fit(X, y)
    importance = compute_shap_importance(model, X.sample(50, random_state=0))

    assert list(importance.index)[0] == "informative"
    assert (importance >= 0).all()


def test_explain_single_prediction_flags_the_informative_feature() -> None:
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame({"informative": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = 5.0 * X["informative"] + 0.01 * rng.normal(size=n)

    model = RandomForestRegressor(random_state=42, n_estimators=50).fit(X, y)
    row = X.iloc[[0]]

    explanation = explain_single_prediction(model, row, top_n=2)

    assert len(explanation) == 2
    assert explanation[0][0] == "informative"
    assert isinstance(explanation[0][1], float)
