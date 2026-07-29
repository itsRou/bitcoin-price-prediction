"""Tests for SHAP-based feature importance on tree models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.models.importance import compute_shap_importance
from sklearn.ensemble import RandomForestRegressor


def test_compute_shap_importance_ranks_the_informative_feature_first() -> None:
    rng = np.random.default_rng(3)
    n = 200
    X = pd.DataFrame({"informative": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = 5.0 * X["informative"] + 0.01 * rng.normal(size=n)

    model = RandomForestRegressor(random_state=42, n_estimators=50).fit(X, y)
    importance = compute_shap_importance(model, X.sample(50, random_state=0))

    assert list(importance.index)[0] == "informative"
    assert (importance >= 0).all()
