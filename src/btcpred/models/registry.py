"""Model registry: name -> zero-arg factory, for both the regression and classification tracks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from btcpred.models.baselines import (
    BuyAndHoldClassifier,
    BuyAndHoldRegressor,
    LagOneLinearRegressor,
    MeanReturnRegressor,
    NaiveFlatClassifier,
    NaiveZeroRegressor,
)
from btcpred.models.boosting import (
    CatBoostClassifierWrapper,
    CatBoostRegressorWrapper,
    LightGBMClassifier,
    LightGBMRegressor,
    XGBoostClassifier,
    XGBoostRegressor,
)
from btcpred.models.classical import get_classification_models, get_regression_models
from btcpred.models.statistical import (
    AutoArimaRegressor,
    HoltWintersRegressor,
    ProphetRegressor,
    VARRegressor,
)


def get_regression_registry() -> dict[str, Callable[[], Any]]:
    """All registered regression models: Tier 0 baselines, Tier 1 statistical, Tier 2/boosting."""
    registry: dict[str, Callable[[], Any]] = {
        "naive_zero": NaiveZeroRegressor,
        "mean_return": MeanReturnRegressor,
        "buy_and_hold": BuyAndHoldRegressor,
        "lag_one_linear": LagOneLinearRegressor,
        "auto_arima": AutoArimaRegressor,
        "holt_winters": HoltWintersRegressor,
        "prophet": ProphetRegressor,
        "var": VARRegressor,
        "xgboost": XGBoostRegressor,
        "lightgbm": LightGBMRegressor,
        "catboost": CatBoostRegressorWrapper,
    }
    registry.update(get_regression_models())
    return registry


def get_classification_registry() -> dict[str, Callable[[], Any]]:
    """All registered classification models: Tier 0 baselines, Tier 2/boosting."""
    registry: dict[str, Callable[[], Any]] = {
        "naive_flat": NaiveFlatClassifier,
        "buy_and_hold": BuyAndHoldClassifier,
        "xgboost": XGBoostClassifier,
        "lightgbm": LightGBMClassifier,
        "catboost": CatBoostClassifierWrapper,
    }
    registry.update(get_classification_models())
    return registry


def get_model(name: str, task: str = "regression") -> Any:
    """Build a fresh, unfitted model instance by name for the given task.

    Args:
        name: Registered model name, e.g. "xgboost", "random_forest", "auto_arima".
        task: "regression" or "classification".

    Returns:
        A new, unfitted estimator instance exposing `.fit(X, y)` / `.predict(X)`.
    """
    registry = get_regression_registry() if task == "regression" else get_classification_registry()
    if name not in registry:
        raise KeyError(f"Unknown model '{name}' for task '{task}'. Available: {sorted(registry)}")
    return registry[name]()
