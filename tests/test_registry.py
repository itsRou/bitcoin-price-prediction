"""Tests for the unified model registry."""

from __future__ import annotations

import pytest

from btcpred.models.registry import get_classification_registry, get_model, get_regression_registry


def test_regression_registry_contains_expected_models() -> None:
    registry = get_regression_registry()
    for name in ("naive_zero", "xgboost", "lightgbm", "catboost", "random_forest", "auto_arima"):
        assert name in registry


def test_classification_registry_contains_expected_models() -> None:
    registry = get_classification_registry()
    for name in ("naive_flat", "xgboost", "logistic_regression", "gaussian_nb"):
        assert name in registry


def test_get_model_returns_fresh_unfitted_instance() -> None:
    model_a = get_model("random_forest", task="regression")
    model_b = get_model("random_forest", task="regression")

    assert model_a is not model_b


def test_get_model_raises_for_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_model("not_a_real_model", task="regression")
