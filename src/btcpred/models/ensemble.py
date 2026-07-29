"""Ensemble strategies: weighted top-k averaging, OOF stacking, regime-conditional selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from btcpred.validation.metrics import rmse
from btcpred.validation.splitters import PurgedWalkForwardSplit


class WeightedAverageEnsemble(BaseEstimator, RegressorMixin):
    """Averages the top-k candidate models by chronological validation error.

    Candidates are scored on a held-out chronological tail, the best `top_k` are kept,
    each refit on the *full* training data, and predictions are averaged with weights
    inversely proportional to validation error (better models count for more).
    """

    def __init__(
        self,
        model_factories: dict[str, Callable[[], Any]],
        top_k: int = 3,
        val_fraction: float = 0.2,
        metric_fn: Callable[[np.ndarray, np.ndarray], float] = rmse,
    ) -> None:
        self.model_factories = model_factories
        self.top_k = top_k
        self.val_fraction = val_fraction
        self.metric_fn = metric_fn

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WeightedAverageEnsemble:
        split_at = max(int(len(X) * (1 - self.val_fraction)), 1)
        X_train, y_train = X.iloc[:split_at], y.iloc[:split_at]
        X_val, y_val = X.iloc[split_at:], y.iloc[split_at:]

        val_scores: dict[str, float] = {}
        for name, factory in self.model_factories.items():
            model = factory().fit(X_train, y_train)
            preds = model.predict(X_val)
            val_scores[name] = self.metric_fn(y_val.to_numpy(), preds)

        best_names = sorted(val_scores, key=lambda name: val_scores[name])[: self.top_k]
        errors = np.array([val_scores[name] for name in best_names])
        inverse_error = 1.0 / np.maximum(errors, 1e-8)
        self.weights_ = inverse_error / inverse_error.sum()

        self.models_ = {name: self.model_factories[name]().fit(X, y) for name in best_names}
        self.selected_names_ = best_names
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.array([self.models_[name].predict(X) for name in self.selected_names_])
        return np.average(predictions, axis=0, weights=self.weights_)


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Ridge-meta-learner stacking, trained only on out-of-fold base-model predictions.

    Training the meta-learner on in-fold predictions would let it implicitly learn each
    base model's tendency to overfit its own training data, inflating apparent accuracy.
    Walking forward and only ever scoring each base model on data it did *not* train on
    avoids that.
    """

    def __init__(
        self,
        model_factories: dict[str, Callable[[], Any]],
        splitter: PurgedWalkForwardSplit | None = None,
        meta_learner: Any = None,
    ) -> None:
        self.model_factories = model_factories
        self.splitter = splitter
        self.meta_learner = meta_learner

    def fit(self, X: pd.DataFrame, y: pd.Series) -> StackingEnsemble:
        splitter = self.splitter or PurgedWalkForwardSplit(n_splits=5, purge=1, embargo=1)
        names = list(self.model_factories)

        oof_predictions: list[np.ndarray] = []
        oof_targets: list[np.ndarray] = []
        for train_idx, test_idx in splitter.split(X):
            fold_preds = np.column_stack(
                [
                    self.model_factories[name]()
                    .fit(X.iloc[train_idx], y.iloc[train_idx])
                    .predict(X.iloc[test_idx])
                    for name in names
                ]
            )
            oof_predictions.append(fold_preds)
            oof_targets.append(y.iloc[test_idx].to_numpy())

        stacked_X = np.vstack(oof_predictions)
        stacked_y = np.concatenate(oof_targets)

        self.meta_learner_ = self.meta_learner or Ridge()
        self.meta_learner_.fit(stacked_X, stacked_y)

        self.base_models_ = {name: self.model_factories[name]().fit(X, y) for name in names}
        self.base_model_names_ = names
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base_predictions = np.column_stack(
            [self.base_models_[name].predict(X) for name in self.base_model_names_]
        )
        return np.asarray(self.meta_learner_.predict(base_predictions))


class RegimeConditionalEnsemble(BaseEstimator, RegressorMixin):
    """Trains a separate model per regime value, falling back to a global model otherwise."""

    def __init__(
        self, model_factory: Callable[[], Any], regime_column: str = "volatility_regime"
    ) -> None:
        self.model_factory = model_factory
        self.regime_column = regime_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RegimeConditionalEnsemble:
        self.global_model_ = self.model_factory().fit(X, y)

        regimes = X[self.regime_column]
        self.regime_models_ = {}
        for regime_value in regimes.dropna().unique():
            mask = regimes == regime_value
            if mask.sum() < 2:
                continue
            self.regime_models_[regime_value] = self.model_factory().fit(X[mask], y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.empty(len(X))
        regimes = X[self.regime_column].to_numpy()
        global_preds = self.global_model_.predict(X)

        for i, regime_value in enumerate(regimes):
            model = self.regime_models_.get(regime_value)
            predictions[i] = model.predict(X.iloc[[i]])[0] if model is not None else global_preds[i]
        return predictions
