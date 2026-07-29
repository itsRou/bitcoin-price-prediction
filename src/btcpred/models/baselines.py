"""Tier-0 baselines. No real model is worth reporting until it beats every one of these.

For log-return targets, "naive" means predicting zero change -- the efficient-market /
random-walk assumption that the best guess for tomorrow's return is no return at all.
This is distinct from (and stricter than) "predict tomorrow's price equals today's price."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression

DEFAULT_LAG_COLUMN = "log_return_lag_1"


class NaiveZeroRegressor(BaseEstimator, RegressorMixin):
    """Predicts zero return every time (the random-walk / EMH baseline)."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveZeroRegressor:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class MeanReturnRegressor(BaseEstimator, RegressorMixin):
    """Predicts the training set's mean return for every row."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MeanReturnRegressor:
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class BuyAndHoldRegressor(BaseEstimator, RegressorMixin):
    """Predicts a fixed small positive return every time, i.e. "always be long"."""

    def __init__(self, assumed_return: float = 1e-4) -> None:
        self.assumed_return = assumed_return

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BuyAndHoldRegressor:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.assumed_return)


class LagOneLinearRegressor(BaseEstimator, RegressorMixin):
    """Ordinary least squares regression on a single lagged-return feature."""

    def __init__(self, lag_column: str = DEFAULT_LAG_COLUMN) -> None:
        self.lag_column = lag_column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LagOneLinearRegressor:
        self.model_ = LinearRegression()
        self.model_.fit(X[[self.lag_column]], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X[[self.lag_column]])


class NaiveFlatClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the "flat" (0) class every time."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveFlatClassifier:
        self.classes_ = np.array([-1, 0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class BuyAndHoldClassifier(BaseEstimator, ClassifierMixin):
    """Predicts the "up" (1) class every time."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BuyAndHoldClassifier:
        self.classes_ = np.array([-1, 0, 1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X))
