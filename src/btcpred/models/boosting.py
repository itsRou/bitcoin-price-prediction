"""Gradient boosting models (XGBoost, LightGBM, CatBoost) with chronological early stopping.

Each wrapper carves the *last* `val_fraction` of the (already chronologically ordered)
training fold off as an early-stopping validation set -- never a random split, since that
would let the model implicitly peek at "future" rows relative to the rest of training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_N_ESTIMATORS = 1000


def _chronological_split(
    X: pd.DataFrame, y: pd.Series, val_fraction: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split a chronologically ordered (X, y) into (train, val) without shuffling."""
    split_at = max(int(len(X) * (1 - val_fraction)), 1)
    return X.iloc[:split_at], y.iloc[:split_at], X.iloc[split_at:], y.iloc[split_at:]


class XGBoostRegressor(BaseEstimator, RegressorMixin):
    """XGBoost regressor with early stopping on a chronological validation tail."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        **xgb_params: object,
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostRegressor:
        X_train, y_train, X_val, y_val = _chronological_split(X, y, self.val_fraction)
        self.model_ = XGBRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.xgb_params,
        )
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier with early stopping on a chronological validation tail."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        **xgb_params: object,
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostClassifier:
        X_train, y_train, X_val, y_val = _chronological_split(X, y, self.val_fraction)
        self.classes_ = np.unique(y)
        self.model_ = XGBClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.xgb_params,
        )
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


class LightGBMRegressor(BaseEstimator, RegressorMixin):
    """LightGBM regressor with early stopping on a chronological validation tail."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        **lgbm_params: Any,
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.lgbm_params = lgbm_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMRegressor:
        import lightgbm as lgb

        X_train, y_train, X_val, y_val = _chronological_split(X, y, self.val_fraction)
        self.model_ = LGBMRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS, random_state=RANDOM_STATE, **self.lgbm_params
        )
        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X))


class LightGBMClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM classifier with early stopping on a chronological validation tail."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        **lgbm_params: Any,
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.lgbm_params = lgbm_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMClassifier:
        import lightgbm as lgb

        X_train, y_train, X_val, y_val = _chronological_split(X, y, self.val_fraction)
        self.classes_ = np.unique(y)
        self.model_ = LGBMClassifier(
            n_estimators=DEFAULT_N_ESTIMATORS, random_state=RANDOM_STATE, **self.lgbm_params
        )
        self.model_.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model_.predict(X))


class CatBoostRegressorWrapper(BaseEstimator, RegressorMixin):
    """CatBoost regressor with early stopping on a chronological validation tail."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        **catboost_params: object,
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.catboost_params = catboost_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CatBoostRegressorWrapper:
        X_train, y_train, X_val, y_val = _chronological_split(X, y, self.val_fraction)
        self.model_ = CatBoostRegressor(
            iterations=DEFAULT_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
            **self.catboost_params,
        )
        self.model_.fit(X_train, y_train, eval_set=(X_val, y_val))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)


class CatBoostClassifierWrapper(BaseEstimator, ClassifierMixin):
    """CatBoost classifier with early stopping on a chronological validation tail."""

    def __init__(
        self,
        val_fraction: float = DEFAULT_VAL_FRACTION,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
        **catboost_params: object,
    ) -> None:
        self.val_fraction = val_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.catboost_params = catboost_params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CatBoostClassifierWrapper:
        X_train, y_train, X_val, y_val = _chronological_split(X, y, self.val_fraction)
        self.classes_ = np.unique(y)
        self.model_ = CatBoostClassifier(
            iterations=DEFAULT_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=False,
            **self.catboost_params,
        )
        self.model_.fit(X_train, y_train, eval_set=(X_val, y_val))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)
