"""Optuna hyperparameter tuning restricted to walk-forward folds, logged to MLflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import mlflow
import numpy as np
import optuna
import pandas as pd


class _Splitter(Protocol):
    def split(self, X: pd.DataFrame, y: pd.Series | None = None, groups: object = None) -> Any: ...


def tune_model(
    model_factory: Callable[[dict[str, Any]], Any],
    param_space_fn: Callable[[optuna.Trial], dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    splitter: _Splitter,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_trials: int = 50,
    direction: str = "minimize",
    experiment_name: str = "btcpred",
    seed: int = 42,
) -> optuna.Study:
    """Tune hyperparameters over walk-forward folds only, logging every trial to MLflow.

    Args:
        model_factory: Builds a fresh, unfitted model instance from a sampled params dict.
        param_space_fn: Given an Optuna trial, returns the params dict to sample and score.
        X: Full feature matrix; only fold-defined subsets are ever used to fit or score.
        y: Target series aligned to `X`.
        splitter: A walk-forward splitter exposing `.split(X)` (e.g. `PurgedWalkForwardSplit`).
        metric_fn: Scores (y_true, y_pred) -> float; direction says which way is "better".
        n_trials: Number of Optuna trials to run.
        direction: "minimize" or "maximize".
        experiment_name: MLflow experiment to log trials under.
        seed: Sampler seed, for reproducibility.

    Returns:
        The completed Optuna study, with `study.best_params` and `study.best_value` set.
    """
    mlflow.set_experiment(experiment_name)

    def objective(trial: optuna.Trial) -> float:
        params = param_space_fn(trial)
        fold_scores = []
        with mlflow.start_run(nested=mlflow.active_run() is not None):
            mlflow.log_params(params)
            for train_idx, test_idx in splitter.split(X):
                model = model_factory(params)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[test_idx])
                fold_scores.append(metric_fn(y.iloc[test_idx].to_numpy(), preds))
            mean_score = float(np.mean(fold_scores))
            mlflow.log_metric("mean_fold_score", mean_score)
        return mean_score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
