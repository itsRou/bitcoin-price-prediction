"""Tests for the Optuna + MLflow walk-forward tuning wrapper."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd

from btcpred.models.baselines import BuyAndHoldRegressor
from btcpred.validation.metrics import rmse
from btcpred.validation.splitters import PurgedWalkForwardSplit
from btcpred.validation.tuning import tune_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_tune_model_finds_the_best_assumed_return(tmp_path: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")

    n = 60
    X = pd.DataFrame({"log_return_lag_1": np.zeros(n)})
    y = pd.Series(np.full(n, 0.003))

    def model_factory(params: dict) -> BuyAndHoldRegressor:
        return BuyAndHoldRegressor(assumed_return=params["assumed_return"])

    def param_space_fn(trial: optuna.Trial) -> dict:
        return {"assumed_return": trial.suggest_float("assumed_return", -0.01, 0.01)}

    splitter = PurgedWalkForwardSplit(n_splits=3, purge=1, embargo=1)

    study = tune_model(
        model_factory,
        param_space_fn,
        X,
        y,
        splitter,
        metric_fn=rmse,
        n_trials=8,
        direction="minimize",
        experiment_name="test-tuning",
        seed=0,
    )

    assert study.best_value < rmse(np.full(n, 0.003), np.zeros(n))
    assert abs(study.best_params["assumed_return"] - 0.003) < 0.01
