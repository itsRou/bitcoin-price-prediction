"""Tests for statistical, financial, and Diebold-Mariano significance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcpred.validation.metrics import (
    cagr,
    calmar_ratio,
    diebold_mariano_test,
    directional_accuracy,
    f1_macro,
    mae,
    mape,
    max_drawdown,
    mcc,
    profit_factor,
    r_squared,
    rmse,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover,
    win_rate,
)


def test_rmse_mae_mape_r2_on_known_values() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 5.0])

    assert rmse(y_true, y_pred) == pytest.approx(0.5)
    assert mae(y_true, y_pred) == pytest.approx(0.25)
    assert mape(y_true, y_pred) > 0
    assert r_squared(y_true, y_true) == pytest.approx(1.0)


def test_directional_accuracy_counts_correct_signs() -> None:
    y_true = np.array([1.0, -1.0, 2.0, -3.0])
    y_pred = np.array([0.5, -0.1, -0.2, -1.0])

    assert directional_accuracy(y_true, y_pred) == 0.75


def test_f1_macro_and_mcc_perfect_prediction() -> None:
    y_true = np.array([-1, 0, 1, -1, 0, 1])
    y_pred = np.array([-1, 0, 1, -1, 0, 1])

    assert f1_macro(y_true, y_pred) == 1.0
    assert mcc(y_true, y_pred) == 1.0


def test_total_return_and_cagr_on_flat_series() -> None:
    returns = pd.Series([0.0] * 365)

    assert total_return(returns) == 0.0
    assert cagr(returns) == pytest.approx(0.0)


def test_sharpe_and_sortino_are_nan_when_no_variance() -> None:
    returns = pd.Series([0.01] * 30)

    assert np.isnan(sharpe_ratio(returns))
    assert np.isnan(sortino_ratio(returns))


def test_sharpe_ratio_positive_for_upward_drifting_noisy_returns() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 500))

    assert sharpe_ratio(returns) > 0


def test_max_drawdown_and_calmar_on_known_path() -> None:
    returns = pd.Series([0.1, -0.5, 0.1])

    mdd = max_drawdown(returns)
    assert mdd < 0
    assert not np.isnan(calmar_ratio(returns))


def test_win_rate_and_profit_factor() -> None:
    returns = pd.Series([0.01, -0.01, 0.02, -0.005])

    assert win_rate(returns) == 0.5
    assert profit_factor(returns) == pytest.approx((0.01 + 0.02) / (0.01 + 0.005))


def test_turnover_measures_average_position_change() -> None:
    positions = pd.Series([0, 1, 1, 0, -1])

    assert turnover(positions) == pytest.approx(0.75)


def test_diebold_mariano_favors_the_more_accurate_model() -> None:
    rng = np.random.default_rng(42)
    errors_good = rng.normal(0, 0.1, 200)
    errors_bad = rng.normal(0, 1.0, 200)

    dm_stat, p_value = diebold_mariano_test(errors_good, errors_bad, h=1)

    assert dm_stat < 0
    assert p_value < 0.05
