"""Statistical, financial, and forecast-significance metrics for the model benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

TRADING_PERIODS_PER_YEAR = 365


# --- Statistical / point-forecast metrics -----------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    return float(r2_score(y_true, y_pred))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions that got the sign of the return right."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 across classification classes."""
    return float(f1_score(y_true, y_pred, average="macro"))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Matthews correlation coefficient."""
    return float(matthews_corrcoef(y_true, y_pred))


# --- Financial / backtest metrics --------------------------------------------------------


def total_return(returns: pd.Series) -> float:
    """Cumulative compounded return over the full period."""
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR) -> float:
    """Compound annual growth rate."""
    n_periods = len(returns)
    if n_periods == 0:
        return float("nan")
    growth = (1.0 + returns).prod()
    return float(growth ** (periods_per_year / n_periods) - 1.0)


_STD_EPSILON = 1e-12


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR, risk_free: float = 0.0
) -> float:
    """Annualized Sharpe ratio of a per-period return series."""
    excess = returns - risk_free / periods_per_year
    std = excess.std()
    if np.isnan(std) or std < _STD_EPSILON:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR, risk_free: float = 0.0
) -> float:
    """Annualized Sortino ratio (penalizes only downside deviation)."""
    excess = returns - risk_free / periods_per_year
    downside_std = excess[excess < 0].std()
    if pd.isna(downside_std) or downside_std < _STD_EPSILON:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of the compounded equity curve (negative number)."""
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = TRADING_PERIODS_PER_YEAR) -> float:
    """CAGR divided by the absolute max drawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with a strictly positive return."""
    return float((returns > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    """Sum of gains divided by sum of losses (>1 means gains outweigh losses)."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("nan")
    return float(gains / losses)


def turnover(positions: pd.Series) -> float:
    """Average absolute change in position size per period (proxy for trading activity)."""
    return float(positions.diff().abs().mean())


# --- Forecast significance ----------------------------------------------------------------


def diebold_mariano_test(
    errors_a: np.ndarray, errors_b: np.ndarray, h: int = 1, power: int = 2
) -> tuple[float, float]:
    """Diebold-Mariano test of equal forecast accuracy between two models.

    Args:
        errors_a: Forecast errors (y_true - y_pred) of model A.
        errors_b: Forecast errors of model B.
        h: Forecast horizon in bars, used for the Newey-West autocovariance truncation.
        power: Loss function exponent (2 = squared error, 1 = absolute error).

    Returns:
        (dm_statistic, two_sided_p_value). A significantly negative statistic means
        model A is more accurate than model B.
    """
    loss_diff = np.abs(errors_a) ** power - np.abs(errors_b) ** power
    n = len(loss_diff)
    mean_diff = loss_diff.mean()

    variance = np.var(loss_diff, ddof=0)
    for lag in range(1, h):
        autocov = np.cov(loss_diff[:-lag], loss_diff[lag:])[0, 1]
        variance += 2 * (1 - lag / h) * autocov

    if variance <= 0:
        return float("nan"), float("nan")

    dm_stat = mean_diff / np.sqrt(variance / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)
