"""Tests for the matplotlib plotting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from btcpred.viz.plots import (
    plot_confusion_matrix,
    plot_equity_curve,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_regime_performance,
)


def test_plot_equity_curve_returns_figure() -> None:
    df = pd.DataFrame({"equity": [1.0, 1.01, 1.02]}, index=pd.date_range("2024-01-01", periods=3))

    fig = plot_equity_curve(df)

    assert isinstance(fig, Figure)


def test_plot_predictions_vs_actual_returns_figure() -> None:
    idx = pd.date_range("2024-01-01", periods=5)
    y_true = pd.Series([0.1, -0.1, 0.2, -0.2, 0.05], index=idx)
    y_pred = np.array([0.05, -0.05, 0.15, -0.1, 0.02])

    fig = plot_predictions_vs_actual(y_true, y_pred)

    assert isinstance(fig, Figure)


def test_plot_feature_importance_returns_figure() -> None:
    importance = pd.Series([0.5, 0.3, 0.1], index=["a", "b", "c"])

    fig = plot_feature_importance(importance, top_n=2)

    assert isinstance(fig, Figure)


def test_plot_confusion_matrix_returns_figure() -> None:
    y_true = pd.Series([-1, 0, 1, -1, 0, 1])
    y_pred = pd.Series([-1, 0, 1, 0, 0, 1])

    fig = plot_confusion_matrix(y_true, y_pred)

    assert isinstance(fig, Figure)


def test_plot_regime_performance_returns_figure() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    regime = pd.Series([0, 1, 0, 1])

    fig = plot_regime_performance(returns, regime)

    assert isinstance(fig, Figure)
