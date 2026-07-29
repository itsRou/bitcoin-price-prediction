"""Matplotlib plotting helpers shared by the Streamlit app and offline reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_equity_curve(backtest_df: pd.DataFrame) -> Figure:
    """Plot the compounded equity curve from a `run_backtest` output."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(backtest_df.index, backtest_df["equity"], color="#2563eb")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (starting at 1.0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_predictions_vs_actual(y_true: pd.Series, y_pred: pd.Series) -> Figure:
    """Plot predicted vs. actual target values over time."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y_true.index, y_true.to_numpy(), label="actual", color="#334155", alpha=0.8)
    ax.plot(y_true.index, y_pred, label="predicted", color="#dc2626", alpha=0.8)
    ax.set_title("Predicted vs. Actual")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, top_n: int = 20) -> Figure:
    """Horizontal bar chart of the top-n most important features."""
    top = importance.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(top))))
    ax.barh(top.index, top.to_numpy(), color="#2563eb")
    ax.set_title(f"Top {len(top)} Features by Importance")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, labels: list[int] | None = None
) -> Figure:
    """Confusion matrix for the 3-class down/flat/up classification track."""
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_regime_performance(returns: pd.Series, regime: pd.Series) -> Figure:
    """Bar chart of mean return per regime label."""
    mean_by_regime = returns.groupby(regime).mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(mean_by_regime.index.astype(str), mean_by_regime.to_numpy(), color="#2563eb")
    ax.set_title("Mean Return by Regime")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig
