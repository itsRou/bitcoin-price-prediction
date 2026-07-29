"""SHAP-based feature importance for tree-based models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def compute_shap_importance(model: object, X_sample: pd.DataFrame) -> pd.Series:
    """Compute mean |SHAP value| per feature for a fitted tree-based model.

    Args:
        model: A fitted tree-based estimator, or a Pipeline whose last step is one.
        X_sample: Feature rows to explain (a representative subset, for speed).

    Returns:
        Mean absolute SHAP value per feature, sorted descending (most important first).
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        # Multi-class classifiers return one array per class; average their magnitudes.
        abs_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        abs_values = np.abs(shap_values)

    importance = pd.Series(abs_values.mean(axis=0), index=X_sample.columns)
    return importance.sort_values(ascending=False)


def explain_single_prediction(
    model: object, row: pd.DataFrame, top_n: int = 3
) -> list[tuple[str, float]]:
    """Local SHAP attribution for one specific prediction, not an aggregate importance.

    Args:
        model: A fitted tree-based estimator, or a Pipeline whose last step is one.
        row: A single-row DataFrame (the exact features used for that one prediction).
        top_n: Number of top-contributing features to return.

    Returns:
        Up to `top_n` (feature_name, signed_shap_value) pairs, sorted by |contribution|
        descending. A positive value pushed the prediction up; negative pushed it down.
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(row)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    contributions = pd.Series(shap_values[0], index=row.columns)
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    return list(ranked.head(top_n).items())
