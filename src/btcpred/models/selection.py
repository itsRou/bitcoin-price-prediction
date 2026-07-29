"""Transparent, rule-based model selection: pick whichever candidate has recently been best.

This is the same idea as `RegimeConditionalEnsemble` but keyed on realized error history
rather than a regime label -- useful for a "which model would we have used, and why"
explanation, since the rule is auditable: pick the lowest mean trailing error, using only
errors known *before* the date being explained.
"""

from __future__ import annotations

import pandas as pd


def select_best_model_by_rolling_error(
    errors_by_model: pd.DataFrame, window: int = 20
) -> pd.Series:
    """For each row, pick the model with the lowest mean absolute error over the trailing window.

    Args:
        errors_by_model: DataFrame indexed by date, one column per candidate model, values
            are that model's absolute prediction error on that date (NaN where a model has
            no prediction for that date).
        window: Number of preceding predictions to average over.

    Returns:
        A Series (same index) of the selected model name per date. The rolling window is
        shifted by one row so a date's own error never influences its own selection.
    """
    rolling_mean_error = errors_by_model.rolling(window=window, min_periods=1).mean().shift(1)
    return rolling_mean_error.idxmin(axis=1)
