"""Tests for the rolling-error-based model selection rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.models.selection import select_best_model_by_rolling_error


def test_selects_the_model_with_lower_trailing_error() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    errors = pd.DataFrame(
        {
            "good_model": [0.01] * 10,
            "bad_model": [0.5] * 10,
        },
        index=idx,
    )

    selection = select_best_model_by_rolling_error(errors, window=3)

    # First row has no history (NaN window before shift); everything after should pick good_model.
    assert (selection.iloc[1:] == "good_model").all()


def test_selection_does_not_use_the_same_day_error() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    errors = pd.DataFrame(
        {
            "model_a": [1.0, 1.0, 1.0, 0.0, 1.0],
            "model_b": [0.0, 0.0, 0.0, 1.0, 0.0],
        },
        index=idx,
    )

    selection = select_best_model_by_rolling_error(errors, window=3)

    # On day index 3, model_a suddenly has zero error, but selection must be based on
    # days 0-2 (model_b was better there), not day 3's own value.
    assert selection.iloc[3] == "model_b"


def test_handles_missing_predictions_as_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    errors = pd.DataFrame(
        {
            "model_a": [0.1, np.nan, 0.1, 0.1],
            "model_b": [0.2, 0.2, np.nan, 0.2],
        },
        index=idx,
    )

    selection = select_best_model_by_rolling_error(errors, window=2)

    # The first row has no prior history to select from (NaN after the shift); every
    # subsequent row should have picked one of the two candidates.
    assert selection.iloc[1:].isin(["model_a", "model_b"]).all()
