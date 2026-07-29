"""Tests for target construction and full feature-matrix assembly."""

from __future__ import annotations

import numpy as np
import pandas as pd
from btcpred.features.build import build_feature_matrix, compute_targets


def test_compute_targets_regression_matches_manual_log_return() -> None:
    close = pd.Series(
        [100.0, 102.0, 101.0, 105.0, 103.0], index=pd.date_range("2024-01-01", periods=5, freq="D")
    )

    targets = compute_targets(close, horizons=(1,), dead_zone_std_multiplier=0.25)

    expected_h1 = np.log(close.shift(-1)) - np.log(close)
    pd.testing.assert_series_equal(targets["y_reg_h1"], expected_h1, check_names=False)


def test_compute_targets_classification_has_dead_zone() -> None:
    close = pd.Series(
        [100.0] * 40 + [100.001], index=pd.date_range("2024-01-01", periods=41, freq="D")
    )

    targets = compute_targets(close, horizons=(1,), dead_zone_std_multiplier=0.25)

    assert set(targets["y_clf_h1"].dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_build_feature_matrix_has_expected_target_columns(synthetic_ohlcv: pd.DataFrame) -> None:
    matrix = build_feature_matrix(synthetic_ohlcv, horizons=(1, 7))

    for h in (1, 7):
        assert f"y_reg_h{h}" in matrix.columns
        assert f"y_clf_h{h}" in matrix.columns
    assert matrix.index.equals(synthetic_ohlcv.index)
