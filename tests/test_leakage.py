"""Leakage tests: prove no feature depends on information from its own or a future bar.

Two independent checks:
1. Structural: the shipped feature matrix must equal the unshifted feature block shifted
   forward by exactly one bar -- i.e. every column genuinely went through the leakage
   guard in `build_feature_matrix`, not just some of them.
2. Behavioral: perturbing (shuffling) the *future* tail of the OHLCV series must not
   change a single feature value in the untouched past. Rolling/expanding windows only
   look backward, so if this ever fails, some feature is reading ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcpred.features.build import build_feature_matrix, compute_raw_features

# sma_200 / bull_regime_200ma is the longest lookback window in the pipeline; keep a
# margin past it so the "safe" comparison region is entirely untouched by the perturbation.
_LOOKBACK_MARGIN = 210


def _make_trending_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0005, 0.02, n)
    close = 20_000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(1_000, 5_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_feature_matrix_equals_raw_features_shifted_by_one_bar(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    raw = compute_raw_features(synthetic_ohlcv)
    matrix = build_feature_matrix(synthetic_ohlcv, horizons=(1, 7))

    feature_cols = [c for c in matrix.columns if not c.startswith(("y_reg_", "y_clf_"))]

    pd.testing.assert_frame_equal(matrix[feature_cols], raw.shift(1)[feature_cols])


def test_perturbing_future_prices_does_not_change_past_features() -> None:
    ohlcv = _make_trending_ohlcv()
    cutoff = 300

    matrix_original = build_feature_matrix(ohlcv, horizons=(1, 7))

    ohlcv_perturbed = ohlcv.copy()
    rng = np.random.default_rng(123)
    future_index = ohlcv_perturbed.index[cutoff:]
    shuffled_index = rng.permutation(future_index)
    ohlcv_perturbed.loc[future_index, :] = ohlcv_perturbed.loc[shuffled_index, :].to_numpy()

    matrix_perturbed = build_feature_matrix(ohlcv_perturbed, horizons=(1, 7))

    feature_cols = [c for c in matrix_original.columns if not c.startswith(("y_reg_", "y_clf_"))]
    safe_end = cutoff - _LOOKBACK_MARGIN

    pd.testing.assert_frame_equal(
        matrix_original[feature_cols].iloc[:safe_end],
        matrix_perturbed[feature_cols].iloc[:safe_end],
    )


def test_target_at_row_t_actually_uses_future_close(synthetic_ohlcv: pd.DataFrame) -> None:
    """Sanity check that targets (unlike features) are intentionally forward-looking."""
    matrix = build_feature_matrix(synthetic_ohlcv, horizons=(1,))
    expected = np.log(synthetic_ohlcv["close"].shift(-1)) - np.log(synthetic_ohlcv["close"])

    pd.testing.assert_series_equal(matrix["y_reg_h1"], expected, check_names=False)
