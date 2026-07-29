"""Assemble the full feature matrix and prediction targets.

Leakage guard: every engineered feature (technical, return, regime, exogenous) is computed
from data available up to and including its own bar, then the *entire* feature block is
shifted forward by one bar in `build_feature_matrix`. This guarantees the feature row at
time t reflects only information known as of t-1 -- what would genuinely be available
before acting on a forecast for the return from t to t+h. Targets are computed separately
and are allowed to look forward, since that is their purpose.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from btcpred.features.exog import compute_exog_features
from btcpred.features.regime import compute_regime_features
from btcpred.features.returns import compute_return_features
from btcpred.features.technical import compute_technical_features

DEFAULT_HORIZONS = (1, 7)
DEFAULT_DEAD_ZONE_STD_MULTIPLIER = 0.25
TARGET_VOL_WINDOW = 30


def compute_raw_features(
    ohlcv: pd.DataFrame, exog_daily: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute the unshifted feature block (technical + return + regime [+ exogenous]).

    Exposed separately from `build_feature_matrix` so tests can verify the final shift
    was applied correctly, rather than trusting it implicitly.
    """
    features = compute_return_features(ohlcv)
    features = features.join(compute_technical_features(ohlcv))
    features = features.join(compute_regime_features(ohlcv))

    if exog_daily is not None:
        exog_input = exog_daily.join(ohlcv[["close"]], how="left")
        features = features.join(compute_exog_features(exog_input))
        fill_flag_cols = [c for c in exog_daily.columns if c.endswith("_was_filled")]
        if fill_flag_cols:
            features = features.join(exog_daily[fill_flag_cols])

    return features


def compute_targets(
    close: pd.Series,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    dead_zone_std_multiplier: float = DEFAULT_DEAD_ZONE_STD_MULTIPLIER,
    vol_window: int = TARGET_VOL_WINDOW,
) -> pd.DataFrame:
    """Compute regression and dead-zoned 3-class targets for each horizon.

    Args:
        close: BTC close price series, UTC-indexed.
        horizons: Forecast horizons in bars (e.g. (1, 7)).
        dead_zone_std_multiplier: Multiple of trailing return volatility defining the
            "flat" dead zone around zero.
        vol_window: Trailing window (bars) used to estimate return volatility for the
            dead zone, using only single-step returns known as of time t.

    Returns:
        A DataFrame with `y_reg_h{h}` and `y_clf_h{h}` columns for each horizon.
    """
    out = pd.DataFrame(index=close.index)
    log_close = np.log(close)
    trailing_std = log_close.diff().rolling(vol_window).std()

    for h in horizons:
        y_reg = log_close.shift(-h) - log_close
        dead_zone = dead_zone_std_multiplier * trailing_std * np.sqrt(h)
        y_clf = np.sign(y_reg).where(y_reg.abs() >= dead_zone, 0.0)
        out[f"y_reg_h{h}"] = y_reg
        out[f"y_clf_h{h}"] = y_clf

    return out


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    exog_daily: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    dead_zone_std_multiplier: float = DEFAULT_DEAD_ZONE_STD_MULTIPLIER,
) -> pd.DataFrame:
    """Build the leakage-safe feature matrix joined with regression/classification targets.

    Args:
        ohlcv: BTC OHLCV DataFrame, UTC-indexed.
        exog_daily: Optional daily-aligned exogenous DataFrame (macro/on-chain/sentiment),
            e.g. the output of `btcpred.data.merge.align_daily`.
        horizons: Forecast horizons in bars.
        dead_zone_std_multiplier: Dead-zone width for the classification target.

    Returns:
        A single DataFrame: shifted (leakage-safe) feature columns plus target columns.
    """
    raw_features = compute_raw_features(ohlcv, exog_daily)
    shifted_features = raw_features.shift(1)
    targets = compute_targets(ohlcv["close"], horizons, dead_zone_std_multiplier)
    return shifted_features.join(targets)
