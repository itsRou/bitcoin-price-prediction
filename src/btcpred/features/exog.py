"""Exogenous features: rolling BTC correlation with macro assets, on-chain and sentiment z-scores.

Expects a single daily-aligned DataFrame (the output of `btcpred.data.merge.align_daily`,
joined with BTC's own daily close) containing the specific column names produced by that
merge step: "{asset}_close" for macro assets, "{metric}_value" for on-chain metrics, and
"sentiment_value" for the Fear & Greed Index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_ASSET_COLUMNS = ("sp500_close", "dxy_close", "gold_close")
ONCHAIN_METRICS = ("hash_rate", "active_addresses", "transaction_count", "miner_revenue")
SENTIMENT_COLUMN = "sentiment_value"
CORR_WINDOW = 30
ZSCORE_WINDOW = 90


def compute_exog_features(
    df: pd.DataFrame, corr_window: int = CORR_WINDOW, zscore_window: int = ZSCORE_WINDOW
) -> pd.DataFrame:
    """Compute rolling macro correlations, on-chain z-scores, and sentiment deltas.

    Args:
        df: Daily-aligned DataFrame with a "close" (BTC) column plus any of
            `MACRO_ASSET_COLUMNS`, `{metric}_value` on-chain columns, or `SENTIMENT_COLUMN`.
        corr_window: Rolling window (days) for the BTC-vs-macro return correlation.
        zscore_window: Rolling window (days) for on-chain/sentiment z-scores.

    Returns:
        A DataFrame of exogenous feature columns aligned to `df.index`.
    """
    if "close" not in df.columns:
        raise ValueError("compute_exog_features requires a 'close' column for BTC price")

    out = pd.DataFrame(index=df.index)
    btc_returns = np.log(df["close"]).diff()

    for col in MACRO_ASSET_COLUMNS:
        if col in df.columns:
            asset_returns = np.log(df[col]).diff()
            out[f"corr_btc_{col}"] = btc_returns.rolling(corr_window).corr(asset_returns)

    for metric in ONCHAIN_METRICS:
        col = f"{metric}_value"
        if col in df.columns:
            rolling = df[col].rolling(zscore_window)
            out[f"{metric}_zscore"] = (df[col] - rolling.mean()) / rolling.std()

    if SENTIMENT_COLUMN in df.columns:
        rolling_fg = df[SENTIMENT_COLUMN].rolling(zscore_window)
        out["fear_greed_delta"] = df[SENTIMENT_COLUMN].diff()
        out["fear_greed_zscore"] = (df[SENTIMENT_COLUMN] - rolling_fg.mean()) / rolling_fg.std()

    return out
