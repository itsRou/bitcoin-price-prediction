"""Technical indicators computed from OHLCV data via pandas-ta.

All indicators here are computed on data up to and including the current bar (the
standard definition of a technical indicator). The leakage guard for these features is
applied once, globally, when `features.build.build_feature_matrix` shifts the entire
feature block by one bar before joining it to the targets.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # noqa: F401  (registers the .ta DataFrame accessor)

EMA_SMA_WINDOWS = (9, 21, 50, 200)


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the technical-indicator feature block.

    Args:
        df: DataFrame with open/high/low/close/volume columns, UTC-indexed.

    Returns:
        A DataFrame of technical indicator columns aligned to `df.index`.
    """
    out = pd.DataFrame(index=df.index)

    out["rsi_14"] = df.ta.rsi(length=14)

    macd = df.ta.macd(fast=12, slow=26, signal=9)
    out["macd"], out["macd_hist"], out["macd_signal"] = (
        macd.iloc[:, 0],
        macd.iloc[:, 1],
        macd.iloc[:, 2],
    )

    bbands = df.ta.bbands(length=20, std=2)
    out["bb_bandwidth"] = bbands.iloc[:, 3]
    out["bb_percent_b"] = bbands.iloc[:, 4]

    out["atr_14"] = df.ta.atr(length=14)
    out["adx_14"] = df.ta.adx(length=14).iloc[:, 0]
    out["cci_20"] = df.ta.cci(length=20)
    out["willr_14"] = df.ta.willr(length=14)

    stoch = df.ta.stoch()
    out["stoch_k"], out["stoch_d"] = stoch.iloc[:, 0], stoch.iloc[:, 1]

    out["obv"] = df.ta.obv()

    vwap = df.ta.vwap()
    out["vwap_distance"] = (df["close"] - vwap) / vwap

    for window in EMA_SMA_WINDOWS:
        out[f"ema_{window}"] = df.ta.ema(length=window)
        out[f"sma_{window}"] = df.ta.sma(length=window)

    out["ema_cross_9_21"] = (out["ema_9"] > out["ema_21"]).astype(int)
    out["ema_cross_21_50"] = (out["ema_21"] > out["ema_50"]).astype(int)
    out["sma_cross_50_200"] = (out["sma_50"] > out["sma_200"]).astype(int)

    return out
