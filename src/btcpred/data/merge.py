"""Align OHLCV, macro, sentiment, and on-chain sources onto one daily UTC index."""

from __future__ import annotations

import pandas as pd


def _is_daily(df: pd.DataFrame) -> bool:
    """Heuristic: treat data as already-daily if its median timestamp spacing is >= 20 hours."""
    if len(df) < 2:
        return True
    median_gap = df.index.to_series().diff().median()
    return bool(median_gap >= pd.Timedelta(hours=20))


def align_daily(
    sources: dict[str, pd.DataFrame], reference_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Forward-fill each source onto a shared daily index, flagging filled rows per source.

    Non-daily sources are first downsampled to daily (last value of the day). Each source's
    columns are prefixed with its name, and a companion `{name}_was_filled` boolean column
    records where the value did not originate on that exact reference date (e.g. weekends).

    Args:
        sources: Mapping of source name -> DataFrame, UTC-indexed.
        reference_index: The daily UTC index every source is aligned onto.

    Returns:
        A single DataFrame with one column-group per source plus its fill flag.
    """
    merged = pd.DataFrame(index=reference_index)
    for name, df in sources.items():
        daily = df if _is_daily(df) else df.resample("1D").last()
        reindexed = daily.reindex(reference_index)
        was_filled = reindexed.isna().any(axis=1)
        reindexed = reindexed.ffill()
        reindexed.columns = [f"{name}_{col}" for col in reindexed.columns]
        merged = merged.join(reindexed)
        merged[f"{name}_was_filled"] = was_filled.ffill().fillna(True)
    return merged
