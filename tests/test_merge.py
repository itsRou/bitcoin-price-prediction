"""Tests for the multi-source daily alignment merge."""

from __future__ import annotations

import pandas as pd

from btcpred.data.merge import align_daily


def test_align_daily_forward_fills_and_flags_weekends() -> None:
    reference_index = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    macro = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"], utc=True),
    )

    merged = align_daily({"dxy": macro}, reference_index)

    assert "dxy_close" in merged.columns
    assert "dxy_was_filled" in merged.columns
    assert merged.loc["2024-01-02", "dxy_close"] == 100.0
    assert bool(merged.loc["2024-01-02", "dxy_was_filled"]) is True
    assert bool(merged.loc["2024-01-01", "dxy_was_filled"]) is False
