"""Tests for backtest tearsheet reporting."""

from __future__ import annotations

import pandas as pd

from btcpred.backtest.report import compute_tearsheet, format_tearsheet_markdown

EXPECTED_KEYS = (
    "total_return",
    "cagr",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "win_rate",
    "profit_factor",
    "turnover",
)


def test_compute_tearsheet_returns_expected_keys() -> None:
    backtest_df = pd.DataFrame({"position": [1.0, 1.0, -1.0], "net_return": [0.01, -0.005, 0.02]})

    tearsheet = compute_tearsheet(backtest_df)

    for key in EXPECTED_KEYS:
        assert key in tearsheet


def test_format_tearsheet_markdown_produces_table() -> None:
    tearsheet = {"total_return": 0.1234, "sharpe": 1.5}

    markdown = format_tearsheet_markdown(tearsheet, title="Test Sheet")

    assert "# Test Sheet" in markdown
    assert "| total_return | 0.1234 |" in markdown
