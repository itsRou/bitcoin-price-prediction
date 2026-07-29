"""Tearsheet reporting for a completed backtest run."""

from __future__ import annotations

import pandas as pd

from btcpred.validation.metrics import (
    cagr,
    calmar_ratio,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover,
    win_rate,
)


def compute_tearsheet(backtest_df: pd.DataFrame) -> dict[str, float]:
    """Compute the standard set of financial metrics from a `run_backtest` output.

    Args:
        backtest_df: DataFrame with "position" and "net_return" columns, as returned by
            `btcpred.backtest.engine.run_backtest`.

    Returns:
        A flat dict of metric name -> value.
    """
    net_returns = backtest_df["net_return"]
    positions = backtest_df["position"]
    return {
        "total_return": total_return(net_returns),
        "cagr": cagr(net_returns),
        "sharpe": sharpe_ratio(net_returns),
        "sortino": sortino_ratio(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "calmar": calmar_ratio(net_returns),
        "win_rate": win_rate(net_returns),
        "profit_factor": profit_factor(net_returns),
        "turnover": turnover(positions),
    }


def format_tearsheet_markdown(
    tearsheet: dict[str, float], title: str = "Backtest Tearsheet"
) -> str:
    """Render a tearsheet dict as a GitHub-flavored markdown table."""
    lines = [f"# {title}", "", "| Metric | Value |", "|---|---|"]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in tearsheet.items())
    return "\n".join(lines)
