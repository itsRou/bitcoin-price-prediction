"""Tests for the vectorized cost-aware backtest engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcpred.backtest.engine import run_backtest, size_positions


def test_size_positions_fixed_mode_returns_sign() -> None:
    signal = pd.Series([0.02, -0.01, 0.0, 0.05])
    positions = size_positions(signal, mode="fixed")

    assert list(positions) == [1.0, -1.0, 0.0, 1.0]


def test_size_positions_vol_target_scales_by_inverse_vol() -> None:
    signal = pd.Series([0.01, 0.01])
    vol = pd.Series([0.02, 0.04])
    positions = size_positions(
        signal, mode="vol_target", realized_vol=vol, target_vol=0.02, max_leverage=10
    )

    assert positions.iloc[0] == pytest.approx(1.0)
    assert positions.iloc[1] == pytest.approx(0.5)


def test_size_positions_vol_target_requires_realized_vol() -> None:
    with pytest.raises(ValueError, match="realized_vol"):
        size_positions(pd.Series([0.01]), mode="vol_target")


def test_run_backtest_charges_cost_only_on_position_changes() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    positions = pd.Series([1.0, 1.0, -1.0, -1.0])

    result = run_backtest(returns, positions, fee=0.001, slippage=0.0005)

    np.testing.assert_allclose(result["cost"].to_numpy(), [0.0015, 0.0, 0.003, 0.0])


def test_run_backtest_equity_compounds_net_returns() -> None:
    returns = pd.Series([0.1, 0.1])
    positions = pd.Series([1.0, 1.0])

    result = run_backtest(returns, positions, fee=0.0, slippage=0.0)

    assert result["equity"].iloc[-1] == pytest.approx(1.1 * 1.1)


def test_run_backtest_flat_position_earns_nothing() -> None:
    returns = pd.Series([0.05, -0.05])
    positions = pd.Series([0.0, 0.0])

    result = run_backtest(returns, positions, fee=0.001, slippage=0.0005)

    assert (result["gross_return"] == 0.0).all()
    assert (result["cost"] == 0.0).all()
