"""Vectorized long/flat/short backtester with transaction costs and slippage.

Convention: `positions[t]` is the exposure held to earn `returns[t]` (the return already
realized over whatever horizon the target represents, e.g. `y_reg_h1`). A cost is charged
whenever the position changes size, proportional to `fee + slippage` per unit of turnover.
Positions start from an implicit flat (0) position before the first row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FEE = 0.001
DEFAULT_SLIPPAGE = 0.0005


def size_positions(
    raw_signal: pd.Series,
    mode: str = "fixed",
    realized_vol: pd.Series | None = None,
    target_vol: float = 0.01,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Convert a directional signal into a position size.

    Args:
        raw_signal: Any signed series (e.g. predicted return); only its sign is used.
        mode: "fixed" for unit long/short/flat, "vol_target" to scale exposure so
            realized volatility matches `target_vol`.
        realized_vol: Trailing realized volatility, required for "vol_target" mode.
        target_vol: Target per-period volatility for "vol_target" mode.
        max_leverage: Cap on the volatility-targeted scale factor.

    Returns:
        A position series with the same index as `raw_signal`.
    """
    direction = np.sign(raw_signal)
    if mode == "fixed":
        return direction
    if mode == "vol_target":
        if realized_vol is None:
            raise ValueError("realized_vol is required for vol_target sizing")
        scale = (target_vol / realized_vol).clip(upper=max_leverage).fillna(0.0)
        return direction * scale
    raise ValueError(f"Unknown sizing mode: {mode}")


def run_backtest(
    returns: pd.Series,
    positions: pd.Series,
    fee: float = DEFAULT_FEE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> pd.DataFrame:
    """Run a cost-aware vectorized backtest.

    Args:
        returns: Realized simple returns per period (convert log returns via `expm1`
            before calling, since equity compounding here assumes simple returns).
        positions: Desired position per period, aligned to `returns.index`.
        fee: Proportional fee per unit of position change (e.g. 0.001 = 0.1%).
        slippage: Proportional slippage per unit of position change.

    Returns:
        DataFrame with position, gross_return, cost, net_return, and equity columns.
    """
    positions = positions.reindex(returns.index).fillna(0.0)
    position_changes = positions.diff()
    position_changes.iloc[0] = positions.iloc[0]

    cost = (fee + slippage) * position_changes.abs()
    gross_return = positions * returns
    net_return = gross_return - cost
    equity = (1.0 + net_return).cumprod()

    return pd.DataFrame(
        {
            "position": positions,
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
        }
    )
