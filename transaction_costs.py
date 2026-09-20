"""Transaction-cost calculations."""
from __future__ import annotations

import pandas as pd


def turnover_from_positions(position: pd.Series) -> pd.Series:
    """Return absolute position changes, treating the first row as an entry from cash."""
    if position.empty:
        return position.copy()
    turnover = position.astype(float).diff().abs()
    turnover.iloc[0] = abs(float(position.iloc[0]))
    return turnover.fillna(0.0)


def transaction_cost_returns(position: pd.Series, cost_bps: float) -> pd.Series:
    """Return daily return drag from position changes for a long-only portfolio."""
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative.")
    return turnover_from_positions(position) * (cost_bps / 10_000.0)
