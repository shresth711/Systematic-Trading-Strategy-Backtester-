"""Time-series momentum strategy."""
from __future__ import annotations

import pandas as pd


def twelve_month_momentum(close: pd.Series, lookback: int = 252) -> pd.Series:
    """Return long when trailing lookback return is strictly positive."""
    if lookback <= 0:
        raise ValueError("lookback must be positive.")
    trailing_return = close.pct_change(lookback)
    return (trailing_return > 0).astype(float)
