"""Moving-average crossover strategy."""
from __future__ import annotations

import pandas as pd


def moving_average_crossover(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Return a binary long/flat signal from a fast/slow SMA crossover.

    Parameters
    ----------
    close:
        Adjusted closing price series.
    fast:
        Fast SMA lookback in trading days.
    slow:
        Slow SMA lookback in trading days.
    """
    if fast <= 0 or slow <= 0:
        raise ValueError("MA windows must be positive integers.")
    if fast >= slow:
        raise ValueError("fast must be smaller than slow.")

    fast_sma = close.rolling(fast, min_periods=fast).mean()
    slow_sma = close.rolling(slow, min_periods=slow).mean()
    return (fast_sma > slow_sma).astype(float)
