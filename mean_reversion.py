"""Bollinger-band mean-reversion strategy."""
from __future__ import annotations

import pandas as pd


def bollinger_mean_reversion(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.Series:
    """Return a stateful long/flat Bollinger-band signal.

    Rule: enter long when the close is below the lower band; remain long until
    the close is at or above the middle band (20-day rolling mean).
    """
    if window <= 1:
        raise ValueError("window must be greater than 1.")
    if num_std <= 0:
        raise ValueError("num_std must be positive.")

    middle = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    lower = middle - num_std * std

    signal = pd.Series(0.0, index=close.index)
    long_state = False
    for date in close.index:
        if pd.isna(lower.loc[date]) or pd.isna(middle.loc[date]):
            signal.loc[date] = 0.0
            continue
        price = float(close.loc[date])
        if not long_state and price < float(lower.loc[date]):
            long_state = True
        elif long_state and price >= float(middle.loc[date]):
            long_state = False
        signal.loc[date] = 1.0 if long_state else 0.0
    return signal
