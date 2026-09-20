"""Portfolio accounting helpers."""
from __future__ import annotations

import pandas as pd


def equity_curve(returns: pd.Series, initial_capital: float = 100_000.0) -> pd.Series:
    """Compound daily returns into an equity curve."""
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()
