"""Performance metrics."""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """Return cumulative return from the equity curve."""
    if equity.empty or equity.iloc[0] <= 0:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Return CAGR using the 252-trading-day annualization convention."""
    if equity.empty or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    n = len(equity)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Return annualized standard deviation of daily returns."""
    if len(returns.dropna()) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    """Return annualized Sharpe ratio from daily simple returns."""
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - daily_rf
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Return maximum peak-to-trough drawdown as a negative fraction."""
    if equity.empty:
        return float("nan")
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())


def win_rate(
    returns: pd.Series, position: pd.Series | None = None
) -> float:
    """Return percentage of positive active trading days.

    Flat/cash days are excluded from the denominator for strategies. For a
    benchmark with continuous exposure, every day is an active trading day.
    """
    clean = returns.copy()
    if position is not None:
        active = position.reindex(clean.index).fillna(0.0) > 0
        clean = clean.loc[active]
    clean = clean.dropna()
    if clean.empty:
        return float("nan")
    return float((clean > 0).mean())


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    position: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Mapping[str, float]:
    """Compute the requested headline metrics in one call."""
    return {
        "Total Return": total_return(equity),
        "CAGR": cagr(equity, periods_per_year),
        "Annualized Volatility": annualized_volatility(returns, periods_per_year),
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "Maximum Drawdown": max_drawdown(equity),
        "Win Rate": win_rate(returns, position),
    }
