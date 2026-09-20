"""Performance comparison and optimization helpers."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from analytics.metrics import compute_metrics
from backtester.engine import BacktestResult, backtest


def metrics_row(
    name: str,
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Convert a BacktestResult into one labeled metrics row."""
    metrics = compute_metrics(
        result.returns,
        result.equity,
        result.position,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
    return {"Strategy": name, **metrics}


def compare_strategies(
    results: Dict[str, BacktestResult],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Build a comparison table for multiple backtests."""
    rows = [
        metrics_row(name, result, risk_free_rate, periods_per_year)
        for name, result in results.items()
    ]
    columns = [
        "Strategy",
        "CAGR",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Annualized Volatility",
        "Win Rate",
        "Total Return",
    ]
    return pd.DataFrame(rows)[columns]


def ma_parameter_sweep(
    close: pd.Series,
    train_index: pd.Index,
    fast_values: Iterable[int],
    slow_values: Iterable[int],
    cost_bps: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every valid MA pair on the training window only."""
    from strategies.moving_average import moving_average_crossover

    train_close = close.loc[train_index]
    rows = []
    for fast in fast_values:
        for slow in slow_values:
            if fast >= slow:
                continue
            signal = moving_average_crossover(train_close, fast=fast, slow=slow)
            result = backtest(train_close, signal, cost_bps=cost_bps)
            m = compute_metrics(
                result.returns,
                result.equity,
                result.position,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            )
            rows.append(
                {
                    "fast": fast,
                    "slow": slow,
                    "CAGR": m["CAGR"],
                    "Sharpe": m["Sharpe Ratio"],
                    "Max Drawdown": m["Maximum Drawdown"],
                    "Volatility": m["Annualized Volatility"],
                    "Win Rate": m["Win Rate"],
                    "Total Return": m["Total Return"],
                }
            )
    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No valid MA parameter combinations were generated.")
    heatmap = results.pivot(index="slow", columns="fast", values="Sharpe")
    return results.sort_values("Sharpe", ascending=False).reset_index(drop=True), heatmap
