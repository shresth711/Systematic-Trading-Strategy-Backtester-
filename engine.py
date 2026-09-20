"""Vectorized backtesting engine with explicit one-day signal lag."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .portfolio import equity_curve
from .transaction_costs import transaction_cost_returns


@dataclass(frozen=True)
class BacktestResult:
    """Container for a strategy backtest."""

    returns: pd.Series
    position: pd.Series
    equity: pd.Series
    turnover: pd.Series
    costs: pd.Series


def prepare_position(signal: pd.Series) -> pd.Series:
    """Shift a signal by one trading day to prevent look-ahead bias.

    The signal is computed using information available at the close of day t,
    but the position only becomes active on day t+1. This explicit `.shift(1)`
    is the core look-ahead-bias safeguard in the framework.
    """
    if not signal.index.is_monotonic_increasing:
        signal = signal.sort_index()
    return signal.astype(float).shift(1).fillna(0.0)


def backtest(
    close: pd.Series,
    signal: pd.Series,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Backtest a long/flat signal against adjusted-close returns."""
    data = pd.concat([close.rename("close"), signal.rename("signal")], axis=1).dropna(subset=["close"])
    data["signal"] = data["signal"].fillna(0.0)
    position = prepare_position(data["signal"])
    return backtest_positioned(
        data["close"], position, cost_bps=cost_bps, initial_capital=initial_capital
    )


def backtest_positioned(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Backtest an already-lagged position series without introducing another lag."""
    frame = pd.concat([close.rename("close"), position.rename("position")], axis=1).dropna()
    frame["position"] = frame["position"].astype(float).clip(lower=0.0, upper=1.0)

    asset_returns = frame["close"].pct_change().fillna(0.0)
    costs = transaction_cost_returns(frame["position"], cost_bps)
    net_returns = frame["position"] * asset_returns - costs
    eq = equity_curve(net_returns, initial_capital=initial_capital)

    turnover = frame["position"].diff().abs()
    if not turnover.empty:
        turnover.iloc[0] = abs(float(frame["position"].iloc[0]))
    turnover = turnover.fillna(0.0)

    return BacktestResult(
        returns=net_returns.rename("return"),
        position=frame["position"].rename("position"),
        equity=eq.rename("equity"),
        turnover=turnover.rename("turnover"),
        costs=costs.rename("cost"),
    )


def buy_and_hold(
    close: pd.Series,
    cost_bps: float = 5.0,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Backtest a benchmark that enters on the first day and stays long."""
    position = pd.Series(1.0, index=close.index, name="position")
    return backtest_positioned(
        close, position, cost_bps=cost_bps, initial_capital=initial_capital
    )
