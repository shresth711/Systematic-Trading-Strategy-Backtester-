"""Run the complete systematic trading strategy backtest."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics.metrics import compute_metrics
from analytics.performance import compare_strategies, ma_parameter_sweep
from backtester.engine import BacktestResult, backtest, backtest_positioned, buy_and_hold
from strategies.mean_reversion import bollinger_mean_reversion
from strategies.momentum import twelve_month_momentum
from strategies.moving_average import moving_average_crossover

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
CACHE_FILE = DATA_DIR / "SPY_10y.csv"
SUPPLIED_FILE = DATA_DIR / "SPY_US_Data.csv"
INITIAL_CAPITAL = 100_000.0
PERIODS_PER_YEAR = 252
RISK_FREE_RATE = 0.0
HEADLINE_COST_BPS = 5.0
COST_SWEEP = [0.0, 5.0, 10.0, 20.0]
FAST_VALUES = [10, 15, 20, 25, 30]
SLOW_VALUES = [40, 50, 60, 80, 100]


def load_from_yfinance(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch SPY daily adjusted OHLCV data using yfinance."""
    import yfinance as yf

    raw = yf.download(
        "SPY",
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"yfinance response missing columns: {missing}")
    return raw[required].dropna().sort_index()


def load_from_stooq(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch SPY daily OHLCV data using Stooq via pandas-datareader."""
    from pandas_datareader import data as web_data

    raw = web_data.DataReader("SPY", "stooq", start, end).sort_index()
    if raw.empty:
        raise RuntimeError("Stooq returned no data.")
    raw = raw.rename(columns={c: c.title() for c in raw.columns})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Stooq response missing columns: {missing}")
    return raw[required].dropna().sort_index()


def download_or_load_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    """Load supplied/cached data, then fall back to online sources."""
    DATA_DIR.mkdir(exist_ok=True)

    if SUPPLIED_FILE.exists():
        raw = pd.read_csv(SUPPLIED_FILE, parse_dates=["Date"])
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise RuntimeError(f"Supplied CSV missing columns: {missing}")
        raw = raw[required].dropna().drop_duplicates("Date").sort_values("Date")
        raw = raw.set_index("Date")
        supplied_end = raw.index.max()
        supplied_start = supplied_end - pd.DateOffset(years=10)
        data = raw.loc[supplied_start:supplied_end].copy()
        data.index.name = "Date"
        return data, "user-supplied CSV (SPY_US_Data.csv; latest 10-year window)"

    if CACHE_FILE.exists():
        cached = pd.read_csv(CACHE_FILE, parse_dates=["Date"], index_col="Date")
        if not cached.empty:
            return cached.sort_index(), "local cache"

    errors = []
    try:
        data = load_from_yfinance(start, end)
        data.index.name = "Date"
        data.to_csv(CACHE_FILE)
        return data, "yfinance (auto_adjust=True)"
    except Exception as exc:
        errors.append(f"yfinance: {exc}")

    try:
        data = load_from_stooq(start, end)
        data.index.name = "Date"
        data.to_csv(CACHE_FILE)
        return data, "Stooq via pandas-datareader"
    except Exception as exc:
        errors.append(f"Stooq: {exc}")

    raise RuntimeError(
        "Unable to download SPY data from either source. " + " | ".join(errors)
        + "\nThe project itself is complete; run it on a machine with internet access."
    )

def normalize_date_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the requested ten-year calendar window ending today."""
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=10)
    return start, end


def split_train_test(index: pd.Index, train_fraction: float = 0.70) -> tuple[pd.Index, pd.Index]:
    """Split chronological observations into approximately 70/30 train/test windows."""
    cutoff = max(1, int(len(index) * train_fraction))
    return index[:cutoff], index[cutoff:]


def build_strategy_signals(close: pd.Series) -> Dict[str, pd.Series]:
    """Create the three requested strategy signals."""
    return {
        "MA Crossover (20/50)": moving_average_crossover(close, 20, 50),
        "Bollinger Mean Reversion": bollinger_mean_reversion(close, 20, 2.0),
        "Momentum (252d)": twelve_month_momentum(close, 252),
    }


def plot_equity_curves(results: Dict[str, BacktestResult], path: Path) -> None:
    """Save a normalized equity-curve comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, result in results.items():
        normalized = result.equity / result.equity.iloc[0]
        ax.plot(normalized.index, normalized, label=name, linewidth=1.5)
    ax.set_title("SPY Systematic Strategies vs Buy & Hold — 5 bps")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_drawdowns(results: Dict[str, BacktestResult], path: Path) -> None:
    """Save a drawdown comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for name, result in results.items():
        dd = result.equity / result.equity.cummax() - 1.0
        ax.plot(dd.index, dd * 100.0, label=name, linewidth=1.25)
    ax.set_title("Drawdowns — SPY Systematic Strategies vs Buy & Hold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ma_heatmap(heatmap: pd.DataFrame, path: Path) -> None:
    """Save the in-sample MA Sharpe-ratio heatmap."""
    fig, ax = plt.subplots(figsize=(10, 7))
    values = heatmap.values.astype(float)
    im = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(heatmap.columns)), labels=[str(x) for x in heatmap.columns])
    ax.set_yticks(np.arange(len(heatmap.index)), labels=[str(x) for x in heatmap.index])
    ax.set_xlabel("Fast MA")
    ax.set_ylabel("Slow MA")
    ax.set_title("In-Sample Sharpe Ratio — MA Parameter Sweep (5 bps)")
    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_self_test() -> None:
    """Run a deterministic offline smoke test for the framework."""
    dates = pd.bdate_range("2024-01-01", periods=400)
    base = 100.0 * np.exp(np.cumsum(np.full(len(dates), 0.0002)))
    close = pd.Series(base, index=dates, name="Close")
    sig = pd.Series(0.0, index=dates)
    sig.iloc[10:40] = 1.0
    result = backtest(close, sig, cost_bps=5)
    assert len(result.equity) == len(close)
    assert result.position.iloc[10] == 0.0
    assert result.position.iloc[11] == 1.0
    assert abs(result.costs.iloc[11] - 0.0005) < 1e-12
    assert abs(result.costs.iloc[41] - 0.0005) < 1e-12
    assert compute_metrics(result.returns, result.equity)["Total Return"] >= 0.0
    print("SELF-TEST: PASS")


def main() -> None:
    """Run data acquisition, backtests, optimization, validation, tables, and plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run an offline deterministic smoke test.")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    start, end = normalize_date_range()
    data, data_source = download_or_load_data(start, end)
    if len(data) < 1000:
        raise RuntimeError(f"Only {len(data)} observations were loaded; expected roughly ten years of daily data.")

    close = data["Close"].astype(float)
    train_index, test_index = split_train_test(close.index)
    train_start, train_end = train_index[0], train_index[-1]
    test_start, test_end = test_index[0], test_index[-1]

    signals = build_strategy_signals(close)
    full_results: Dict[str, BacktestResult] = {
        name: backtest(close, signal, cost_bps=HEADLINE_COST_BPS, initial_capital=INITIAL_CAPITAL)
        for name, signal in signals.items()
    }
    full_results["Buy & Hold SPY"] = buy_and_hold(
        close, cost_bps=HEADLINE_COST_BPS, initial_capital=INITIAL_CAPITAL
    )

    summary = compare_strategies(
        full_results, risk_free_rate=RISK_FREE_RATE, periods_per_year=PERIODS_PER_YEAR
    )
    summary.to_csv(RESULTS_DIR / "final_summary_5bps.csv", index=False)

    sweep, heatmap = ma_parameter_sweep(
        close,
        train_index=train_index,
        fast_values=FAST_VALUES,
        slow_values=SLOW_VALUES,
        cost_bps=HEADLINE_COST_BPS,
        risk_free_rate=RISK_FREE_RATE,
        periods_per_year=PERIODS_PER_YEAR,
    )
    sweep.to_csv(RESULTS_DIR / "ma_parameter_sweep.csv", index=False)
    best = sweep.iloc[0]
    best_fast, best_slow = int(best["fast"]), int(best["slow"])

    best_signal_full = moving_average_crossover(close, best_fast, best_slow)
    best_position_full = best_signal_full.astype(float).shift(1).fillna(0.0)
    train_best = backtest_positioned(
        close.loc[train_index],
        best_position_full.loc[train_index],
        cost_bps=HEADLINE_COST_BPS,
        initial_capital=INITIAL_CAPITAL,
    )
    oos_best = backtest_positioned(
        close.loc[test_index],
        best_position_full.loc[test_index],
        cost_bps=HEADLINE_COST_BPS,
        initial_capital=INITIAL_CAPITAL,
    )
    is_metrics = compute_metrics(
        train_best.returns, train_best.equity, train_best.position, RISK_FREE_RATE, PERIODS_PER_YEAR
    )
    oos_metrics = compute_metrics(
        oos_best.returns, oos_best.equity, oos_best.position, RISK_FREE_RATE, PERIODS_PER_YEAR
    )
    validation = pd.DataFrame([
        {
            "Strategy": f"Optimized MA ({best_fast}/{best_slow}) — In-Sample",
            "Window": f"{train_start.date()} to {train_end.date()}",
            "CAGR": is_metrics["CAGR"],
            "Sharpe": is_metrics["Sharpe Ratio"],
            "Max Drawdown": is_metrics["Maximum Drawdown"],
            "Volatility": is_metrics["Annualized Volatility"],
            "Win Rate": is_metrics["Win Rate"],
            "Total Return": is_metrics["Total Return"],
        },
        {
            "Strategy": f"Optimized MA ({best_fast}/{best_slow}) — Out-of-Sample",
            "Window": f"{test_start.date()} to {test_end.date()}",
            "CAGR": oos_metrics["CAGR"],
            "Sharpe": oos_metrics["Sharpe Ratio"],
            "Max Drawdown": oos_metrics["Maximum Drawdown"],
            "Volatility": oos_metrics["Annualized Volatility"],
            "Win Rate": oos_metrics["Win Rate"],
            "Total Return": oos_metrics["Total Return"],
        },
    ])
    validation.to_csv(RESULTS_DIR / "ma_in_sample_vs_oos.csv", index=False)

    sensitivity_rows = []
    for cost in COST_SWEEP:
        for name, signal in signals.items():
            result = backtest(close, signal, cost_bps=cost, initial_capital=INITIAL_CAPITAL)
            metrics = compute_metrics(
                result.returns, result.equity, result.position, RISK_FREE_RATE, PERIODS_PER_YEAR
            )
            sensitivity_rows.append({
                "Strategy": name,
                "Cost (bps)": cost,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe Ratio"],
            })
        bh = buy_and_hold(close, cost_bps=cost, initial_capital=INITIAL_CAPITAL)
        bh_metrics = compute_metrics(
            bh.returns, bh.equity, bh.position, RISK_FREE_RATE, PERIODS_PER_YEAR
        )
        sensitivity_rows.append({
            "Strategy": "Buy & Hold SPY",
            "Cost (bps)": cost,
            "CAGR": bh_metrics["CAGR"],
            "Sharpe": bh_metrics["Sharpe Ratio"],
        })
    cost_sensitivity = pd.DataFrame(sensitivity_rows)
    cost_sensitivity.to_csv(RESULTS_DIR / "cost_sensitivity.csv", index=False)

    plot_equity_curves(full_results, PLOTS_DIR / "equity_curves_5bps.png")
    plot_drawdowns(full_results, PLOTS_DIR / "drawdown_5bps.png")
    plot_ma_heatmap(heatmap, PLOTS_DIR / "ma_sharpe_heatmap_is.png")

    metadata = pd.Series({
        "Data source": data_source,
        "Data start": data.index.min().date(),
        "Data end": data.index.max().date(),
        "Observations": len(data),
        "Train start": train_start.date(),
        "Train end": train_end.date(),
        "Test start": test_start.date(),
        "Test end": test_end.date(),
        "Risk-free rate": RISK_FREE_RATE,
        "Headline cost bps": HEADLINE_COST_BPS,
        "Best in-sample fast MA": best_fast,
        "Best in-sample slow MA": best_slow,
    })
    import json
    (RESULTS_DIR / "run_metadata.json").write_text(
        json.dumps({k: str(v) if hasattr(v, "isoformat") else v for k, v in metadata.items()}, indent=2),
        encoding="utf-8",
    )

    print("\n=== DATA ===")
    print(f"Source: {data_source}")
    print(f"Range: {data.index.min().date()} to {data.index.max().date()} ({len(data):,} trading days)")
    print(f"Train: {train_start.date()} to {train_end.date()} | Test: {test_start.date()} to {test_end.date()}")
    print("Risk-free rate: 0.00% | Trading days/year: 252")
    print("Win rate: % of positive active trading days; flat days excluded for strategies.")

    print("\n=== SUMMARY — 5 BPS ===")
    printable = summary.copy()
    for col in ["CAGR", "Maximum Drawdown", "Annualized Volatility", "Win Rate", "Total Return"]:
        printable[col] = printable[col].map(lambda x: f"{x:.2%}")
    printable["Sharpe Ratio"] = printable["Sharpe Ratio"].map(lambda x: f"{x:.2f}")
    print(printable.to_string(index=False))

    print(f"\n=== MA SENSITIVITY — IN-SAMPLE ({len(sweep)} valid combinations) ===")
    print(sweep.to_string(index=False, formatters={
        "CAGR": "{:.2%}".format,
        "Sharpe": "{:.2f}".format,
        "Max Drawdown": "{:.2%}".format,
        "Volatility": "{:.2%}".format,
        "Win Rate": "{:.2%}".format,
        "Total Return": "{:.2%}".format,
    }))
    print(f"\nSelected parameters: fast={best_fast}, slow={best_slow}, chosen by highest in-sample Sharpe at 5 bps.")

    print("\n=== OPTIMIZED MA — TRAIN VS TEST ===")
    vprint = validation.copy()
    for col in ["CAGR", "Max Drawdown", "Volatility", "Win Rate", "Total Return"]:
        vprint[col] = vprint[col].map(lambda x: f"{x:.2%}")
    vprint["Sharpe"] = vprint["Sharpe"].map(lambda x: f"{x:.2f}")
    print(vprint.to_string(index=False))

    print("\n=== COST SENSITIVITY ===")
    cprint = cost_sensitivity.pivot(index="Strategy", columns="Cost (bps)", values=["CAGR", "Sharpe"])
    for metric in ["CAGR", "Sharpe"]:
        print(f"\n{metric}:")
        table = cprint[metric].copy()
        if metric == "CAGR":
            table = table.map(lambda x: f"{x:.2%}")
        else:
            table = table.map(lambda x: f"{x:.2f}")
        print(table.to_string())

    print("\n=== FILES ===")
    print(f"Summary: {RESULTS_DIR / 'final_summary_5bps.csv'}")
    print(f"Cost sensitivity: {RESULTS_DIR / 'cost_sensitivity.csv'}")
    print(f"Train/test validation: {RESULTS_DIR / 'ma_in_sample_vs_oos.csv'}")
    print(f"Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
