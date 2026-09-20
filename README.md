# Systematic Trading Strategy Backtester

A modular Python framework for testing systematic trading strategies on daily SPY data, with explicit transaction-cost modeling, no-look-ahead signal handling, parameter sensitivity analysis, and a strict chronological train/test split. The project is structured like a small quant research codebase rather than a single notebook or script, making the workflow reproducible and suitable as a quantitative research / quantitative developer portfolio project.

## Features

- SPY daily OHLCV data using the supplied historical CSV.
- Moving-average crossover: 20-day SMA vs 50-day SMA.
- Bollinger-band mean reversion: 20-day mean ± 2 standard deviations.
- 12-month momentum: 252-trading-day trailing return.
- Buy-and-hold SPY benchmark.
- Vectorized long/flat backtesting.
- Explicit one-day signal lag to prevent look-ahead bias.
- Transaction-cost sweep at 0, 5, 10, and 20 bps.
- Total return, CAGR, annualized volatility, Sharpe ratio, maximum drawdown, and win rate.
- 25-combination MA parameter sensitivity grid.
- Training-only parameter selection followed by untouched out-of-sample evaluation.
- Equity curve, drawdown, and in-sample MA Sharpe heatmap plots.
- CSV outputs containing the computed results and run metadata.
- Offline deterministic self-test.

## Project structure

```text
quant-backtester/
├── data/
│   └── SPY_US_Data.csv         # Supplied SPY historical OHLCV data
├── strategies/
│   ├── __init__.py
│   ├── moving_average.py       # SMA crossover signal
│   ├── mean_reversion.py       # Bollinger-band signal
│   └── momentum.py             # 252-day momentum signal
├── backtester/
│   ├── __init__.py
│   ├── engine.py               # Signal alignment and vectorized backtest
│   ├── portfolio.py            # Equity-curve accounting
│   └── transaction_costs.py    # Turnover-based transaction costs
├── analytics/
│   ├── __init__.py
│   ├── metrics.py              # Performance metrics
│   └── performance.py          # Strategy comparison and MA optimization
├── plots/
│   ├── equity_curves_5bps.png
│   ├── drawdown_5bps.png
│   └── ma_sharpe_heatmap_is.png
├── results/
│   ├── final_summary_5bps.csv
│   ├── cost_sensitivity.csv
│   ├── ma_in_sample_vs_oos.csv
│   ├── ma_parameter_sweep.csv
│   └── run_metadata.json
├── main.py                     # Complete one-command pipeline
├── requirements.txt            # Pinned dependencies
├── .gitignore
└── README.md
```

## Setup

Create or clone the repository, enter the project directory, and install the pinned dependencies:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd quant-backtester
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the GitHub URL of your own fork/repository.

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The supplied dataset is already included, so the main pipeline does not need to contact Yahoo Finance or another external data service to reproduce the current results.

## How to run

Run the complete pipeline:

```bash
python main.py
```

The pipeline performs data validation/loading, strategy generation, 5 bps headline backtesting, the 25-combination MA training sweep, frozen-parameter out-of-sample evaluation, the 0/5/10/20 bps sensitivity analysis, and plot generation.

Run the offline framework test:

```bash
python main.py --self-test
```

Expected output:

```text
SELF-TEST: PASS
```

Generated outputs are saved under `results/` and `plots/`.

## Methodology

### No look-ahead bias

Signals are calculated from information available through the close of day *t*. The engine then shifts the signal by one trading day:

```python
position = signal.shift(1).fillna(0.0)
```

Therefore, a signal observed after day *t*'s close only changes the held position from day *t+1* onward. The same alignment is used when the selected MA parameters are carried into the out-of-sample window.

### Transaction costs

A trade is represented by a change in portfolio position. Daily turnover is:

```text
abs(position_t - position_(t-1))
```

The cost deducted from that day's portfolio return is:

```text
turnover × cost_bps / 10,000
```

The four tested cost assumptions are 0, 5, 10, and 20 basis points. Results in the main comparison use 5 bps.

### Train/test split

The 2,515-observation ten-year sample is split chronologically at approximately 70% / 30%:

- Training: 2015-09-18 to 2022-09-14.
- Out-of-sample test: 2022-09-15 to 2025-09-18.

The 25 valid MA parameter combinations are ranked using only training-period Sharpe ratio at 5 bps. The highest-Sharpe configuration is frozen and then evaluated on the untouched test period.

### Strategy definitions

**Moving-average crossover:** long when the 20-day SMA is above the 50-day SMA; otherwise cash.

**Bollinger mean reversion:** enter long when the close is below the lower 20-day Bollinger band. Remain long until the close is at or above the middle 20-day rolling mean, then return to cash.

**Momentum:** long when the trailing 252-trading-day close-to-close return is positive; otherwise cash.

**Benchmark:** long SPY continuously from the start of the sample, with the same proportional transaction-cost convention applied to the initial entry.

### Metrics

- **Total Return:** final portfolio value relative to initial portfolio value.
- **CAGR:** annualized compounded growth using 252 trading days per year.
- **Annualized Volatility:** daily return standard deviation annualized by `sqrt(252)`.
- **Sharpe Ratio:** annualized mean daily excess return divided by daily excess-return volatility; risk-free rate is 0%.
- **Maximum Drawdown:** largest peak-to-trough decline in the equity curve.
- **Win Rate:** percentage of positive active trading days; cash days are excluded for strategies.

## Results

The supplied SPY CSV contained 2,658 daily observations from 2015-02-25 through 2025-09-18. To match the requested ten-year window while using only available observations, the pipeline selected the latest ten years in that file: **2015-09-18 through 2025-09-18**, giving **2,515 trading days**. The final run used the user-supplied CSV directly.

### Headline comparison — 5 bps

| Strategy | CAGR | Sharpe | Max Drawdown | Volatility | Win Rate | Total Return |
|---|---:|---:|---:|---:|---:|---:|
| MA Crossover (20/50) | 8.21% | 0.75 | -29.24% | 11.39% | 55.57% | 119.71% |
| Bollinger Mean Reversion | 4.66% | 0.41 | -29.52% | 13.07% | 55.68% | 57.50% |
| Momentum (252d) | 8.79% | 0.69 | -31.37% | 13.63% | 56.27% | 131.89% |
| Buy & Hold SPY | 14.86% | 0.86 | -33.72% | 18.08% | 55.11% | 298.49% |

Over the full ten-year sample, buy-and-hold had the highest CAGR and Sharpe, while the three systematic strategies operated at lower annualized volatility. The Bollinger strategy was the most sensitive to trading costs in this implementation; momentum's CAGR and Sharpe changed the least over the tested cost range.

### Transaction-cost sensitivity

| Strategy | 0 bps CAGR | 0 bps Sharpe | 5 bps CAGR | 5 bps Sharpe | 10 bps CAGR | 10 bps Sharpe | 20 bps CAGR | 20 bps Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MA Crossover (20/50) | 8.45% | 0.77 | 8.21% | 0.75 | 7.96% | 0.73 | 7.48% | 0.69 |
| Bollinger Mean Reversion | 5.11% | 0.45 | 4.66% | 0.41 | 4.21% | 0.38 | 3.31% | 0.31 |
| Momentum (252d) | 8.92% | 0.70 | 8.79% | 0.69 | 8.67% | 0.68 | 8.42% | 0.66 |
| Buy & Hold SPY | 14.86% | 0.86 | 14.86% | 0.86 | 14.86% | 0.86 | 14.86% | 0.86 |

### In-sample vs out-of-sample

The grid contained **25 valid parameter combinations**. The highest in-sample Sharpe at 5 bps occurred at **10-day fast / 100-day slow**. Those parameters were frozen before the final 30% test period was evaluated.

| Period | Window | CAGR | Sharpe | Max Drawdown | Volatility | Win Rate | Total Return |
|---|---|---:|---:|---:|---:|---:|---:|
| In-sample | 2015-09-18 to 2022-09-14 | 10.47% | 0.93 | -16.60% | 11.46% | 56.15% | 100.47% |
| Out-of-sample | 2022-09-15 to 2025-09-18 | 15.01% | 1.28 | -9.02% | 11.40% | 56.05% | 52.03% |

In this specific historical test window, the optimized 10/100 configuration did not deteriorate out-of-sample: its test-period Sharpe and CAGR were higher than in-sample and its drawdown was smaller. This result should not be interpreted as proof of future robustness; it is one historical split and should be supplemented with walk-forward and multi-asset validation.

## Plots

### Equity curves

![Equity curves](plots/equity_curves_5bps.png)

### Drawdown

![Drawdown](plots/drawdown_5bps.png)

### In-sample MA Sharpe heatmap

![MA Sharpe heatmap](plots/ma_sharpe_heatmap_is.png)

## Limitations & disclaimer

This is a research backtester, not a production execution simulator. It uses daily bars, binary long/flat positions, and a proportional transaction-cost model. It does not model bid/ask spread dynamics, market impact, partial fills, borrow costs, taxes, exchange fees, or detailed intraday execution.

The strategy universe is intentionally small and the optimization uses one historical training window. The dataset is a single ETF, so results do not establish cross-asset robustness. The project also does not perform walk-forward re-optimization, regime conditioning, portfolio-level risk budgeting, or live/paper execution.

Because the supplied dataset contains daily SPY observations rather than a full corporate-action-adjusted institutional data feed, users should independently verify adjustment conventions before treating the results as investable research. This project is educational and is not investment advice.
