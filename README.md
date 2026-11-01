# Regime-Switching Pairs Trading with Cointegration Break Detection

This is a runnable prototype of the strategy architecture:

- Hybrid regime detection: continuous jump model plus a forecasting layer.
- Regime-specific cointegration: rolling hedge ratios weighted by regime probability.
- Cointegration break detection: residual instability, hedge-ratio drift, and error-correction weakness.
- Adaptive trading rules: entry/exit z-score thresholds vary by regime.
- Regime-aware risk: volatility targeting, regime confidence scaling, and break-risk de-risking.

The code is dependency-light and runs with Python + NumPy. A fitted XGBoost-style
classifier can be passed into `RegimeSwitchingPairsTrader(..., regime_forecaster_model=model)`
as long as it exposes `predict_proba`. Without one, the demo uses a Markov forecasting
fallback behind the same interface.

## Run

```bash
python3 run_demo.py
```

This writes:

- `demo_equity_curve.csv`
- `demo_trades.csv`
- `demo_summary.json`

To run on real daily equity data:

```bash
python3 run_real_pair.py
```

The default pair is `KO`/`PEP` from `2018-01-01` through `2026-06-10`.
The runner tries Stooq first, then Yahoo Finance as a fallback. It caches raw
provider files into `data_cache/`, aligns shared dates, and writes
`real_KO_PEP_*` result files.

For the more aggressive research workflow, scan multiple candidate pairs and
build a selected-pair portfolio:

```bash
python3 run_pair_scan.py
```

This writes:

- `pair_scan_results.csv`
- `pair_scan_summary.json`
- `pair_scan_portfolio_equity.csv`
- `pair_scan_portfolio_weights.csv`
- `algorithm_3d_model.html`

For a higher-return concentrated run:

```bash
python3 run_pair_scan.py --target-vol 0.12 --max-portfolio-leverage 20 --top-n 1
```

For the single-digit leverage return target:

```bash
python3 run_pair_scan.py --mode single-digit-return --objective return --target-vol 0.20 --max-portfolio-leverage 9 --top-n 1
```

This mode uses a capital-efficiency config grid and return-focused selection.
In the latest cached run, it selected `BMY/LLY` and produced annual return above
10% while keeping reported portfolio leverage below 9x.

For a diversified aggressive run:

```bash
python3 run_pair_scan.py --target-vol 0.12 --max-portfolio-leverage 8 --top-n 4
```

You can also run a different single pair:

```bash
python3 run_real_pair.py --pair XOM,CVX --profile high
```

If this workspace has DNS/network disabled, the command will print a short
message instead of a traceback. You can still place cached provider files in
`data_cache/` and rerun.

## Strategy Notes

The model has three regimes:

- `bull`: wider entry threshold, slower mean reversion assumed.
- `bear`: tighter entry threshold, higher volatility sensitivity.
- `transition`: no new entries; existing risk is reduced.

The strategy explicitly avoids the common pairs-trading failure mode where a spread looks
statistically extreme during a cointegration breakdown. Break probability gates both signal
generation and position sizing, and fresh entries are blocked when the z-score is already
beyond the stop band.

## Improvements Added

- Real-data downloader/loader for Stooq daily prices.
- Adaptive entry thresholds calibrated from recent regime-specific z-scores.
- Break detector with CUSUM, hedge-ratio drift, residual mean shift, variance shift,
  and error-correction weakness.
- Position sizing based on realized spread risk rather than broad pair volatility.
- Regime and break-risk PnL diagnostics in the summary JSON.
- High-conviction config grid with pair selection and out-of-sample evaluation.
- Spread direction search: each pair can choose mean-reversion or breakout-style
  spread behavior based on train/validation performance.
- Portfolio-level volatility targeting and train-score weighting.

## Sharpe Target Note

Leverage can raise return, but it cannot raise Sharpe by itself. A `1.5` Sharpe
requires either stronger pair selection, better signals, lower costs, or a larger
candidate universe. The scanner now includes a broader default universe; if only
cached prices are available, missing symbols are skipped.

## 3D Model Output

Every `run_pair_scan.py` run now generates `algorithm_3d_model.html`.
Open it in a browser after the scan finishes. It is an interactive Three.js model
of the algorithm: candidate pair towers, selected portfolio core, regime engine,
break-risk shield, leverage coil, and equity path.

## Alpaca Paper Trading

Set your Alpaca paper credentials:

```bash
export ALPACA_API_KEY_ID="your_paper_key"
export ALPACA_API_SECRET_KEY="your_paper_secret"
```

Dry-run the latest signal against your Alpaca paper account:

```bash
python3 run_alpaca_paper.py --pair BMY,LLY --capital-fraction 0.10
```

Submit the displayed orders to Alpaca paper trading:

```bash
python3 run_alpaca_paper.py --pair BMY,LLY --capital-fraction 0.10 --execute
```

The script compares the strategy's target pair exposure to your current paper
positions and only sends the difference. It defaults to dry-run mode.

## Real Data Integration

Replace the synthetic series in `run_demo.py` with two aligned price arrays:

```python
from strategy import StrategyConfig, RegimeSwitchingPairsTrader

config = StrategyConfig()
trader = RegimeSwitchingPairsTrader(config, regime_forecaster_model=None)
result = trader.run(price_a, price_b)
```

Use daily or intraday close prices. The arrays must be positive, same length, and time aligned.
