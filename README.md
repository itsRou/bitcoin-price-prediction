# Bitcoin Price Prediction — A Rigorous ML Benchmark

> **This is a research and ML-engineering project, not a trading system.** Crypto markets are
> close to efficient at short horizons. The goal is a leakage-free, walk-forward-validated
> benchmark of many algorithms on BTC/USDT log returns — not a promise of profit. Nothing here
> is investment advice.

## Status

✅ All 8 planned phases complete. See [Roadmap](#roadmap) and [Results](#results).

## What this project does

Predicts `log(close[t+h] / close[t])` for BTC/USDT at horizons h=1 and h=7, framed both as:

- **Regression**: the log-return value itself
- **Classification**: down / flat / up, with a volatility-scaled dead zone around zero

Success is **not** "low RMSE on price levels" (trivially gamed by predicting yesterday's price).
Success is:

1. Directional accuracy statistically significantly above 50% out-of-sample
2. Beating buy-and-hold on risk-adjusted return (Sharpe) after realistic fees and slippage
3. A clean, reproducible benchmark spanning statistical, classical ML, boosting, and deep
   learning models, validated with purged/embargoed walk-forward cross-validation

## Quickstart

```bash
uv sync
pre-commit install
pytest
```

```bash
# Full model sweep (~30 models, purged walk-forward CV) -> reports/results.md
python scripts/train_all.py

# Faster Tier-0-baselines-only version of the same thing
python scripts/run_baselines.py

# Interactive dashboard: data explorer, leaderboard, backtest tearsheet, single prediction
streamlit run app/streamlit_app.py
```

**Note on the CLI:** `src/btcpred/cli.py` (`btcpred fetch|features|train|backtest|report`) is
scaffolded with stub commands from Phase 1 and intentionally left unwired — the underlying
library functions (`btcpred.data.fetch`, `btcpred.features.build`, `btcpred.models.registry`,
`btcpred.backtest.engine`) are fully implemented and tested, and are what `scripts/train_all.py`
and the Streamlit app actually call. Wiring the CLI is a small, separate follow-up.

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1. Scaffolding | repo, CI, pre-commit, CLI stubs | ✅ done |
| 2. Data layer | OHLCV/macro/on-chain/sentiment fetchers, cleaning, daily merge | ✅ done |
| 3. Features | technical/return/regime/exogenous features, leakage-safe assembly | ✅ done |
| 4. Validation harness | purged walk-forward CV, metrics, Diebold-Mariano, Tier-0 baselines | ✅ done |
| 5. Classical + boosting sweep | linear/trees/SVM/KNN + XGBoost/LightGBM/CatBoost | ✅ done |
| 6. Deep learning sweep | LSTM/GRU/CNN/Transformer/TCN in PyTorch | ✅ done |
| 7. Ensembles + backtest | stacking, regime-conditional selection, cost-aware backtest | ✅ done |
| 8. Reporting + app | Streamlit dashboard, leaderboard, final README | ✅ done |

## Repository layout

See [`configs/`](configs/) for tunables, [`src/btcpred/`](src/btcpred/) for the package, and
[`reports/results.md`](reports/results.md) for the current leaderboard.

## Results (real BTC-USD data)

The full benchmark has been run on **real daily BTC-USD data (2017-01-01 to 2026-09-30)**, frozen
in [`data/snapshots/`](data/snapshots/) so every number is reproducible. 32 models plus the
random walk were evaluated with purged walk-forward CV over 13 folds (Sep 2019 to Sep 2026,
~2,560 out-of-sample forecasts per model). The write-up is in [`paper/main.pdf`](paper/main.pdf).

| | 1-day | 7-day |
|---|---|---|
| Models significantly *better* than the zero-return forecast (DM test, Holm-adjusted) | **0 / 32** | **0 / 32** |
| Models significantly *worse* | 23 / 32 | 25 / 32 |
| Best directional accuracy | 51.1% (not significant) | 54.4% (base-rate artefact) |
| Trading strategies beating buy-and-hold Sharpe after costs | 1 / 60 (deflated Sharpe 0.05) | — |

In short, **no model beats a random walk**, and the more flexible the model, the worse it does
out of sample. Removing non-stationary price-level features improves most models but still
leaves none better than the random walk (`reports/paper_stationary/`).

Reproduce everything:

```bash
python scripts/run_real_benchmark.py        # out-of-sample predictions -> reports/paper/oof_h*.csv
python scripts/analyze_real_benchmark.py    # metrics, DM/binomial tests, backtests, figures
python scripts/make_paper_tables.py         # LaTeX tables for the paper
STATIONARY=1 python scripts/run_real_benchmark.py 1 <models>   # robustness run
python scripts/analyze_robustness.py
```

`reports/results.md` (from `scripts/train_all.py`) is the original synthetic-data harness smoke
test and is kept for reference only.

## Honest limitations

- **Efficient market critique**: if a simple feature reliably predicted next-day BTC returns,
  it would be arbitraged away quickly. Any edge found here is likely small, regime-dependent,
  and fragile to costs.
- **Regime bias**: 2017–2021 data is dominated by a structural bull market; models trained
  on it may not generalize to sideways or bear regimes.
- **Transaction costs**: fees and slippage destroy most apparent statistical edges. All
  backtests here are cost-aware (0.1% fee + 0.05% slippage) for this reason.
- **Multiple-testing overfitting**: with 40+ models evaluated, some will beat baselines by
  chance. Reported results should be read alongside a deflated Sharpe ratio / significance
  test, not in isolation.
- **Not deployment-ready**: a real trading system needs a live data feed, latency budget,
  exchange/counterparty risk handling, and position sizing beyond what's modeled here.

## License

MIT — see [LICENSE](LICENSE).
