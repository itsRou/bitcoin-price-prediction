# Bitcoin Price Prediction — A Rigorous ML Benchmark

> **This is a research and ML-engineering project, not a trading system.** Crypto markets are
> close to efficient at short horizons. The goal is a leakage-free, walk-forward-validated
> benchmark of many algorithms on BTC/USDT log returns — not a promise of profit. Nothing here
> is investment advice.

## Status

🚧 Phase 5 (classical + boosting sweep) — up next. See [Roadmap](#roadmap).

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
btcpred fetch      # pull OHLCV + macro + on-chain + sentiment data
btcpred features   # build the feature matrix and targets
btcpred train --model xgboost
btcpred backtest --model xgboost
btcpred report     # leaderboard + tearsheet
```

## Roadmap

| Phase | Deliverable | Status |
|---|---|---|
| 1. Scaffolding | repo, CI, pre-commit, CLI stubs | ✅ done |
| 2. Data layer | OHLCV/macro/on-chain/sentiment fetchers, cleaning, daily merge | ✅ done |
| 3. Features | technical/return/regime/exogenous features, leakage-safe assembly | ✅ done |
| 4. Validation harness | purged walk-forward CV, metrics, Diebold-Mariano, Tier-0 baselines | ✅ done |
| 5. Classical + boosting sweep | linear/trees/SVM/KNN + XGBoost/LightGBM/CatBoost | 🚧 in progress |
| 6. Deep learning sweep | LSTM/GRU/CNN/Transformer/TCN in PyTorch | ⬜ |
| 7. Ensembles + backtest | stacking, regime-conditional selection, cost-aware backtest | ⬜ |
| 8. Reporting + app | Streamlit dashboard, leaderboard, final README | ⬜ |

## Repository layout

See [`configs/`](configs/) for tunables, [`src/btcpred/`](src/btcpred/) for the package, and
[`reports/results.md`](reports/results.md) for the leaderboard once models are trained.

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
