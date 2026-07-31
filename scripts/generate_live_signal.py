"""Generate today's real BUY/SELL/HOLD signal from live BTC market data.

Unlike `generate_prediction_dataset.py` (which evaluates models on historical, already-
resolved dates for the web explorer's backtest view), this script fetches *current* BTC
market data, runs the same purged walk-forward backtest to pick which curated model has
recently been most accurate, fits that model on all available history, and predicts the
return from the latest closed daily candle to the next one. The predicted return is turned
into a BUY/SELL/HOLD signal using the same volatility-scaled dead-zone rule already used for
the project's 3-class classification target (`btcpred.features.build.compute_targets`), so
the live signal is methodologically consistent with the rest of the benchmark.

Data comes from Yahoo Finance (`BTC-USD`) rather than the ccxt/Binance fetcher used
elsewhere in this project: Binance rejects requests from GitHub Actions' US-hosted runners
with an HTTP 451 geo-restriction, which would break the nightly scheduled refresh.

This is a research/educational demo, not investment advice -- see the "disclaimer" field in
the output and the site's honest-limitations section.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from btcpred.features.build import (
    DEFAULT_DEAD_ZONE_STD_MULTIPLIER,
    TARGET_VOL_WINDOW,
    build_feature_matrix,
)
from btcpred.models.deep import TrainingConfig
from btcpred.models.importance import explain_single_prediction
from btcpred.models.registry import get_regression_registry
from btcpred.models.selection import select_best_model_by_rolling_error
from btcpred.validation.metrics import directional_accuracy, mae, r_squared, rmse
from btcpred.validation.splitters import PurgedWalkForwardSplit

TICKER = "BTC-USD"
START_DATE = "2019-01-01"
HORIZON = 1
N_SPLITS = 12
LOOKBACK_ROWS = 900
SELECTION_WINDOW = 20
OUTPUT_PATH = Path("reports/live_signal.json")

CURATED_MODELS = (
    "buy_and_hold",
    "naive_zero",
    "auto_arima",
    "ridge",
    "random_forest",
    "xgboost",
    "lstm",
    "transformer",
)

MODEL_DESCRIPTIONS = {
    "buy_and_hold": "Always predicts a small fixed positive return -- the 'stay long' baseline.",
    "naive_zero": "Always predicts zero return -- the efficient-market / random-walk baseline.",
    "auto_arima": "Classical autoregressive model fit on the return series alone, no other "
    "features.",
    "ridge": "Linear regression with L2 regularization over the full engineered feature set.",
    "random_forest": "An ensemble of decision trees, each voting on a bootstrapped subset of "
    "history.",
    "xgboost": "Gradient-boosted decision trees that iteratively correct the previous trees' "
    "errors.",
    "lstm": "A recurrent neural network that reads the last 30 days of features as a sequence.",
    "transformer": "A self-attention encoder that weighs all 30 days in its input window jointly.",
}
TREE_MODELS = {"random_forest", "xgboost"}
FAST_DEEP_CONFIG = TrainingConfig(max_epochs=15, patience=4, batch_size=16, val_fraction=0.2)


def _build_model(name: str, factory: Any) -> Any:
    model = factory()
    if name in ("lstm", "transformer"):
        model.config = FAST_DEEP_CONFIG
    return model


def _drop_forming_candle(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Drop the most recent candle if it's still the current (incomplete) UTC day."""
    if ohlcv.empty:
        return ohlcv
    today = pd.Timestamp.now(tz="UTC").normalize()
    if ohlcv.index[-1] >= today:
        return ohlcv.iloc[:-1]
    return ohlcv


def _fetch_btc_ohlcv(start_date: str) -> pd.DataFrame:
    """Fetch daily BTC-USD candles from Yahoo Finance, normalized to lowercase OHLCV."""
    raw = yf.download(TICKER, start=start_date, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.index.name = "timestamp"
    return raw[["open", "high", "low", "close", "volume"]]


def main() -> None:
    ohlcv = _fetch_btc_ohlcv(START_DATE)
    ohlcv = _drop_forming_candle(ohlcv)

    matrix = build_feature_matrix(ohlcv, horizons=(HORIZON,))
    feature_cols = [c for c in matrix.columns if not c.startswith(("y_reg_", "y_clf_"))]
    target_col = f"y_reg_h{HORIZON}"

    valid = matrix.dropna(subset=feature_cols).tail(LOOKBACK_ROWS + 1)
    latest = valid.iloc[[-1]]
    train_rows = valid.iloc[:-1].dropna(subset=[target_col])

    X_train = train_rows[feature_cols]
    y_train = train_rows[target_col]
    X_latest = latest[feature_cols]
    latest_date = latest.index[0]
    current_price = float(ohlcv.loc[latest_date, "close"])

    registry = get_regression_registry()
    splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, purge=HORIZON, embargo=HORIZON)

    predictions: dict[str, dict[Any, float]] = {name: {} for name in CURATED_MODELS}
    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X_train), start=1):
        print(f"Fold {fold_num}/{N_SPLITS}: train={len(train_idx)} test={len(test_idx)}")
        for name in CURATED_MODELS:
            model = _build_model(name, registry[name])
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = np.asarray(model.predict(X_train.iloc[test_idx]))
            for date, pred in zip(X_train.index[test_idx], preds, strict=True):
                predictions[name][date] = float(pred)

    pred_df = pd.DataFrame(predictions).dropna(how="all")
    actual = y_train.reindex(pred_df.index)
    error_df = pred_df.sub(actual, axis=0).abs()
    selected_model_series = select_best_model_by_rolling_error(error_df, window=SELECTION_WINDOW)
    chosen = selected_model_series.dropna().iloc[-1]

    leaderboard = {}
    for name in CURATED_MODELS:
        pred_valid = pred_df[name].dropna()
        y_true = actual.reindex(pred_valid.index).to_numpy()
        y_pred = pred_valid.to_numpy()
        leaderboard[name] = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r_squared(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred),
            "n_predictions": int(len(pred_valid)),
        }

    final_model = _build_model(chosen, registry[chosen])
    final_model.fit(X_train, y_train)
    predicted_log_return = float(np.asarray(final_model.predict(X_latest))[0])
    predicted_next_price = current_price * float(np.exp(predicted_log_return))

    top_features = None
    if chosen in TREE_MODELS:
        try:
            top_features = explain_single_prediction(final_model, X_latest, top_n=3)
        except Exception:  # noqa: BLE001 - explanation is best-effort, never fatal
            top_features = None

    log_close = np.log(ohlcv["close"])
    trailing_std = log_close.diff().rolling(TARGET_VOL_WINDOW).std()
    dead_zone = (
        DEFAULT_DEAD_ZONE_STD_MULTIPLIER * float(trailing_std.loc[latest_date]) * (HORIZON**0.5)
    )

    if predicted_log_return >= dead_zone:
        signal = "BUY"
    elif predicted_log_return <= -dead_zone:
        signal = "SELL"
    else:
        signal = "HOLD"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "as_of_date": latest_date.strftime("%Y-%m-%d"),
        "current_price": current_price,
        "predicted_log_return": predicted_log_return,
        "predicted_next_price": predicted_next_price,
        "dead_zone_threshold": dead_zone,
        "signal": signal,
        "model": chosen,
        "model_description": MODEL_DESCRIPTIONS[chosen],
        "selection_reason": (
            "Lowest mean absolute error among the 8 curated models over the preceding "
            f"{SELECTION_WINDOW} out-of-fold predictions on real BTC/USDT data."
        ),
        "recent_stats": leaderboard[chosen],
        "leaderboard": leaderboard,
        "top_features": (
            [[feat, float(val)] for feat, val in top_features] if top_features else None
        ),
        "horizon_days": HORIZON,
        "lookback_rows": LOOKBACK_ROWS,
        "data_source": f"{TICKER} daily candles via Yahoo Finance",
        "disclaimer": (
            "Educational research demo, not investment advice. This signal comes from "
            "models with directional accuracy near a coin flip on out-of-sample data "
            "(see the leaderboard) and does not account for trading fees, slippage, or "
            "risk management. Do not use this to make real trading decisions."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Signal: {signal} | model={chosen} | pred_return={predicted_log_return:.5f}")
    print(f"Wrote live signal to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
