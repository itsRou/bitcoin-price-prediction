"""Generate a per-date, per-model prediction dataset for the interactive web explorer.

Precomputes predictions from a curated set of models across the full walk-forward test
coverage of the synthetic OHLCV demo dataset, so a web page can look up "what would each
model have predicted on date X" without training anything at request time. For every date
covered by some out-of-fold test window, this also runs the rolling-error model-selection
rule (`btcpred.models.selection`) to pick a single "featured" model per date and attaches a
local SHAP explanation when that model is tree-based.

Like the other smoke-test scripts, this runs against synthetic data -- see reports/results.md
and the README for why real BTC history hasn't been fetched yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from btcpred.features.build import build_feature_matrix
from btcpred.models.deep import TrainingConfig
from btcpred.models.importance import explain_single_prediction
from btcpred.models.registry import get_regression_registry
from btcpred.models.selection import select_best_model_by_rolling_error
from btcpred.validation.metrics import directional_accuracy, mae, r_squared, rmse
from btcpred.validation.splitters import PurgedWalkForwardSplit

HORIZON = 1
N_SPLITS = 15
N_ROWS = 800
SELECTION_WINDOW = 20
OUTPUT_PATH = Path("reports/predictions.json")

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


def _make_synthetic_ohlcv(n: int = N_ROWS, seed: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0002, 0.03, n)
    close = 5_000 * np.exp(np.cumsum(log_returns))
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    volume = rng.uniform(1_000, 10_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _build_model(name: str, factory: Any) -> Any:
    model = factory()
    if name in ("lstm", "transformer"):
        model.config = FAST_DEEP_CONFIG
    return model


def main() -> None:
    ohlcv = _make_synthetic_ohlcv()
    matrix = build_feature_matrix(ohlcv, horizons=(HORIZON,))
    feature_cols = [c for c in matrix.columns if not c.startswith(("y_reg_", "y_clf_"))]
    matrix = matrix.dropna(subset=[*feature_cols, f"y_reg_h{HORIZON}"])

    X = matrix[feature_cols]
    y = matrix[f"y_reg_h{HORIZON}"]
    close = ohlcv.loc[matrix.index, "close"]

    registry = get_regression_registry()
    splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, purge=HORIZON, embargo=HORIZON)

    predictions: dict[str, dict[Any, float]] = {name: {} for name in CURATED_MODELS}
    fitted_tree_models: dict[str, dict[Any, Any]] = {name: {} for name in TREE_MODELS}

    for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        print(f"Fold {fold_num}/{N_SPLITS}: train={len(train_idx)} test={len(test_idx)}")
        for name in CURATED_MODELS:
            model = _build_model(name, registry[name])
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = np.asarray(model.predict(X.iloc[test_idx]))
            for date, pred in zip(X.index[test_idx], preds, strict=True):
                predictions[name][date] = float(pred)
            if name in TREE_MODELS:
                for date in X.index[test_idx]:
                    fitted_tree_models[name][date] = model

    pred_df = pd.DataFrame(predictions).dropna(how="all")
    actual = y.reindex(pred_df.index)

    error_df = pred_df.sub(actual, axis=0).abs()
    selected_model = select_best_model_by_rolling_error(error_df, window=SELECTION_WINDOW)

    leaderboard = {}
    for name in CURATED_MODELS:
        valid = pred_df[name].dropna()
        y_true = actual.reindex(valid.index).to_numpy()
        y_pred = valid.to_numpy()
        leaderboard[name] = {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r_squared(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred),
            "n_predictions": int(len(valid)),
        }

    dates_payload = {}
    for date in pred_df.index:
        date_key = date.strftime("%Y-%m-%d")
        current_close = float(close.loc[date])
        actual_log_return = float(actual.loc[date])

        model_entries = {}
        for name in CURATED_MODELS:
            pred = pred_df.loc[date, name]
            if pd.isna(pred):
                continue
            pred = float(pred)
            model_entries[name] = {
                "pred_log_return": pred,
                "pred_next_close": current_close * float(np.exp(pred)),
                "abs_error": abs(pred - actual_log_return),
                "direction_correct": bool(np.sign(pred) == np.sign(actual_log_return)),
            }

        chosen = selected_model.loc[date] if date in selected_model.index else None
        chosen = chosen if isinstance(chosen, str) else None

        explanation = None
        if chosen is not None:
            top_features = None
            if chosen in TREE_MODELS and date in fitted_tree_models[chosen]:
                tree_model = fitted_tree_models[chosen][date]
                row = X.loc[[date]]
                try:
                    top_features = explain_single_prediction(tree_model, row, top_n=3)
                except Exception:  # noqa: BLE001 - explanation is best-effort, never fatal
                    top_features = None
            explanation = {
                "model_description": MODEL_DESCRIPTIONS[chosen],
                "top_features": (
                    [[feat, float(val)] for feat, val in top_features] if top_features else None
                ),
            }

        dates_payload[date_key] = {
            "close": current_close,
            "actual_log_return": actual_log_return,
            "actual_next_close": current_close * float(np.exp(actual_log_return)),
            "models": model_entries,
            "selected_model": chosen,
            "selection_reason": (
                "Lowest mean absolute error among candidates over the preceding "
                f"{SELECTION_WINDOW} out-of-fold predictions."
                if chosen is not None
                else None
            ),
            "explanation": explanation,
        }

    payload = {
        "meta": {
            "data_note": "Synthetic OHLCV demo data -- not real BTC history.",
            "horizon_days": HORIZON,
            "n_dates": len(dates_payload),
            "date_range": [min(dates_payload), max(dates_payload)] if dates_payload else [],
            "models": list(CURATED_MODELS),
            "model_descriptions": MODEL_DESCRIPTIONS,
            "selection_window": SELECTION_WINDOW,
        },
        "leaderboard": leaderboard,
        "dates": dates_payload,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(dates_payload)} dates x {len(CURATED_MODELS)} models to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
