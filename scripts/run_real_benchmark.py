"""Run the full model registry on real BTC-USD daily data and save out-of-fold predictions.

This is the real-data counterpart to `train_all.py` (which only smoke-tests the harness on
synthetic data). It:

1. Loads a frozen snapshot of daily BTC-USD candles (fetched once from Yahoo Finance and
   saved under data/snapshots/ so every rerun uses identical data).
2. Builds the leakage-safe feature matrix for each horizon.
3. Runs every registered regression model through purged, embargoed, expanding-window
   walk-forward CV, scoring only folds with at least MIN_TRAIN_ROWS of training history.
4. Saves the out-of-fold predictions per horizon to reports/paper/oof_h{h}.csv, so all
   metrics, significance tests, and backtests in `analyze_real_benchmark.py` are computed
   from the exact same predictions.

Sequence (deep) models get the real preceding `window - 1` feature rows as context at
predict time instead of the padded approximation in `TorchSequenceRegressor.predict` --
those rows' features are known before the test period starts, so this adds no leakage.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from btcpred.features.build import build_feature_matrix
from btcpred.models.deep import TrainingConfig
from btcpred.models.registry import DEEP_ARCHITECTURES, get_regression_registry
from btcpred.validation.splitters import PurgedWalkForwardSplit

TICKER = "BTC-USD"
START_DATE = "2017-01-01"
END_DATE = "2026-09-30"  # inclusive; frozen for reproducibility
SNAPSHOT_PATH = Path(f"data/snapshots/btc_usd_daily_{START_DATE}_{END_DATE}.csv")
# Robustness variant: STATIONARY=1 drops price-level features (moving averages, OBV, MACD,
# ATR) whose test-period values fall outside the training range as BTC trends upward.
STATIONARY = os.environ.get("STATIONARY") == "1"
LEVEL_FEATURES = ("ema_", "sma_9", "sma_21", "sma_50", "sma_200", "obv", "macd", "atr_14")
OUT_DIR = Path("reports/paper_stationary" if STATIONARY else "reports/paper")

N_SPLITS = 16
MIN_TRAIN_ROWS = 730  # only score folds with >= ~2 years of training history
SKIPPED_MODELS = {"var"}  # degenerates to plain AR without exogenous columns
DEEP_CONFIG = TrainingConfig(max_epochs=50, patience=5, batch_size=64, val_fraction=0.15)


def load_snapshot() -> pd.DataFrame:
    if not SNAPSHOT_PATH.exists():
        end_exclusive = (pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(TICKER, start=START_DATE, end=end_exclusive, progress=False,
                          auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        raw.index = pd.to_datetime(raw.index, utc=True)
        raw.index.name = "timestamp"
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(SNAPSHOT_PATH)
    df = pd.read_csv(SNAPSHOT_PATH, index_col="timestamp", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_model(name, factory):
    model = factory()
    if name in DEEP_ARCHITECTURES:
        model.config = DEEP_CONFIG
    return model


def run_horizon(ohlcv: pd.DataFrame, h: int, only: set[str] | None) -> None:
    matrix = build_feature_matrix(ohlcv, horizons=(h,))
    feature_cols = [c for c in matrix.columns if not c.startswith(("y_reg_", "y_clf_"))]
    target = f"y_reg_h{h}"
    # Drop warm-up rows using the FULL feature set, so the stationary variant is scored on
    # exactly the same rows and folds as the main run.
    matrix = matrix.dropna(subset=[*feature_cols, target])
    if STATIONARY:
        feature_cols = [c for c in feature_cols if not c.startswith(LEVEL_FEATURES)
                        or c.startswith(("ema_cross", "sma_cross"))]
    X, y = matrix[feature_cols], matrix[target]

    splitter = PurgedWalkForwardSplit(n_splits=N_SPLITS, purge=h, embargo=h)
    folds = [(tr, te) for tr, te in splitter.split(X) if len(tr) >= MIN_TRAIN_ROWS]
    print(f"h={h}: {len(X)} rows, {len(feature_cols)} features, {len(folds)} scored folds, "
          f"test span {X.index[folds[0][1][0]].date()} -> {X.index[folds[-1][1][-1]].date()}",
          flush=True)

    out_path = OUT_DIR / f"oof_h{h}.csv"
    if out_path.exists():
        oof = pd.read_csv(out_path, index_col=0)
        oof.index = pd.to_datetime(oof.index, utc=True)
        oof = oof.reindex(X.index)
    else:
        oof = pd.DataFrame(index=X.index)
    oof["y_true"] = y
    oof["fold"] = np.nan
    for i, (_, te) in enumerate(folds):
        oof.iloc[te, oof.columns.get_loc("fold")] = i

    timings_path = OUT_DIR / f"timings_h{h}.csv"
    timings = pd.read_csv(timings_path, index_col=0)["seconds"].to_dict() \
        if timings_path.exists() else {}

    for name, factory in get_regression_registry().items():
        if name in SKIPPED_MODELS or (only and name not in only):
            continue
        if name in oof.columns and oof[name].notna().any() and not only:
            continue  # resume support
        t0 = time.time()
        preds = np.full(len(X), np.nan)
        for tr, te in folds:
            try:
                model = build_model(name, factory)
                model.fit(X.iloc[tr], y.iloc[tr])
                if name in DEEP_ARCHITECTURES:
                    ctx_start = te[0] - (model.window - 1)
                    p = np.asarray(model.predict(X.iloc[ctx_start: te[-1] + 1]))
                    preds[te] = p[-len(te):]
                else:
                    preds[te] = np.asarray(model.predict(X.iloc[te]))
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {name} fold failed: {exc}", flush=True)
        oof[name] = preds
        timings[name] = time.time() - t0
        print(f"  {name}: {timings[name]:.0f}s", flush=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        oof.to_csv(out_path)
        pd.Series(timings, name="seconds").to_csv(timings_path)


def main() -> None:
    horizons = [int(a) for a in sys.argv[1:2]] or [1, 7]
    only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    ohlcv = load_snapshot()
    print(f"snapshot: {len(ohlcv)} daily candles {ohlcv.index[0].date()} -> "
          f"{ohlcv.index[-1].date()}", flush=True)
    for h in horizons:
        run_horizon(ohlcv, h, only)


if __name__ == "__main__":
    main()
