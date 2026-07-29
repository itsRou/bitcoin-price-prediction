"""Streamlit dashboard for the Bitcoin price prediction benchmark.

Runs out of the box on synthetic OHLCV data (no fetched BTC history is required), so
anyone cloning the repo can `streamlit run app/streamlit_app.py` and see the whole
pipeline work end to end. Swap `load_ohlcv()` for a real `data/raw/*.parquet` file once
`btcpred fetch` has been run against live data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from btcpred.backtest.engine import run_backtest, size_positions
from btcpred.backtest.report import compute_tearsheet
from btcpred.features.build import build_feature_matrix
from btcpred.models.deep import TrainingConfig
from btcpred.models.registry import DEEP_ARCHITECTURES, get_regression_registry
from btcpred.viz.plots import plot_equity_curve, plot_predictions_vs_actual

HORIZON = 1
RESULTS_PATH = Path("reports/results.md")
FAST_DEEP_CONFIG = TrainingConfig(max_epochs=15, patience=4, batch_size=16, val_fraction=0.2)

st.set_page_config(page_title="BTC Prediction Benchmark", layout="wide")


@st.cache_data
def load_ohlcv(n: int = 1000, seed: int = 1) -> pd.DataFrame:
    """Synthetic BTC-like OHLCV series. Replace with a real data/raw/*.parquet read once fetched."""
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


@st.cache_data
def load_feature_matrix() -> pd.DataFrame:
    ohlcv = load_ohlcv()
    matrix = build_feature_matrix(ohlcv, horizons=(HORIZON,))
    feature_cols = [c for c in matrix.columns if not c.startswith(("y_reg_", "y_clf_"))]
    return matrix.dropna(subset=[*feature_cols, f"y_reg_h{HORIZON}"])


def get_feature_columns(matrix: pd.DataFrame) -> list[str]:
    return [c for c in matrix.columns if not c.startswith(("y_reg_", "y_clf_"))]


def build_model(name: str) -> Any:
    factory = get_regression_registry()[name]
    model = factory()
    if name in DEEP_ARCHITECTURES:
        model.config = FAST_DEEP_CONFIG
    return model


st.title("Bitcoin Price Prediction — Research Benchmark")
st.caption(
    "A leakage-free, walk-forward-validated benchmark of ML algorithms on BTC/USDT log "
    "returns. This is a research project, not investment advice."
)

tab_data, tab_leaderboard, tab_backtest, tab_predict = st.tabs(
    ["Data Explorer", "Model Leaderboard", "Backtest Tearsheet", "Single Prediction"]
)

with tab_data:
    st.subheader("OHLCV Data")
    ohlcv = load_ohlcv()
    st.line_chart(ohlcv["close"], height=300)
    st.caption("Synthetic demo data (see docstring at the top of this file to swap in real data).")

    st.subheader("Feature Matrix Preview")
    matrix = load_feature_matrix()
    st.dataframe(matrix.tail(20), width="stretch")
    st.caption(
        f"{matrix.shape[0]} rows x {matrix.shape[1]} columns after dropping NaN warm-up rows."
    )

with tab_leaderboard:
    st.subheader("Model Leaderboard")
    if RESULTS_PATH.exists():
        st.markdown(RESULTS_PATH.read_text())
    else:
        st.info(
            "No leaderboard yet. Run `python scripts/train_all.py` (or "
            "`python scripts/run_baselines.py` for a quicker version) to generate "
            "reports/results.md."
        )

with tab_backtest:
    st.subheader("Backtest a Model")
    matrix = load_feature_matrix()
    feature_cols = get_feature_columns(matrix)
    registry_names = sorted(get_regression_registry())
    model_name = st.selectbox("Model", registry_names, index=registry_names.index("xgboost"))

    if st.button("Run backtest", type="primary"):
        with st.spinner(f"Training {model_name} and running the backtest..."):
            split_at = int(len(matrix) * 0.8)
            train, test = matrix.iloc[:split_at], matrix.iloc[split_at:]

            model = build_model(model_name)
            model.fit(train[feature_cols], train[f"y_reg_h{HORIZON}"])
            preds = np.asarray(model.predict(test[feature_cols]))

            simple_returns = np.expm1(test[f"y_reg_h{HORIZON}"])
            positions = size_positions(pd.Series(preds, index=test.index), mode="fixed")
            backtest_df = run_backtest(simple_returns, positions)
            tearsheet = compute_tearsheet(backtest_df)

        col_chart, col_metrics = st.columns([2, 1])
        with col_chart:
            st.pyplot(plot_equity_curve(backtest_df))
        with col_metrics:
            st.dataframe(
                pd.DataFrame(tearsheet.items(), columns=["metric", "value"]),
                hide_index=True,
                width="stretch",
            )
        st.pyplot(plot_predictions_vs_actual(test[f"y_reg_h{HORIZON}"], preds))

with tab_predict:
    st.subheader("Single-Row Prediction")
    matrix = load_feature_matrix()
    feature_cols = get_feature_columns(matrix)
    registry_names = sorted(get_regression_registry())
    predict_model_name = st.selectbox(
        "Model", registry_names, index=registry_names.index("xgboost"), key="predict_model"
    )

    if st.button("Predict latest row"):
        with st.spinner(f"Training {predict_model_name}..."):
            train = matrix.iloc[:-1]
            latest_row = matrix.iloc[[-1]]

            model = build_model(predict_model_name)
            model.fit(train[feature_cols], train[f"y_reg_h{HORIZON}"])
            prediction = float(np.asarray(model.predict(latest_row[feature_cols]))[0])

        st.metric(
            label=f"Predicted log return (h={HORIZON})",
            value=f"{prediction:.5f}",
            delta=f"{np.expm1(prediction) * 100:.3f}% simple return",
        )
        st.caption(f"As of {latest_row.index[0].date()}")
