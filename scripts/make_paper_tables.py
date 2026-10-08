"""Write the paper's LaTeX tables straight from the analysis CSVs (no hand-copied numbers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path("reports/paper")
OUT = Path("paper/tables")

NAMES = {
    "naive_zero": "Zero return (random walk)", "mean_return": "Mean return",
    "buy_and_hold": "Always long", "lag_one_linear": "Lag-1 OLS", "auto_arima": "Auto-ARIMA",
    "holt_winters": "Holt--Winters", "prophet": "Prophet", "linear": "OLS (all features)",
    "ridge": "Ridge", "lasso": "Lasso", "elastic_net": "Elastic net",
    "bayesian_ridge": "Bayesian ridge", "huber": "Huber", "knn": "$k$-NN", "svr": "SVR",
    "decision_tree": "Decision tree", "random_forest": "Random forest",
    "extra_trees": "Extra trees", "adaboost": "AdaBoost", "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "mlp": "MLP",
    "rnn": "RNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM", "cnn1d": "1-D CNN",
    "cnn_lstm": "CNN--LSTM", "lstm_attention": "LSTM + attention", "transformer": "Transformer",
    "tcn": "TCN",
}
FAMILY_ORDER = ["Baseline", "Statistical", "Linear", "Kernel / instance", "Tree ensemble",
                "Gradient boosting", "Deep learning"]


def fmt_r2(v: float) -> str:
    return f"{v * 100:+.2f}" if abs(v) < 1 else f"{v * 100:+.0f}"


def fmt_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def main_table() -> None:
    t1 = pd.read_csv(PAPER / "table_h1.csv", index_col=0)
    t7 = pd.read_csv(PAPER / "table_h7.csv", index_col=0)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Out-of-sample forecast accuracy, 13 walk-forward folds (Sep.~2019--Sep.~2026). "
        r"$R^2_{OS}$ is relative to the zero-return forecast (\%). DM is the Diebold--Mariano "
        r"statistic against the zero forecast (positive = less accurate); $p_{\mathrm{Holm}}$ is "
        r"Holm-adjusted over 32 models. DA is directional accuracy (\%). Bold $R^2_{OS}$ marks the "
        r"only positive values; none is significant.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{4}{c}{1-day horizon} & \multicolumn{3}{c}{7-day horizon} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}",
        r"Family & Model & $R^2_{OS}$ & DA & DM & $p_{\mathrm{Holm}}$ & $R^2_{OS}$ & DM & "
        r"$p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for fam in FAMILY_ORDER:
        members = t1[t1["family"] == fam].sort_values("r2_os", ascending=False).index
        for i, m in enumerate(members):
            a, b = t1.loc[m], t7.loc[m]
            r2a = fmt_r2(a.r2_os) if m != "naive_zero" else "0"
            r2b = fmt_r2(b.r2_os) if m != "naive_zero" else "0"
            if m != "naive_zero" and a.r2_os > 0:
                r2a = rf"\textbf{{{r2a}}}"
            if m != "naive_zero" and b.r2_os > 0:
                r2b = rf"\textbf{{{r2b}}}"
            da = "--" if np.isnan(a.dir_acc) else f"{a.dir_acc * 100:.1f}"
            dma = "--" if np.isnan(a.dm_stat) else f"{a.dm_stat:.2f}"
            dmb = "--" if np.isnan(b.dm_stat) else f"{b.dm_stat:.2f}"
            label = fam.replace(" / instance", "") if i == 0 else ""
            lines.append(f"{label} & {NAMES[m]} & {r2a} & {da} & {dma} & {fmt_p(a.dm_p_holm)} & "
                         f"{r2b} & {dmb} & {fmt_p(b.dm_p_holm)} \\\\")
        lines.append(r"\addlinespace[2pt]")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table*}"]
    (OUT / "main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def backtest_table() -> None:
    bt = pd.read_csv(PAPER / "backtest_h1.csv", index_col=0)
    bh = bt.loc["Buy & hold (benchmark)"]
    active = bt[(bt["turnover"] > 0)]
    top = active.head(6)
    ls = bt[bt["strategy"] == "long_short"]
    lf = bt[bt["strategy"] == "long_flat"]

    def row(name: str, r: pd.Series) -> str:
        return (f"{name} & {r.sharpe:.2f} & {r.cagr * 100:.1f} & {r.max_dd * 100:.1f} & "
                f"{r.turnover:.2f} \\\\")

    def label(idx: str) -> str:
        model, strat = idx.rsplit(" (", 1)
        return f"{NAMES[model]} ({'L/F' if strat.startswith('long_flat') else 'L/S'})"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cost-aware backtests of 1-day forecasts (0.10\% fee + 0.05\% slippage per "
        r"unit of position change). Top six trading strategies by Sharpe ratio and medians over "
        r"all 60 model-driven strategies. L/F = long/flat, L/S = long/short, "
        r"DD = maximum drawdown, Turn. = mean daily turnover.}",
        r"\label{tab:backtest}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Strategy & Sharpe & CAGR\% & DD\% & Turn. \\",
        r"\midrule",
        row("Buy \\& hold", bh),
        r"\midrule",
        *[row(label(i), r) for i, r in top.iterrows()],
        r"\midrule",
        f"Median L/S (n={len(ls)}) & {ls.sharpe.median():.2f} & "
        f"{ls.cagr.median() * 100:.1f} & {ls.max_dd.median() * 100:.1f} & "
        f"{ls.turnover.median():.2f} \\\\",
        f"Median L/F (n={len(lf)}) & {lf.sharpe.median():.2f} & "
        f"{lf.cagr.median() * 100:.1f} & {lf.max_dd.median() * 100:.1f} & "
        f"{lf.turnover.median():.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    (OUT / "backtest.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main_table()
    backtest_table()
    print("tables written")
