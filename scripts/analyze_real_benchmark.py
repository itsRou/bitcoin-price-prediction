"""Turn the saved out-of-fold predictions into the paper's tables, tests, backtests and figures.

Reads reports/paper/oof_h{h}.csv (written by `run_real_benchmark.py`) and writes:
  reports/paper/table_h{h}.csv    per-model forecast metrics + significance tests
  reports/paper/backtest_h1.csv   cost-aware long/short and long/flat backtests (h=1)
  reports/paper/fold_da_h1.csv    per-fold directional accuracy (stability)
  reports/paper/summary.json      headline numbers quoted in the paper
  reports/paper/fig_*.pdf/png     figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from btcpred.backtest.engine import run_backtest
from btcpred.validation.metrics import (
    cagr,
    diebold_mariano_test,
    max_drawdown,
    sharpe_ratio,
)

sys.path.insert(0, str(Path(__file__).parent))
from make_paper_tables import NAMES as _TEX_NAMES  # noqa: E402

NAMES = {k: v.replace("--", "–") for k, v in _TEX_NAMES.items()}

PAPER = Path("reports/paper")
FEE, SLIPPAGE = 0.001, 0.0005
BASELINES = {"naive_zero", "mean_return", "buy_and_hold"}

FAMILY = {
    "naive_zero": "Baseline", "mean_return": "Baseline", "buy_and_hold": "Baseline",
    "lag_one_linear": "Linear", "auto_arima": "Statistical", "holt_winters": "Statistical",
    "prophet": "Statistical", "linear": "Linear", "ridge": "Linear", "lasso": "Linear",
    "elastic_net": "Linear", "bayesian_ridge": "Linear", "huber": "Linear",
    "knn": "Kernel / instance", "svr": "Kernel / instance", "decision_tree": "Tree ensemble",
    "random_forest": "Tree ensemble", "extra_trees": "Tree ensemble",
    "gradient_boosting": "Gradient boosting", "adaboost": "Tree ensemble",
    "xgboost": "Gradient boosting", "lightgbm": "Gradient boosting",
    "catboost": "Gradient boosting",
}
for _d in ("mlp", "rnn", "lstm", "gru", "bilstm", "cnn1d", "cnn_lstm", "lstm_attention",
           "transformer", "tcn"):
    FAMILY[_d] = "Deep learning"

# Categorical slots in fixed order (reference palette, light mode); baselines in neutral ink.
FAMILY_COLOR = {
    "Baseline": "#8a8985", "Statistical": "#2a78d6", "Linear": "#eb6834",
    "Kernel / instance": "#1baf7a", "Tree ensemble": "#eda100",
    "Gradient boosting": "#e87ba4", "Deep learning": "#4a3aa7",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200,
})


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values (NaNs passed through)."""
    p = pvals.dropna().sort_values()
    m = len(p)
    adj = np.maximum.accumulate([min(1.0, (m - i) * v) for i, v in enumerate(p.to_numpy())])
    return pd.Series(adj, index=p.index).reindex(pvals.index)


def deflated_sharpe(sr_best: float, sr_all: np.ndarray, returns: pd.Series) -> float:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio, per-period (non-annualized) SRs."""
    n_trials = len(sr_all)
    var_sr = np.var(sr_all, ddof=1)
    gamma = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n_trials)
                             + gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    t = len(returns)
    skew, kurt = stats.skew(returns), stats.kurtosis(returns, fisher=False)
    denom = np.sqrt(1 - skew * sr_best + (kurt - 1) / 4 * sr_best**2)
    return float(stats.norm.cdf((sr_best - sr0) * np.sqrt(t - 1) / denom))


def forecast_table(oof: pd.DataFrame, h: int) -> pd.DataFrame:
    scored = oof[oof["fold"].notna()]
    y = scored["y_true"].to_numpy()
    models = [c for c in scored.columns if c not in ("y_true", "fold")]
    sse_zero = np.sum(y**2)
    e_zero = y - 0.0
    rows = {}
    for m in models:
        p = scored[m].to_numpy()
        ok = ~np.isnan(p)
        if ok.sum() < 0.95 * len(p):
            continue
        yy, pp = y[ok], p[ok]
        err = yy - pp
        nz = (pp != 0) & (yy != 0)
        hits = int(np.sum(np.sign(pp[nz]) == np.sign(yy[nz])))
        n_dir = int(nz.sum())
        da = hits / n_dir if n_dir else np.nan
        # Binomial test is only valid for non-overlapping (h=1) targets.
        binom_p = (stats.binomtest(hits, n_dir, 0.5, alternative="greater").pvalue
                   if (h == 1 and n_dir) else np.nan)
        dm, dm_p = (np.nan, np.nan) if m == "naive_zero" else \
            diebold_mariano_test(err, e_zero[ok], h=h, power=2)
        rows[m] = {
            "family": FAMILY.get(m, "Other"),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mae": float(np.mean(np.abs(err))),
            "r2_os": float(1 - np.sum(err**2) / np.sum(yy**2)) if ok.all() else
            float(1 - np.sum(err**2) / np.sum(e_zero[ok] ** 2)),
            "dir_acc": da,
            "n_dir": n_dir,
            "binom_p": binom_p,
            "dm_stat": dm,
            "dm_p": dm_p,
        }
    tab = pd.DataFrame(rows).T
    for c in ("rmse", "mae", "r2_os", "dir_acc", "binom_p", "dm_stat", "dm_p"):
        tab[c] = tab[c].astype(float)
    tab["n_dir"] = tab["n_dir"].astype(int)
    tab["dm_p_holm"] = holm(tab["dm_p"])
    tab["binom_p_holm"] = holm(tab["binom_p"]) if h == 1 else np.nan
    _ = sse_zero
    return tab.sort_values("rmse")


def backtests(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    scored = oof[oof["fold"].notna()]
    simple = np.expm1(scored["y_true"])
    models = [c for c in scored.columns if c not in ("y_true", "fold")]
    rows, curves = {}, {}
    bh = run_backtest(simple, pd.Series(1.0, index=simple.index), FEE, SLIPPAGE)["net_return"]
    curves["Buy & hold"] = bh
    rows["Buy & hold (benchmark)"] = {
        "strategy": "long", "sharpe": sharpe_ratio(bh), "cagr": cagr(bh),
        "max_dd": max_drawdown(bh), "turnover": 0.0,
    }
    for m in models:
        if m in ("naive_zero", "buy_and_hold", "mean_return"):
            continue
        p = scored[m]
        if p.isna().mean() > 0.05:
            continue
        sig = np.sign(p.fillna(0.0))
        for label, pos in (("long_short", sig), ("long_flat", sig.clip(lower=0))):
            bt = run_backtest(simple, pos, FEE, SLIPPAGE)
            r = bt["net_return"]
            rows[f"{m} ({label})"] = {
                "strategy": label, "sharpe": sharpe_ratio(r), "cagr": cagr(r),
                "max_dd": max_drawdown(r), "turnover": float(pos.diff().abs().mean()),
            }
            curves[f"{m} ({label})"] = r
    tab = pd.DataFrame(rows).T
    for c in ("sharpe", "cagr", "max_dd", "turnover"):
        tab[c] = tab[c].astype(float)
    return tab.sort_values("sharpe", ascending=False), curves


def fig_r2(tabs: dict[int, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.6), sharey=False)
    for ax, (h, tab) in zip(axes, tabs.items(), strict=True):
        t = tab.drop(index="naive_zero").sort_values("r2_os")
        t = t[t["r2_os"] > -0.5]  # keep the axis readable; dropped models listed in caption
        colors = [FAMILY_COLOR[f] for f in t["family"]]
        ax.barh([NAMES[m] for m in t.index], t["r2_os"] * 100, color=colors, height=0.7)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_title(f"{h}-day horizon", color=INK, fontsize=9)
        ax.set_xlabel("Out-of-sample $R^2$ vs. zero forecast (%)")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, FAMILY_COLOR.keys(), loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / f"fig_r2_os.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_da(tab: pd.DataFrame, n_obs: int) -> None:
    t = tab.drop(index=[m for m in ("naive_zero",) if m in tab.index]).sort_values("dir_acc")
    fig, ax = plt.subplots(figsize=(3.4, 4.4))
    ci = 1.96 * np.sqrt(0.25 / t["n_dir"])
    ax.axvspan(50 - ci.mean() * 100, 50 + ci.mean() * 100, color=GRID, alpha=0.8, lw=0,
               label="95% band under a coin flip")
    ax.scatter(t["dir_acc"] * 100, [NAMES[m] for m in t.index], s=14, zorder=3,
               c=[FAMILY_COLOR[f] for f in t["family"]])
    ax.axvline(50, color=INK, linewidth=0.8)
    ax.set_xlabel("Directional accuracy, 1-day horizon (%)")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.5)
    _ = n_obs
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / f"fig_dir_acc.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_equity(bt: pd.DataFrame, curves: dict[str, pd.Series]) -> None:
    # Skip strategies that never trade (e.g. lasso shrunk to a constant positive forecast):
    # their curves are identical to buy-and-hold.
    active = bt[(bt["turnover"] > 0) & ~bt.index.str.startswith("Buy & hold")]
    top = list(active.index[:3])
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    eq = (1 + curves["Buy & hold"]).cumprod()
    ax.plot(eq.index, eq, color=INK2, linewidth=1.6, label="Buy & hold")
    for name, color in zip(top, ("#2a78d6", "#eb6834", "#1baf7a"), strict=False):
        e = (1 + curves[name]).cumprod()
        model, strat = name.rsplit(" (", 1)
        nice = f"{NAMES[model]} ({strat.rstrip(')').replace('_', '/')})"
        ax.plot(e.index, e, color=color, linewidth=1.2, label=nice)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of \\$1 (log scale, net of costs)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / f"fig_equity.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig_price(oof: pd.DataFrame, close: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    ax.plot(close.index, close, color="#2a78d6", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("BTC-USD close (log)")
    scored = oof[oof["fold"].notna()]
    first = scored.index[0]
    ax.axvspan(close.index[0], first, color=GRID, alpha=0.7, lw=0)
    ax.text(close.index[0] + pd.Timedelta(days=60), close.max() * 0.6,
            "initial training\nonly", fontsize=7, color=INK2)
    for _, g in scored.groupby("fold"):
        ax.axvline(g.index[0], color=INK2, linewidth=0.4, linestyle=":")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(PAPER / f"fig_price_folds.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tabs, summary = {}, {}
    for h in (1, 7):
        path = PAPER / f"oof_h{h}.csv"
        if not path.exists():
            continue
        oof = pd.read_csv(path, index_col=0)
        oof.index = pd.to_datetime(oof.index, utc=True)
        tab = forecast_table(oof, h)
        tab.to_csv(PAPER / f"table_h{h}.csv")
        tabs[h] = tab
        scored = oof[oof["fold"].notna()]
        summary[f"h{h}"] = {
            "n_models": int(len(tab)),
            "n_test_obs": int(len(scored)),
            "test_start": str(scored.index[0].date()),
            "test_end": str(scored.index[-1].date()),
            "n_folds": int(scored["fold"].nunique()),
            "best_rmse_model": tab.index[0],
            "n_positive_r2": int((tab["r2_os"] > 0).sum()),
            "n_dm_better_raw": int(((tab["dm_stat"] < 0) & (tab["dm_p"] < 0.05)).sum()),
            "n_dm_better_holm": int(((tab["dm_stat"] < 0) & (tab["dm_p_holm"] < 0.05)).sum()),
            "n_dm_worse_holm": int(((tab["dm_stat"] > 0) & (tab["dm_p_holm"] < 0.05)).sum()),
            "n_dir_acc_above_50": int((tab["dir_acc"] > 0.5).sum()),
            "max_dir_acc": float(tab["dir_acc"].max()),
            "max_dir_acc_model": str(tab["dir_acc"].idxmax()),
        }
        if h == 1:
            summary["h1"]["n_binom_raw"] = int((tab["binom_p"] < 0.05).sum())
            summary["h1"]["n_binom_holm"] = int((tab["binom_p_holm"] < 0.05).sum())
            bt, curves = backtests(oof)
            bt.to_csv(PAPER / "backtest_h1.csv")
            strat = bt.drop(index="Buy & hold (benchmark)")
            per_period = {k: v for k, v in curves.items() if k != "Buy & hold"}
            sr_pp = np.array([r.mean() / r.std() for r in per_period.values()])
            best = strat.index[0]
            r_best = per_period[best]
            summary["h1"]["backtest"] = {
                "buy_hold_sharpe": float(bt.loc["Buy & hold (benchmark)", "sharpe"]),
                "buy_hold_cagr": float(bt.loc["Buy & hold (benchmark)", "cagr"]),
                "buy_hold_max_dd": float(bt.loc["Buy & hold (benchmark)", "max_dd"]),
                "n_strategies": int(len(strat)),
                "n_beat_bh_sharpe": int(
                    (strat["sharpe"] > bt.loc["Buy & hold (benchmark)", "sharpe"]).sum()),
                "best_strategy": best,
                "best_sharpe": float(strat.iloc[0]["sharpe"]),
                "best_deflated_sharpe_prob": deflated_sharpe(
                    r_best.mean() / r_best.std(), sr_pp, r_best),
            }
            fold_models = [m for m in tab.index if m != "naive_zero"]
            fold_da = scored.groupby("fold").apply(
                lambda g, ms=fold_models: pd.Series(
                    {m: np.mean(np.sign(g[m]) == np.sign(g["y_true"])) for m in ms}))
            fold_da.to_csv(PAPER / "fold_da_h1.csv")
            fig_da(tab, len(scored))
            fig_equity(bt, curves)
            snap = sorted(Path("data/snapshots").glob("*.csv"))[0]
            close = pd.read_csv(snap, index_col=0)["close"]
            close.index = pd.to_datetime(close.index, utc=True)
            fig_price(oof, close)
    if tabs:
        fig_r2(tabs)
    (PAPER / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
