"""Compare full-feature vs. stationary-feature (price-level features removed) results at h=1.

Writes paper/tables/robustness.tex and reports/paper_stationary/summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_real_benchmark import forecast_table  # noqa: E402
from make_paper_tables import NAMES, fmt_p, fmt_r2  # noqa: E402

FULL = Path("reports/paper/oof_h1.csv")
STAT = Path("reports/paper_stationary/oof_h1.csv")


def load(path: Path, models: list[str] | None = None) -> pd.DataFrame:
    oof = pd.read_csv(path, index_col=0)
    oof.index = pd.to_datetime(oof.index, utc=True)
    if models is not None:
        oof = oof[["y_true", "fold", *models]]
    return oof


def main() -> None:
    stat = load(STAT)
    models = [c for c in stat.columns if c not in ("y_true", "fold")]
    t_stat = forecast_table(stat, 1)
    # Re-test the full-feature run on the same model subset so both Holm corrections
    # cover the same number of comparisons.
    t_full = forecast_table(load(FULL, models), 1)

    rows = []
    for m in [m for m in t_full.sort_values("r2_os", ascending=False).index if m != "naive_zero"]:
        a, b = t_full.loc[m], t_stat.loc[m]
        rows.append(
            f"{NAMES[m]} & {fmt_r2(a.r2_os)} & {fmt_p(a.dm_p_holm)} & "
            f"{fmt_r2(b.r2_os)} & {fmt_p(b.dm_p_holm)} & {b.dir_acc * 100:.1f} \\\\"
        )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robustness: 1-day results with all 66 features vs.\ 53 stationary features "
        r"(13 price-level features removed). $R^2_{OS}$ in \%; $p_{\mathrm{Holm}}$ from the DM "
        r"test against the zero forecast, adjusted over the 16 models shown; DA in \%.}",
        r"\label{tab:robust}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{All features} & \multicolumn{3}{c}{Stationary only} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Model & $R^2_{OS}$ & $p_{\mathrm{Holm}}$ & $R^2_{OS}$ & $p_{\mathrm{Holm}}$ & DA \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    Path("paper/tables/robustness.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ts = t_stat.drop(index="naive_zero")
    tf = t_full.drop(index="naive_zero")
    summary = {
        "n_models": len(ts),
        "n_positive_r2": int((ts["r2_os"] > 0).sum()),
        "positive_r2_models": {m: float(v) for m, v in ts["r2_os"][ts["r2_os"] > 0].items()},
        "n_better_holm": int(((ts["dm_stat"] < 0) & (ts["dm_p_holm"] < 0.05)).sum()),
        "n_worse_holm": int(((ts["dm_stat"] > 0) & (ts["dm_p_holm"] < 0.05)).sum()),
        "n_worse_holm_full_same_subset": int(
            ((tf["dm_stat"] > 0) & (tf["dm_p_holm"] < 0.05)).sum()
        ),
        "median_r2_full": float(tf["r2_os"].median()),
        "median_r2_stationary": float(ts["r2_os"].median()),
        "n_improved": int((ts["r2_os"] > tf["r2_os"].reindex(ts.index)).sum()),
        "max_dir_acc": float(ts["dir_acc"].max()),
        "max_dir_acc_model": str(ts["dir_acc"].idxmax()),
        "min_binom_p_raw": float(ts["binom_p"].min()),
        "r2": {m: [float(tf.loc[m, "r2_os"]), float(ts.loc[m, "r2_os"])] for m in ts.index},
    }
    Path("reports/paper_stationary/summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    _ = np


if __name__ == "__main__":
    main()
