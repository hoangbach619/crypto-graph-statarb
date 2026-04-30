#!/usr/bin/env python
"""Compare v1, v2, v3 results, generate plots and a comparison table."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIG_DIR, OOS_START, PROCESSED_DIR, SEED, TBL_DIR
from src.viz.plots import (
    drawdown,
    equity_curve,
    feature_importance,
    monthly_returns_heatmap,
    rank_ic_distribution,
    regime_breakdown,
    rolling_sharpe,
)

np.random.seed(SEED)


def _load_daily(name: str) -> pd.DataFrame:
    p = TBL_DIR / f"{name}_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _load_diag(name: str) -> pd.DataFrame:
    p = TBL_DIR / f"{name}_diagnostics.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date")
    return d


def _load_imp(name: str) -> pd.DataFrame:
    p = TBL_DIR / f"{name}_importances.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date")
    return d


def main():
    summary_rows = []
    for v in ["v1", "v2", "v3"]:
        path = TBL_DIR / f"{v}_results.csv"
        if path.exists():
            summary_rows.append(pd.read_csv(path).iloc[0].to_dict())
    if not summary_rows:
        print("no result files found, run scripts 03-05 first")
        return
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TBL_DIR / "comparison.csv", index=False)
    print("\n=== HEADLINE COMPARISON ===")
    cols = ["strategy", "ann_return_net", "sharpe_net", "max_drawdown",
            "calmar", "mean_ic", "ic_t", "topbot_spread", "topbot_t", "n_days"]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    daily_v1 = _load_daily("v1")
    daily_v2 = _load_daily("v2")
    daily_v3 = _load_daily("v3")
    rets = {}
    if not daily_v1.empty:
        rets["v1 pairs"] = daily_v1.loc[daily_v1.index >= OOS_START, "net_ret"]
    if not daily_v2.empty:
        rets["v2 graph-ML"] = daily_v2.loc[daily_v2.index >= OOS_START, "net_ret"]
    if not daily_v3.empty:
        rets["v3 + alt-data"] = daily_v3.loc[daily_v3.index >= OOS_START, "net_ret"]

    print("\nplots ...")
    print(" ", equity_curve(rets))
    print(" ", drawdown(rets))
    print(" ", rolling_sharpe(rets))
    print(" ", rank_ic_distribution({"v2": _load_diag("v2"), "v3": _load_diag("v3")}))
    print(" ", feature_importance(_load_imp("v2"), "feature_importance_v2.png"))
    print(" ", feature_importance(_load_imp("v3"), "feature_importance_v3.png"))

    fg_path = PROCESSED_DIR / "fear_greed.parquet"
    fg = pd.read_parquet(fg_path)["fear_greed"] if fg_path.exists() else None
    print(" ", regime_breakdown(rets, fg=fg))

    if rets:
        # heatmap of the strongest available strategy
        last = list(rets.keys())[-1]
        print(" ", monthly_returns_heatmap(rets[last]))


if __name__ == "__main__":
    main()
