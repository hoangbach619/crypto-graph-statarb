#!/usr/bin/env python
"""Run v2 (graph-ML) on the precomputed feature panel."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import apply_backtest
from src.backtest.metrics import summary_metrics
from src.config import OOS_START, PROCESSED_DIR, SEED, TBL_DIR
from src.models.graph_ml import run_strategy

np.random.seed(SEED)


def main():
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet").sort_index()
    prices.index = pd.to_datetime(prices.index)
    returns = prices.pct_change()

    funding_path = PROCESSED_DIR / "funding.parquet"
    funding = pd.read_parquet(funding_path).sort_index() if funding_path.exists() else None
    if funding is not None:
        funding.index = pd.to_datetime(funding.index)

    panel = pd.read_parquet(PROCESSED_DIR / "features_v2.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    print("running v2 graph-ML strategy ...")
    weights, diag, imp = run_strategy(panel, list(prices.columns))

    weights.to_parquet(TBL_DIR / "v2_weights.parquet")
    if not diag.empty:
        diag.to_csv(TBL_DIR / "v2_diagnostics.csv")
    if not imp.empty:
        imp.to_csv(TBL_DIR / "v2_importances.csv")

    bt = apply_backtest(weights, returns, funding=funding)
    bt.to_parquet(TBL_DIR / "v2_daily.parquet")
    bt.to_csv(TBL_DIR / "v2_daily.csv")

    oos_mask = bt.index >= pd.Timestamp(OOS_START)
    sub = bt.loc[oos_mask]
    summary = summary_metrics("v2_graph_ml", sub["net_ret"], gross=sub["gross_ret"], diag=diag)
    pd.DataFrame([summary]).to_csv(TBL_DIR / "v2_results.csv", index=False)

    print("\nv2 summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
