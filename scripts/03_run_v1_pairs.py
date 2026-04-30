#!/usr/bin/env python
"""Run v1 (Engle-Granger pairs) over the OOS window and save results."""
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
from src.models.baseline_pairs import run_pairs_strategy

np.random.seed(SEED)


def main():
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet").sort_index()
    prices.index = pd.to_datetime(prices.index)
    returns = prices.pct_change()

    funding_path = PROCESSED_DIR / "funding.parquet"
    funding = pd.read_parquet(funding_path).sort_index() if funding_path.exists() else None
    if funding is not None:
        funding.index = pd.to_datetime(funding.index)

    print("running v1 pairs strategy ...")
    weights = run_pairs_strategy(prices)
    weights.to_parquet(TBL_DIR.parent / "tables" / "v1_weights.parquet")

    bt = apply_backtest(weights, returns, funding=funding)
    bt.to_parquet(TBL_DIR / "v1_daily.parquet")
    bt.to_csv(TBL_DIR / "v1_daily.csv")

    oos_mask = bt.index >= pd.Timestamp(OOS_START)
    sub = bt.loc[oos_mask]
    summary = summary_metrics("v1_pairs", sub["net_ret"], gross=sub["gross_ret"])
    pd.DataFrame([summary]).to_csv(TBL_DIR / "v1_results.csv", index=False)

    print("\nv1 summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")


if __name__ == "__main__":
    main()
