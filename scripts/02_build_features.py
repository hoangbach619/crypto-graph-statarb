#!/usr/bin/env python
"""Precompute v2 and v3 cross-sectional feature panels at every weekly rebalance.

Outputs:
  data/processed/features_v2.parquet
  data/processed/features_v3.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_START, PROCESSED_DIR, REBALANCE_FREQ, SEED
from src.models.altdata_ml import build_panel_v3
from src.models.graph_ml import build_panel as build_panel_v2

np.random.seed(SEED)


def main():
    prices = pd.read_parquet(PROCESSED_DIR / "prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    returns = prices.pct_change()

    funding = pd.read_parquet(PROCESSED_DIR / "funding.parquet") if (PROCESSED_DIR / "funding.parquet").exists() else pd.DataFrame()
    oi = pd.read_parquet(PROCESSED_DIR / "open_interest.parquet") if (PROCESSED_DIR / "open_interest.parquet").exists() else pd.DataFrame()
    fg = pd.read_parquet(PROCESSED_DIR / "fear_greed.parquet") if (PROCESSED_DIR / "fear_greed.parquet").exists() else pd.DataFrame()

    if not funding.empty:
        funding.index = pd.to_datetime(funding.index)
        funding = funding.sort_index()
    if not oi.empty:
        oi.index = pd.to_datetime(oi.index)
        oi = oi.sort_index()
    if not fg.empty:
        fg.index = pd.to_datetime(fg.index)
        fg = fg.sort_index()

    rebals = pd.date_range(prices.index.min(), prices.index.max(), freq=REBALANCE_FREQ)
    rebals = [d for d in rebals if d in prices.index or d <= prices.index.max()]
    print(f"rebalance dates: {len(rebals)} from {rebals[0].date()} to {rebals[-1].date()}")

    print("[v2] building feature panel...")
    panel_v2 = build_panel_v2(prices, returns, rebals)
    print(f"  v2 panel rows={len(panel_v2)} unique dates={panel_v2['date'].nunique() if not panel_v2.empty else 0}")
    panel_v2.to_parquet(PROCESSED_DIR / "features_v2.parquet")

    print("[v3] building feature panel (with alt-data)...")
    panel_v3 = build_panel_v3(prices, returns, funding, oi, fg, rebals)
    print(f"  v3 panel rows={len(panel_v3)} unique dates={panel_v3['date'].nunique() if not panel_v3.empty else 0}")
    panel_v3.to_parquet(PROCESSED_DIR / "features_v3.parquet")

    if not panel_v3.empty:
        coverage = (panel_v3[["funding_rate_z30d", "oi_zscore_30d", "fg_x_vol"]] != 0).mean().to_dict()
        print(f"  alt-data non-zero coverage: {coverage}")

    print("Done.")


if __name__ == "__main__":
    main()
