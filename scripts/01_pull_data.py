#!/usr/bin/env python
"""Pull universe, prices, funding rates, open interest and Fear & Greed index.

Idempotent: if data/processed/* already exists and --refresh is not passed,
files are loaded from cache. Run with --refresh to force redownload.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import random

from src.config import DATA_START, PROCESSED_DIR, SEED
from src.data.binance import (
    clean_prices,
    fetch_all_prices,
    get_universe,
    save_prices,
)
from src.data.funding import fetch_all_funding, save_funding
from src.data.altdata import (
    fetch_all_open_interest,
    fetch_fear_greed,
    save_fg,
    save_oi,
)

np.random.seed(SEED)
random.seed(SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="ignore caches")
    args = ap.parse_args()

    print("[1/5] universe")
    syms = get_universe(refresh=args.refresh)
    print(f"  universe ({len(syms)}): {syms}")

    print("\n[2/5] prices")
    prices_path = PROCESSED_DIR / "prices.parquet"
    qv_path = PROCESSED_DIR / "quote_volume.parquet"
    if prices_path.exists() and qv_path.exists() and not args.refresh:
        prices = pd.read_parquet(prices_path)
        print(f"  cached: {prices_path} shape={prices.shape}")
    else:
        prices, qv = fetch_all_prices(syms, start=DATA_START)
        prices = clean_prices(prices)
        save_prices(prices, "prices")
        qv.to_parquet(qv_path)
        print(f"  saved: {prices_path} shape={prices.shape}")

    print("\n[3/5] funding rates")
    funding_path = PROCESSED_DIR / "funding.parquet"
    if funding_path.exists() and not args.refresh:
        funding = pd.read_parquet(funding_path)
        print(f"  cached: {funding_path} shape={funding.shape}")
    else:
        funding = fetch_all_funding(syms, start=DATA_START)
        save_funding(funding)
        print(f"  saved: {funding_path} shape={funding.shape}")

    print("\n[4/5] open interest")
    oi_path = PROCESSED_DIR / "open_interest.parquet"
    if oi_path.exists() and not args.refresh:
        oi = pd.read_parquet(oi_path)
        print(f"  cached: {oi_path} shape={oi.shape}")
    else:
        oi, oi_value = fetch_all_open_interest(syms, start=DATA_START)
        save_oi(oi, "open_interest")
        if not oi_value.empty:
            oi_value.to_parquet(PROCESSED_DIR / "open_interest_value.parquet")
        print(f"  saved: {oi_path} shape={oi.shape}")

    print("\n[5/5] fear & greed")
    fg_path = PROCESSED_DIR / "fear_greed.parquet"
    if fg_path.exists() and not args.refresh:
        fg = pd.read_parquet(fg_path)
        print(f"  cached: {fg_path} shape={fg.shape}")
    else:
        fg = fetch_fear_greed()
        fg = fg.set_index("date")
        save_fg(fg)
        print(f"  saved: {fg_path} shape={fg.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
