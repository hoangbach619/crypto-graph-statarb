"""Funding rate fetcher for Binance USDT perps."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from src.config import DATA_START, PROCESSED_DIR
from src.data.binance import _get, _to_ms

FUNDING_LIMIT = 1000


def fetch_funding_rates(symbol: str, start: str = DATA_START, end: str | None = None) -> pd.DataFrame:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        data = _get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": FUNDING_LIMIT},
        )
        if not data:
            break
        rows.extend(data)
        last_t = data[-1]["fundingTime"]
        if len(data) < FUNDING_LIMIT:
            break
        cur = last_t + 1
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame(columns=["date", "funding_rate"])

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.tz_convert(None)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    df["date"] = df["funding_time"].dt.normalize()
    daily = df.groupby("date")["funding_rate"].sum().reset_index()
    return daily


def fetch_all_funding(symbols: list[str], start: str = DATA_START, end: str | None = None) -> pd.DataFrame:
    out = {}
    for sym in symbols:
        df = fetch_funding_rates(sym, start=start, end=end)
        if df.empty:
            print(f"  [skip] funding {sym}: empty")
            continue
        out[sym] = df.set_index("date")["funding_rate"]
        print(f"  [ok] funding {sym}: {len(df)} daily rows")
    funding = pd.DataFrame(out).sort_index()
    return funding


def save_funding(funding: pd.DataFrame, name: str = "funding"):
    path = PROCESSED_DIR / f"{name}.parquet"
    funding.to_parquet(path)
    return path
