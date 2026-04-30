"""Alternative data: open interest and Fear & Greed Index."""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

from src.config import DATA_START, PROCESSED_DIR
from src.data.binance import APIClientError, _get, _to_ms

OI_LIMIT = 500


def fetch_open_interest(symbol: str, start: str = DATA_START, end: str | None = None) -> pd.DataFrame:
    """Open interest history at 1d granularity. Binance returns at most ~30 days
    per call. We walk *backwards* from `end` in 29-day chunks and stop the
    first time the API returns 400 / empty data, since older history is
    not retained.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None)

    rows: list[dict] = []
    chunk = pd.Timedelta(days=29)
    cur_end = end_dt
    while cur_end > start_dt:
        cur_start = max(cur_end - chunk, start_dt)
        try:
            data = _get(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol,
                    "period": "1d",
                    "limit": OI_LIMIT,
                    "startTime": _to_ms(cur_start),
                    "endTime": _to_ms(cur_end),
                },
            )
        except APIClientError as e:
            # 400 from this endpoint typically means out-of-range historical
            # data. Stop walking further back.
            data = []
            break_after = True
        except Exception as e:
            print(f"  [warn] OI {symbol} chunk {cur_start.date()}-{cur_end.date()}: {e}")
            data = []
            break_after = False
        else:
            break_after = False
        if data:
            rows.extend(data)
        if break_after:
            break
        if not data:
            # endpoint returned empty for this range — older windows won't have data either
            break
        cur_end = cur_start - pd.Timedelta(days=1)
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame(columns=["date", "open_interest", "open_interest_value"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    df["open_interest_value"] = pd.to_numeric(df["sumOpenInterestValue"])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[["date", "open_interest", "open_interest_value"]]


def fetch_all_open_interest(symbols: list[str], start: str = DATA_START, end: str | None = None):
    oi = {}
    oi_value = {}
    for sym in symbols:
        df = fetch_open_interest(sym, start=start, end=end)
        if df.empty:
            print(f"  [skip] OI {sym}: empty")
            continue
        oi[sym] = df.set_index("date")["open_interest"]
        oi_value[sym] = df.set_index("date")["open_interest_value"]
        print(f"  [ok] OI {sym}: {len(df)} rows")
    return pd.DataFrame(oi).sort_index(), pd.DataFrame(oi_value).sort_index()


def fetch_fear_greed() -> pd.DataFrame:
    """Daily Fear & Greed Index from alternative.me (Bitcoin sentiment proxy)."""
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(data)
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    df["date"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(None).dt.normalize()
    df["fear_greed"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "fear_greed"]).sort_values("date").reset_index(drop=True)
    return df[["date", "fear_greed"]]


def save_oi(oi: pd.DataFrame, name: str = "open_interest"):
    path = PROCESSED_DIR / f"{name}.parquet"
    oi.to_parquet(path)
    return path


def save_fg(fg: pd.DataFrame, name: str = "fear_greed"):
    path = PROCESSED_DIR / f"{name}.parquet"
    fg.to_parquet(path)
    return path
