"""Binance USDT-margined futures data fetchers (public endpoints, no key)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import (
    DATA_START,
    EXCLUDED_BASES,
    EXCLUDED_PREFIXES,
    PROCESSED_DIR,
    UNIVERSE_SIZE,
)

FAPI = "https://fapi.binance.com"
KLINE_LIMIT = 1500


def _to_ms(ts: str | datetime) -> int:
    if isinstance(ts, str):
        ts = pd.Timestamp(ts, tz="UTC")
    if isinstance(ts, pd.Timestamp):
        return int(ts.timestamp() * 1000)
    return int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)


class APIClientError(Exception):
    """Raised on 4xx (excluding 429); not retried."""


def _get(path: str, params: dict | None = None, retries: int = 5, backoff: float = 1.0):
    url = f"{FAPI}{path}"
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params or {}, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):  # rate-limit / IP banned: backoff
                time.sleep(backoff * (2 ** i))
                continue
            if 400 <= r.status_code < 500:
                raise APIClientError(f"{r.status_code} {r.reason} for {url} params={params}")
            r.raise_for_status()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"GET {url} failed after {retries} retries: {last_err}")


def _first_kline_date(symbol: str) -> pd.Timestamp | None:
    """Return the timestamp of the first daily kline ever, or None."""
    try:
        data = _get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": "1d", "startTime": 0, "limit": 1},
        )
    except Exception:
        return None
    if not data:
        return None
    return pd.to_datetime(data[0][0], unit="ms", utc=True).tz_convert(None).normalize()


def get_universe(refresh: bool = False, data_start: str | None = None,
                 listing_grace_days: int | None = None) -> list[str]:
    """Top USDT perps by 30d quote volume whose first daily kline is on or
    before `data_start + grace_days`. Filters out stablecoins, wrapped tokens
    and 1000-prefix tokens.
    """
    from src.config import DATA_START, LISTING_GRACE_DAYS
    if listing_grace_days is None:
        listing_grace_days = LISTING_GRACE_DAYS
    cache = PROCESSED_DIR / "universe.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())["symbols"]

    cutoff = pd.Timestamp(data_start or DATA_START) + pd.Timedelta(days=listing_grace_days)

    info = _get("/fapi/v1/exchangeInfo")
    candidates: list[str] = []
    for s in info["symbols"]:
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING":
            continue
        base = s["baseAsset"]
        sym = s["symbol"]
        if base in EXCLUDED_BASES:
            continue
        if any(sym.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        candidates.append(sym)

    tickers = {t["symbol"]: t for t in _get("/fapi/v1/ticker/24hr")}

    by_volume: list[tuple[str, float]] = []
    for sym in candidates:
        t = tickers.get(sym)
        if t is None:
            continue
        try:
            qv = float(t["quoteVolume"])
        except (KeyError, ValueError):
            continue
        by_volume.append((sym, qv))

    by_volume.sort(key=lambda x: x[1], reverse=True)

    chosen: list[str] = []
    for sym, _ in by_volume:
        if len(chosen) >= UNIVERSE_SIZE:
            break
        first = _first_kline_date(sym)
        if first is None or first > cutoff:
            continue
        chosen.append(sym)
        time.sleep(0.05)

    cache.write_text(
        json.dumps(
            {
                "symbols": chosen,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "data_start": data_start or DATA_START,
                "screened_count": len(by_volume),
                "kept_count": len(chosen),
            },
            indent=2,
        )
    )
    return chosen


def fetch_klines(symbol: str, start: str = DATA_START, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(datetime.now(timezone.utc).timestamp() * 1000)

    rows: list[list] = []
    cur = start_ms
    while cur < end_ms:
        data = _get(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cur,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            },
        )
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        if len(data) < KLINE_LIMIT:
            break
        cur = last_open + 1
        time.sleep(0.05)

    if not rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None).dt.normalize()
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume", "quote_volume"]]


def fetch_all_prices(symbols: list[str], start: str = DATA_START, end: str | None = None) -> pd.DataFrame:
    out = {}
    volumes = {}
    for sym in symbols:
        df = fetch_klines(sym, start=start, end=end)
        if df.empty:
            print(f"  [skip] {sym}: empty")
            continue
        out[sym] = df.set_index("date")["close"]
        volumes[sym] = df.set_index("date")["quote_volume"]
        print(f"  [ok] {sym}: {len(df)} rows {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    prices = pd.DataFrame(out).sort_index()
    quote_vol = pd.DataFrame(volumes).sort_index()
    return prices, quote_vol


def clean_prices(prices: pd.DataFrame, min_coverage: float = 0.80) -> pd.DataFrame:
    """Forward fill (max 3 days) and drop columns with insufficient coverage."""
    full_idx = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    p = prices.reindex(full_idx)
    p.index.name = "date"
    p = p.ffill(limit=3)
    coverage = p.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = [c for c in p.columns if c not in keep]
    if dropped:
        print(f"  [drop] insufficient coverage: {dropped}")
    return p[keep]


def save_prices(prices: pd.DataFrame, name: str = "prices"):
    path = PROCESSED_DIR / f"{name}.parquet"
    prices.to_parquet(path)
    return path
