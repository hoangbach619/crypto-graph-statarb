"""Rolling correlation matrices on returns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> dict:
    """Return dict of {date -> correlation matrix} for each date with full window."""
    out = {}
    if len(returns) < window:
        return out
    arr = returns.values
    n = len(returns)
    for t in range(window, n):
        sl = arr[t - window:t, :]
        mask = ~np.isnan(sl).any(axis=0)
        if mask.sum() < 3:
            continue
        sub = sl[:, mask]
        c = np.corrcoef(sub, rowvar=False)
        idx = returns.columns[mask]
        out[returns.index[t]] = pd.DataFrame(c, index=idx, columns=idx)
    return out


def correlation_at(returns: pd.DataFrame, asof: pd.Timestamp, window: int = 60) -> pd.DataFrame:
    """One correlation matrix as of a given date, using prior `window` days."""
    end = returns.index.searchsorted(asof, side="right")
    start = end - window
    if start < 0:
        return pd.DataFrame()
    sl = returns.iloc[start:end].dropna(axis=1, thresh=int(window * 0.9))
    if sl.shape[1] < 3:
        return pd.DataFrame()
    return sl.corr()
