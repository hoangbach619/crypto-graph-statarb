"""v1 baseline: Engle-Granger cointegrated pairs trading.

Train window selects pairs by cointegration p-value on log prices, fits a
hedge ratio via OLS, then trades z-scored spreads with z>2 / z<-2 entries
and z=0 exit. Each rebalance recomputes the universe of pairs from a
rolling lookback that ends BEFORE the rebalance date (no look-ahead).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from src.config import (
    OOS_START,
    REBALANCE_FREQ,
    SEED,
)

PAIR_LOOKBACK = 252
SPREAD_WINDOW = 60
Z_ENTRY = 2.0
Z_EXIT = 0.0
TOP_N_PAIRS = 20
COINT_PVAL = 0.05


def _ols_hedge(y: np.ndarray, x: np.ndarray) -> float:
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    return float(res.params[1])


def select_pairs(prices: pd.DataFrame, asof: pd.Timestamp,
                 lookback: int = PAIR_LOOKBACK,
                 top_n: int = TOP_N_PAIRS) -> list[tuple[str, str, float, float]]:
    """Return list of (a, b, hedge_ratio, t_stat) for pairs with p < COINT_PVAL,
    ranked by t-statistic magnitude (more negative = stronger), top_n only.
    """
    end = prices.index.searchsorted(asof, side="right")
    start = max(0, end - lookback)
    win = prices.iloc[start:end].dropna(axis=1)
    if win.shape[0] < lookback // 2 or win.shape[1] < 4:
        return []

    log_p = np.log(win)
    cols = list(log_p.columns)
    np.random.default_rng(SEED)  # determinism

    cands: list[tuple[str, str, float, float]] = []
    for a, b in combinations(cols, 2):
        try:
            t_stat, p_val, _ = coint(log_p[a].values, log_p[b].values, autolag="AIC")
        except Exception:
            continue
        if p_val < COINT_PVAL and np.isfinite(t_stat):
            try:
                hr = _ols_hedge(log_p[a].values, log_p[b].values)
            except Exception:
                continue
            cands.append((a, b, hr, t_stat))

    cands.sort(key=lambda r: r[3])  # most negative t-stat first
    return cands[:top_n]


def pairs_signal_to_weights(prices: pd.DataFrame, pair: tuple[str, str, float, float],
                            asof: pd.Timestamp,
                            spread_window: int = SPREAD_WINDOW) -> dict[str, float]:
    """Return dollar-neutral leg weights for one pair as of `asof`.

    A long-spread position holds +1 unit of leg A and -hedge_ratio units of leg B
    in log-price space. We translate to dollar weights of equal gross size per
    leg: +0.5 / -0.5 of the *pair*, and the pair contributes 1 unit of gross.
    """
    a, b, hr, _ = pair
    end = prices.index.searchsorted(asof, side="right")
    start = max(0, end - spread_window)
    win = prices.iloc[start:end][[a, b]].dropna()
    if len(win) < spread_window // 2 or hr == 0:
        return {a: 0.0, b: 0.0}

    log_p = np.log(win)
    spread = log_p[a] - hr * log_p[b]
    mu = spread.mean()
    sd = spread.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return {a: 0.0, b: 0.0}
    z = float((spread.iloc[-1] - mu) / sd)

    if z > Z_ENTRY:
        side = -1.0  # short spread
    elif z < -Z_ENTRY:
        side = +1.0  # long spread
    else:
        side = 0.0

    if side == 0.0:
        return {a: 0.0, b: 0.0}

    # equal dollar per leg, opposite signs
    return {a: 0.5 * side, b: -0.5 * side}


def run_pairs_strategy(prices: pd.DataFrame,
                       oos_start: str = OOS_START,
                       rebalance_freq: str = REBALANCE_FREQ) -> pd.DataFrame:
    """Walk forward weekly. Weights at each rebalance sum to zero across the
    selected pairs, gross long = gross short, equal per pair.
    """
    rebals = pd.date_range(oos_start, prices.index.max(), freq=rebalance_freq)
    rebals = [d for d in rebals if d in prices.index or d <= prices.index.max()]

    weights = pd.DataFrame(0.0, index=rebals, columns=prices.columns)

    for d in rebals:
        pairs = select_pairs(prices, d)
        if not pairs:
            continue
        sleeve = {}
        for p in pairs:
            w = pairs_signal_to_weights(prices, p, d)
            for sym, val in w.items():
                sleeve[sym] = sleeve.get(sym, 0.0) + val

        if not sleeve:
            continue

        s = pd.Series(sleeve)
        long_gross = s[s > 0].sum()
        short_gross = -s[s < 0].sum()
        gross = long_gross + short_gross
        if gross > 0:
            s = s / gross  # normalise so total gross = 1
        for sym, w in s.items():
            if sym in weights.columns:
                weights.loc[d, sym] = w

    return weights
