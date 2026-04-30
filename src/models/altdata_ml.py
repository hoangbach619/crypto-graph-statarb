"""v3 strategy: v2 features + alternative data (funding, OI, fear-greed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import OOS_START
from src.models.graph_ml import (
    FEATURE_COLS_V2,
    build_features_at,
    build_panel as build_panel_v2,
    fit_predict_at,
    run_strategy,
)
from src.models.graph_ml import _safe_z


FEATURE_COLS_V3 = FEATURE_COLS_V2 + [
    "funding_rate_z30d",
    "funding_rate_change_5d",
    "oi_change_5d",
    "oi_zscore_30d",
    "fg_x_vol",
]


def _altdata_features_at(funding: pd.DataFrame, oi: pd.DataFrame, fg: pd.DataFrame,
                         vol_20: pd.Series, asof: pd.Timestamp) -> pd.DataFrame:
    """Compute alt-data features for the symbols available in `vol_20.index`
    at date `asof` using only data <= asof.
    """
    syms = list(vol_20.index)

    # funding: aggregate daily already; compute z over 30d and 5d change
    if not funding.empty:
        f_end = funding.index.searchsorted(asof, side="right")
        f_win = funding.iloc[max(0, f_end - 31):f_end]
        if len(f_win) >= 5:
            mu = f_win.mean()
            sd = f_win.std(ddof=1).replace(0, np.nan)
            last = f_win.iloc[-1]
            f_z = ((last - mu) / sd).reindex(syms).fillna(0.0)
            f_chg = (f_win.iloc[-1] - f_win.iloc[max(0, len(f_win) - 6)]).reindex(syms).fillna(0.0)
        else:
            f_z = pd.Series(0.0, index=syms)
            f_chg = pd.Series(0.0, index=syms)
    else:
        f_z = pd.Series(0.0, index=syms)
        f_chg = pd.Series(0.0, index=syms)

    # open interest: 5d pct change and 30d z-score
    if not oi.empty:
        o_end = oi.index.searchsorted(asof, side="right")
        o_win = oi.iloc[max(0, o_end - 31):o_end]
        if len(o_win) >= 6:
            o_chg = (o_win.iloc[-1] / o_win.iloc[max(0, len(o_win) - 6)] - 1).reindex(syms).fillna(0.0)
            mu = o_win.mean()
            sd = o_win.std(ddof=1).replace(0, np.nan)
            o_z = ((o_win.iloc[-1] - mu) / sd).reindex(syms).fillna(0.0)
        else:
            o_chg = pd.Series(0.0, index=syms)
            o_z = pd.Series(0.0, index=syms)
    else:
        o_chg = pd.Series(0.0, index=syms)
        o_z = pd.Series(0.0, index=syms)

    # fear-greed: scalar today, interacted with each stock's 20d vol
    if not fg.empty:
        fg_end = fg.index.searchsorted(asof, side="right")
        if fg_end > 0:
            fg_today = float(fg.iloc[fg_end - 1]["fear_greed"])
        else:
            fg_today = 50.0
    else:
        fg_today = 50.0
    fg_z = (fg_today - 50.0) / 50.0  # -1..+1 scale
    fg_x_vol = pd.Series(fg_z * vol_20.values, index=syms)

    out = pd.DataFrame(
        {
            "funding_rate_z30d": f_z.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "funding_rate_change_5d": f_chg.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "oi_change_5d": o_chg.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "oi_zscore_30d": o_z.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "fg_x_vol": fg_x_vol.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        }
    )
    out.index.name = "symbol"
    return out


def build_panel_v3(prices: pd.DataFrame, returns: pd.DataFrame,
                   funding: pd.DataFrame, oi: pd.DataFrame, fg: pd.DataFrame,
                   rebals: list[pd.Timestamp]) -> pd.DataFrame:
    """v2 panel plus alt-data columns merged on (date, symbol)."""
    rows = []
    for d in rebals:
        f = build_features_at(prices, returns, d)
        if f.empty:
            continue
        end = returns.index.searchsorted(d, side="right")
        fwd_end = end + 21
        if fwd_end >= len(returns):
            continue
        fwd_window = returns.iloc[end:fwd_end]
        fwd_ret = (fwd_window.add(1).prod(min_count=fwd_window.shape[0]) - 1).reindex(f.index)

        alt = _altdata_features_at(funding, oi, fg, f["vol_20d"], d)
        merged = f.join(alt, how="left").fillna(0.0)
        merged["date"] = d
        merged["fwd_ret_21d"] = fwd_ret.values
        merged = merged[merged["fwd_ret_21d"].notna()]
        if len(merged) < 8:
            continue
        from scipy.stats import rankdata
        merged["fwd_rank"] = rankdata(merged["fwd_ret_21d"]) / len(merged)
        rows.append(merged.reset_index())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_v3(prices_columns: list[str], panel: pd.DataFrame,
           oos_start: str = OOS_START):
    return run_strategy(panel, prices_columns, oos_start=oos_start, feature_cols=FEATURE_COLS_V3)
