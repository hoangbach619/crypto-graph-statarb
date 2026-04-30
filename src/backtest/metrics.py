"""Performance metrics. Crypto trades 365 days/year so we annualise on 365."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import ANNUALISATION_FACTOR


def sharpe_ratio(r: pd.Series, ann: int = ANNUALISATION_FACTOR) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ann))


def sortino(r: pd.Series, ann: int = ANNUALISATION_FACTOR) -> float:
    r = r.dropna()
    downside = r[r < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / downside.std(ddof=1) * np.sqrt(ann))


def annualised_return(r: pd.Series, ann: int = ANNUALISATION_FACTOR) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    cum = (1 + r).prod()
    years = len(r) / ann
    if years == 0:
        return float("nan")
    return float(cum ** (1 / years) - 1)


def max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def calmar(r: pd.Series, ann: int = ANNUALISATION_FACTOR) -> float:
    mdd = max_drawdown(r)
    if mdd == 0 or not np.isfinite(mdd):
        return float("nan")
    return annualised_return(r, ann=ann) / abs(mdd)


def win_rate(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def profit_factor(r: pd.Series) -> float:
    r = r.dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("inf")
    return float(pos / neg)


def t_statistic(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 3:
        return float("nan")
    mean, sd = r.mean(), r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(mean / (sd / np.sqrt(len(r))))


def newey_west_t(series, lag):
    """HAC t-stat with Newey-West correction for `lag` periods of autocorrelation."""
    import numpy as np
    s = series.dropna().values
    n = len(s)
    if n < lag + 2:
        return np.nan
    mean = s.mean()
    e = s - mean
    gamma_0 = (e**2).sum() / n
    var_nw = gamma_0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        gamma_k = (e[k:] * e[:-k]).sum() / n
        var_nw += 2 * w * gamma_k
    if var_nw <= 0:
        return np.nan
    se_nw = np.sqrt(var_nw / n)
    return mean / se_nw


def rank_ic_summary(diag: pd.DataFrame) -> dict:
    if diag is None or diag.empty or "rank_ic" not in diag.columns:
        return {"mean_ic": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "ic_t": float("nan"), "n_obs": 0}
    s = diag["rank_ic"].dropna()
    if len(s) == 0:
        return {"mean_ic": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "ic_t": float("nan"), "n_obs": 0}
    mean_ic = s.mean()
    ic_std = s.std(ddof=1) if len(s) > 1 else float("nan")
    icir = mean_ic / ic_std if ic_std and ic_std != 0 else float("nan")
    ic_t = mean_ic / (ic_std / np.sqrt(len(s))) if ic_std and ic_std != 0 else float("nan")
    return {"mean_ic": mean_ic, "ic_std": ic_std, "icir": icir, "ic_t": ic_t, "n_obs": len(s)}


def top_bottom_spread(diag: pd.DataFrame) -> dict:
    if diag is None or diag.empty or "top_bot_spread" not in diag.columns:
        return {"mean_spread": float("nan"), "t_stat": float("nan")}
    s = diag["top_bot_spread"].dropna()
    if len(s) < 2:
        return {"mean_spread": float("nan"), "t_stat": float("nan")}
    return {"mean_spread": float(s.mean()),
            "t_stat": float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))}


def summary_metrics(name: str, r: pd.Series, gross: pd.Series | None = None,
                    diag: pd.DataFrame | None = None) -> dict:
    out = {
        "strategy": name,
        "ann_return_net": annualised_return(r),
        "sharpe_net": sharpe_ratio(r),
        "sortino_net": sortino(r),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar(r),
        "win_rate_daily": win_rate(r),
        "profit_factor": profit_factor(r),
        "t_stat_daily": t_statistic(r),
        "n_days": int(r.dropna().shape[0]),
    }
    if gross is not None:
        out["ann_return_gross"] = annualised_return(gross)
        out["sharpe_gross"] = sharpe_ratio(gross)
    ics = rank_ic_summary(diag if diag is not None else pd.DataFrame())
    out.update(ics)
    if diag is not None and not diag.empty and "rank_ic" in diag.columns:
        # 21-day forward return / 7-day rebalance = 3 weeks of overlap, use lag=3
        ic_t_nw = newey_west_t(diag["rank_ic"], lag=3)
    else:
        ic_t_nw = float("nan")
    out["ic_t_naive"] = out.get("ic_t", float("nan"))
    out["ic_t_newey_west"] = ic_t_nw
    tb = top_bottom_spread(diag if diag is not None else pd.DataFrame())
    out["topbot_spread"] = tb["mean_spread"]
    out["topbot_t"] = tb["t_stat"]
    return out
