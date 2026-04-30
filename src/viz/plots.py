"""Plot helpers. Save 150 DPI PNGs with no titles inside the image."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIG_DIR

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 0,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    }
)


def _save(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def equity_curve(returns_dict: dict[str, pd.Series], name: str = "equity_curve_v1_v2_v3.png") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, r in returns_dict.items():
        cum = (1 + r.fillna(0)).cumprod()
        ax.plot(cum.index, cum.values, label=label, lw=1.6)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return ($1 = 1.0)")
    ax.legend(loc="best")
    return _save(fig, name)


def drawdown(returns_dict: dict[str, pd.Series], name: str = "drawdown_v1_v2_v3.png") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, r in returns_dict.items():
        cum = (1 + r.fillna(0)).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        ax.plot(dd.index, dd.values, label=label, lw=1.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    return _save(fig, name)


def rolling_sharpe(returns_dict: dict[str, pd.Series], window: int = 63,
                   name: str = "rolling_sharpe.png") -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, r in returns_dict.items():
        rs = r.rolling(window).mean() / r.rolling(window).std(ddof=1) * np.sqrt(365)
        ax.plot(rs.index, rs.values, label=label, lw=1.2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Rolling {window}d Sharpe (annualised)")
    ax.legend(loc="best")
    return _save(fig, name)


def rank_ic_distribution(diags: dict[str, pd.DataFrame],
                          name: str = "rank_ic_distribution.png") -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, d in diags.items():
        if d is None or d.empty or "rank_ic" not in d.columns:
            continue
        s = d["rank_ic"].dropna()
        if len(s) == 0:
            continue
        ax.hist(s, bins=20, alpha=0.5, label=f"{label} (mean={s.mean():.3f})")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Per-rebalance Spearman rank IC")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    return _save(fig, name)


def feature_importance(imp: pd.DataFrame, name: str) -> Path:
    if imp is None or imp.empty:
        fig, ax = plt.subplots(figsize=(7, 1))
        ax.text(0.5, 0.5, "no importance data", ha="center", va="center")
        ax.axis("off")
        return _save(fig, name)
    means = imp.mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(means))))
    ax.barh(means.index, means.values, color="steelblue")
    ax.set_xlabel("Mean RF importance across rebalances")
    return _save(fig, name)


def regime_breakdown(returns_dict: dict[str, pd.Series],
                     fg: pd.Series | None = None,
                     name: str = "regime_breakdown.png") -> Path:
    """Bar chart: average daily return by regime (bull/bear) using a 60d
    rolling Bitcoin-proxy or fear-greed index. Falls back to up/down market.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    rows = []
    for label, r in returns_dict.items():
        if r is None or r.dropna().empty:
            continue
        if fg is not None and not fg.empty:
            f = fg.reindex(r.index).ffill().dropna()
            r_aligned = r.reindex(f.index)
            high = r_aligned[f > 50].mean()
            low = r_aligned[f <= 50].mean()
        else:
            r_aligned = r.copy()
            high = r_aligned[r_aligned > 0].mean()
            low = r_aligned[r_aligned <= 0].mean()
        rows.append((label, "high_sentiment", high))
        rows.append((label, "low_sentiment", low))
    if not rows:
        ax.text(0.5, 0.5, "no data", ha="center")
        ax.axis("off")
        return _save(fig, name)
    df = pd.DataFrame(rows, columns=["strategy", "regime", "avg_daily_ret"])
    pivot = df.pivot(index="strategy", columns="regime", values="avg_daily_ret")
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Average daily net return")
    ax.legend(title="Regime")
    plt.xticks(rotation=0)
    return _save(fig, name)


def monthly_returns_heatmap(r: pd.Series, name: str = "monthly_returns_heatmap.png") -> Path:
    r = r.dropna()
    if r.empty:
        fig, ax = plt.subplots(figsize=(6, 1))
        ax.text(0.5, 0.5, "no data", ha="center")
        ax.axis("off")
        return _save(fig, name)
    monthly = (1 + r).resample("ME").prod() - 1
    table = monthly.to_frame("ret")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot_table(index="year", columns="month", values="ret")
    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(pivot.index))))
    sns.heatmap(pivot * 100, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                cbar_kws={"label": "Monthly return (%)"}, ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return _save(fig, name)
