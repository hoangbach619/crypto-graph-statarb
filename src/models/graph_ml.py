"""v2 strategy: graph features + LASSO + RandomForest predicting cross-sectional
21-day forward rank. Weekly rebalance, expanding training window, dollar
neutral, equal weight within sleeve.
"""
from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, spearmanr

from src.config import (
    CORR_WINDOW,
    FORWARD_HORIZON_DAYS,
    KNN_K,
    LASSO_CV_FOLDS,
    LAPLACIAN_ALPHA,
    N_LONGS,
    N_SHORTS,
    OOS_START,
    REBALANCE_FREQ,
    RF_MAX_DEPTH,
    RF_MAX_FEATURES,
    RF_N_ESTIMATORS,
    SEED,
)
from src.features.correlation import correlation_at
from src.features.graph import build_knn_graph, node_features
from src.features.laplacian import laplacian_smooth

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _safe_z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def build_features_at(prices: pd.DataFrame, returns: pd.DataFrame,
                      asof: pd.Timestamp,
                      corr_window: int = CORR_WINDOW,
                      knn_k: int = KNN_K,
                      laplacian_alpha: float = LAPLACIAN_ALPHA) -> pd.DataFrame:
    """Cross-sectional feature row for date `asof` using only data <= asof.
    Returns a DataFrame indexed by symbol.
    """
    end = returns.index.searchsorted(asof, side="right")
    if end < 60:
        return pd.DataFrame()
    past_ret = returns.iloc[:end]
    past_px = prices.iloc[:end]

    corr = correlation_at(past_ret, asof, window=corr_window)
    if corr.empty:
        return pd.DataFrame()

    G = build_knn_graph(corr, k=knn_k)
    nf = node_features(G)

    today_ret = past_ret.iloc[-1].reindex(corr.columns)
    g_resid = laplacian_smooth(today_ret, G, alpha=laplacian_alpha)

    # 20d rolling std of residual approximation: use std of last 20 daily returns
    vol_20 = past_ret[corr.columns].iloc[-20:].std(ddof=1)
    vol_60 = past_ret[corr.columns].iloc[-60:].std(ddof=1)
    vol_ratio = (vol_20 / vol_60).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    graph_z = (g_resid / vol_20.reindex(g_resid.index).replace(0, np.nan)).fillna(0.0)

    rev_5 = past_px[corr.columns].iloc[-1] / past_px[corr.columns].iloc[-6] - 1
    rev_10 = past_px[corr.columns].iloc[-1] / past_px[corr.columns].iloc[-11] - 1
    mom_21 = past_px[corr.columns].iloc[-1] / past_px[corr.columns].iloc[-22] - 1
    mom_63 = past_px[corr.columns].iloc[-1] / past_px[corr.columns].iloc[-64] - 1 if len(past_px) > 63 else pd.Series(0.0, index=corr.columns)

    feats = pd.DataFrame(index=corr.columns)
    feats["graph_zscore"] = graph_z
    feats["node_degree"] = nf.reindex(corr.columns)["degree"]
    feats["clustering_coeff"] = nf.reindex(corr.columns)["clustering"]
    feats["eigen_centrality"] = nf.reindex(corr.columns)["eigen_centrality"]
    feats["reversal_5d"] = rev_5
    feats["reversal_10d"] = rev_10
    feats["momentum_21d"] = mom_21
    feats["momentum_63d"] = mom_63
    feats["vol_20d"] = vol_20
    feats["vol_60d"] = vol_60
    feats["vol_ratio"] = vol_ratio

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats.index.name = "symbol"
    return feats


def build_panel(prices: pd.DataFrame, returns: pd.DataFrame,
                rebals: list[pd.Timestamp]) -> pd.DataFrame:
    """Stack feature rows across all rebalance dates. Adds forward 21d return
    and forward rank as the supervised target."""
    rows = []
    for d in rebals:
        f = build_features_at(prices, returns, d)
        if f.empty:
            continue
        end = returns.index.searchsorted(d, side="right")
        fwd_end = end + FORWARD_HORIZON_DAYS
        if fwd_end >= len(returns):
            continue
        fwd_window = returns.iloc[end:fwd_end]
        fwd_ret = (fwd_window.add(1).prod(min_count=fwd_window.shape[0]) - 1).reindex(f.index)
        f = f.copy()
        f["date"] = d
        f["fwd_ret_21d"] = fwd_ret.values
        valid = f["fwd_ret_21d"].notna()
        f = f[valid]
        if len(f) < 8:  # need a usable cross-section
            continue
        f["fwd_rank"] = rankdata(f["fwd_ret_21d"]) / len(f)
        rows.append(f.reset_index())
    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    return panel


FEATURE_COLS_V2 = [
    "graph_zscore", "node_degree", "clustering_coeff", "eigen_centrality",
    "reversal_5d", "reversal_10d", "momentum_21d", "momentum_63d",
    "vol_20d", "vol_60d", "vol_ratio",
]


def fit_predict_at(panel: pd.DataFrame, asof: pd.Timestamp,
                   feature_cols: list[str] = None) -> tuple[pd.DataFrame, dict, dict]:
    """Train LASSO + RF on all panel rows with date < asof, predict on asof.
    Returns (predictions DF for asof, lasso coeffs dict, rf importances dict).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_V2
    train = panel[panel["date"] < asof].dropna(subset=feature_cols + ["fwd_rank"])
    test = panel[panel["date"] == asof].dropna(subset=feature_cols)

    if len(train) < 50 or test.empty:
        return pd.DataFrame(), {}, {}

    X_tr = train[feature_cols].values
    y_tr = train["fwd_rank"].values
    X_te = test[feature_cols].values

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    lasso = LassoCV(cv=LASSO_CV_FOLDS, random_state=SEED, n_jobs=1, max_iter=20000).fit(X_tr_s, y_tr)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        max_features=RF_MAX_FEATURES,
        random_state=SEED,
        n_jobs=1,
    ).fit(X_tr, y_tr)

    p_lasso = lasso.predict(X_te_s)
    p_rf = rf.predict(X_te)
    pred = (p_lasso + p_rf) / 2

    out = test[["symbol", "date"]].copy()
    out["pred"] = pred
    out["actual_rank"] = test["fwd_rank"].values if "fwd_rank" in test.columns else np.nan
    coefs = dict(zip(feature_cols, lasso.coef_))
    imps = dict(zip(feature_cols, rf.feature_importances_))
    return out, coefs, imps


def run_strategy(panel: pd.DataFrame,
                 prices_columns: list[str],
                 oos_start: str = OOS_START,
                 feature_cols: list[str] = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate weekly weights from cross-sectional predictions.

    Returns:
      weights: DataFrame (date x symbol)
      diagnostics: DataFrame with per-rebalance rank IC, top-bottom spread
      importances: DataFrame of feature importances per rebalance
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_V2

    rebals = sorted(panel["date"].unique())
    rebals = [d for d in rebals if pd.Timestamp(d) >= pd.Timestamp(oos_start)]

    weights = pd.DataFrame(0.0, index=rebals, columns=prices_columns)
    diag_rows = []
    imp_rows = []

    for d in rebals:
        preds, coefs, imps = fit_predict_at(panel, d, feature_cols=feature_cols)
        if preds.empty:
            continue

        preds = preds.sort_values("pred")
        n = len(preds)
        n_long = min(N_LONGS, n // 2)
        n_short = min(N_SHORTS, n // 2)
        shorts = preds.iloc[:n_short]["symbol"].tolist()
        longs = preds.iloc[-n_long:]["symbol"].tolist()

        w_long = 0.5 / n_long if n_long else 0
        w_short = -0.5 / n_short if n_short else 0
        for s in longs:
            if s in weights.columns:
                weights.loc[d, s] = w_long
        for s in shorts:
            if s in weights.columns:
                weights.loc[d, s] = w_short

        if preds["actual_rank"].notna().all() and len(preds) > 3:
            ic, _ = spearmanr(preds["pred"], preds["actual_rank"])
        else:
            ic = np.nan

        actual = preds["actual_rank"]
        if actual.notna().any():
            top = preds.nlargest(n_long, "pred")["actual_rank"].mean()
            bot = preds.nsmallest(n_short, "pred")["actual_rank"].mean()
            tb_spread = top - bot
        else:
            tb_spread = np.nan

        diag_rows.append({"date": d, "rank_ic": ic, "top_bot_spread": tb_spread, "n": n})
        imp_rows.append({"date": d, **imps})

    diag = pd.DataFrame(diag_rows).set_index("date") if diag_rows else pd.DataFrame()
    imp = pd.DataFrame(imp_rows).set_index("date") if imp_rows else pd.DataFrame()
    return weights, diag, imp
