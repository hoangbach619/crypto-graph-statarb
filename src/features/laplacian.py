"""Graph Laplacian smoothing for cross-sectional mean reversion signals."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def normalised_laplacian(G: nx.Graph, nodes: list[str]) -> np.ndarray:
    """Return symmetric normalised Laplacian L = I - D^{-1/2} W D^{-1/2} for the
    induced subgraph on the requested node ordering, using |weight|.
    """
    if G.number_of_nodes() == 0 or not nodes:
        return np.zeros((len(nodes), len(nodes)))

    n = len(nodes)
    W = np.zeros((n, n))
    idx = {s: i for i, s in enumerate(nodes)}
    for u, v, d in G.edges(data=True):
        if u in idx and v in idx:
            w = abs(float(d.get("weight", 0.0)))
            W[idx[u], idx[v]] = w
            W[idx[v], idx[u]] = w
    deg = W.sum(axis=1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
    return L


def laplacian_smooth(returns_today: pd.Series, G: nx.Graph, alpha: float = 0.7) -> pd.Series:
    """residual_i = r_i - alpha * (L @ r)_i

    A large positive residual => stock outperformed its graph neighbourhood
    today; a large negative residual => underperformed. We use this as a
    mean-reversion signal in v2/v3.
    """
    common = [s for s in returns_today.index if s in G.nodes()]
    r = returns_today.reindex(common).fillna(0.0).values.astype(float)
    L = normalised_laplacian(G, common)
    smoothed = alpha * (L @ r)
    residual = r - smoothed
    return pd.Series(residual, index=common, name="graph_residual")
