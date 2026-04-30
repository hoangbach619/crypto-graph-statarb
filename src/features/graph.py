"""KNN correlation graph and node-level structural features."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_knn_graph(corr: pd.DataFrame, k: int = 5) -> nx.Graph:
    """For each node keep its top-k neighbours by absolute correlation.
    Edge weights are the signed correlation. Self-loops are excluded.
    """
    if corr.empty:
        return nx.Graph()

    nodes = list(corr.columns)
    G = nx.Graph()
    G.add_nodes_from(nodes)

    abs_corr = corr.abs()
    abs_corr_arr = abs_corr.to_numpy(copy=True)
    np.fill_diagonal(abs_corr_arr, np.nan)
    abs_corr = pd.DataFrame(abs_corr_arr, index=abs_corr.index, columns=abs_corr.columns)

    for u in nodes:
        s = abs_corr[u].dropna()
        if s.empty:
            continue
        top = s.nlargest(k).index
        for v in top:
            w = float(corr.loc[u, v])
            if G.has_edge(u, v):
                # keep the larger absolute weight to break ties
                if abs(G[u][v]["weight"]) < abs(w):
                    G[u][v]["weight"] = w
            else:
                G.add_edge(u, v, weight=w)
    return G


def node_features(G: nx.Graph) -> pd.DataFrame:
    """Per-node: degree, clustering coefficient, eigenvector centrality.
    Eigenvector centrality is computed on |weights| since signed weights can
    cause numerical issues; degree and clustering use signed weights.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["degree", "clustering", "eigen_centrality"])

    deg = dict(G.degree(weight="weight"))
    clust = nx.clustering(G, weight="weight")

    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["weight"] = abs(d.get("weight", 1.0))
    try:
        eig = nx.eigenvector_centrality_numpy(H, weight="weight")
    except Exception:
        eig = {n: 0.0 for n in H.nodes()}

    rows = []
    for n in G.nodes():
        rows.append(
            {
                "symbol": n,
                "degree": deg.get(n, 0.0),
                "clustering": clust.get(n, 0.0),
                "eigen_centrality": eig.get(n, 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")
