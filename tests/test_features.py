"""Lightweight sanity checks for graph + Laplacian + backtest cost math."""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.engine import apply_backtest
from src.features.graph import build_knn_graph
from src.features.laplacian import laplacian_smooth, normalised_laplacian


def test_knn_graph_size():
    rng = np.random.default_rng(42)
    n = 10
    syms = [f"S{i}" for i in range(n)]
    arr = rng.uniform(-1, 1, (n, n))
    arr = (arr + arr.T) / 2
    np.fill_diagonal(arr, 1.0)
    corr = pd.DataFrame(arr, index=syms, columns=syms)
    G = build_knn_graph(corr, k=3)
    # each node has at least k neighbours (some may have more from being top-k of others)
    for node in G.nodes():
        assert G.degree(node) >= 3, f"{node} has degree {G.degree(node)}"


def test_laplacian_zero_input():
    G = nx.cycle_graph(5)
    G = nx.relabel_nodes(G, {i: f"S{i}" for i in range(5)})
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    r = pd.Series(0.0, index=[f"S{i}" for i in range(5)])
    out = laplacian_smooth(r, G, alpha=0.7)
    assert (out == 0.0).all()


def test_backtest_costs():
    idx = pd.date_range("2022-01-01", periods=5)
    syms = ["A", "B"]
    weights = pd.DataFrame(0.0, index=idx[:1], columns=syms)
    weights.iloc[0] = [0.5, -0.5]
    rets = pd.DataFrame(0.0, index=idx, columns=syms)
    out = apply_backtest(weights, rets, funding=None)
    # turnover on day 1 = |0.5| + |-0.5| = 1.0; cost = 1.0 * 4bp = 0.0004
    assert abs(out["turnover"].iloc[0] - 1.0) < 1e-9
    assert abs(out["cost"].iloc[0] - 4 / 10000) < 1e-9


if __name__ == "__main__":
    test_knn_graph_size()
    test_laplacian_zero_input()
    test_backtest_costs()
    print("all tests passed")
