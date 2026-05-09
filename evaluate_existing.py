#!/usr/bin/env python3
"""
Evaluate existing Yerevan transport routes using the same fitness function
as the GA-based route optimizer.
"""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from route_optimizer import (
    RETURN_PARTNER_JSON,
    eval_route_set,
    load_return_partner_map,
)

# ── Type aliases ──────────────────────────────────────────────────────────────
Route = list[str]          # ordered stop-ID sequence
RouteSet = list[Route]     # R routes forming one "solution"

# ── Parameters ────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")

# Fitness weights (ω1, ω2, ω3)
OMEGA           = (1.0, 1.0, 1.0)
# Max-score constants for F1, F2, F3
K1 = K2 = K3   = 10.0
# Demand-satisfaction weights for dT (α ≥ β ≥ γ)
ALPHA, BETA_W, GAMMA = 1.0, 0.5, 0.25
# Acceptable upper limit of IVT excess over T_min (km)
X_M             = 15.0
# Transfer penalty
U_TRANSFER      = 5.0


# ── Data loading ──────────────────────────────────────────────────────────────

class GraphData(NamedTuple):
    nodes:      dict               # nodeId -> {coords, activityLevel, lines, ...}
    adj:        dict[str, dict[str, float]]  # nodeId -> {targetId: min_weight}
    adj_list:   dict[str, list[tuple[str, float]]]  # nodeId -> [(targetId, weight)]
    demand:     np.ndarray         # (N, N) float64
    node_ids:   list[str]          # index -> nodeId
    id_to_idx:  dict[str, int]     # nodeId -> index


def load_graph_data() -> GraphData:
    with open(DATA_DIR / "transitGraph.json", encoding="utf-8") as f:
        graph = json.load(f)

    nodes: dict = graph["nodes"]
    raw_edges: dict = graph["edges"]

    with open(DATA_DIR / "node_index.json", encoding="utf-8") as f:
        id_to_idx: dict[str, int] = json.load(f)

    node_ids: list[str] = list(id_to_idx.keys())
    demand: np.ndarray = np.load(DATA_DIR / "demand_matrix.npy")

    # Deduplicate multigraph: keep minimum weight per (src, tgt) pair
    adj: dict[str, dict[str, float]] = {nid: {} for nid in node_ids}
    valid_ids = set(node_ids)
    for src, edges in raw_edges.items():
        if src not in valid_ids:
            continue
        for e in edges:
            tgt: str = e["target"]
            if tgt not in valid_ids:
                continue
            w: float = float(e["weight"])
            if tgt not in adj[src] or w < adj[src][tgt]:
                adj[src][tgt] = w

    adj_list: dict[str, list[tuple[str, float]]] = {
        nid: list(neighbors.items()) for nid, neighbors in adj.items()
    }

    return GraphData(nodes, adj, adj_list, demand, node_ids, id_to_idx)


def load_existing_routes(id_to_idx: dict[str, int]) -> list[Route]:
    """Parse allYerevanTransportLines.json → list of valid stop-ID sequences."""
    with open(DATA_DIR / "allYerevanTransportLines.json", encoding="utf-8") as f:
        lines: list[dict] = json.load(f)

    valid_ids = set(id_to_idx.keys())
    routes: list[Route] = []

    for entry in lines:
        for direction_key in ("line",):
            seq: list[dict] = entry.get(direction_key, [])
            stop_ids: Route = [
                item["id"]
                for item in seq
                if "id" in item and item["id"] in valid_ids
            ]
            if len(stop_ids) >= 2:
                routes.append(stop_ids)

    return routes


# ── Precomputation ────────────────────────────────────────────────────────────

def compute_activity(demand: np.ndarray) -> np.ndarray:
    """a[i] = Σ demand[i,:] + Σ demand[:,i]  (Eq. 1 — activity level)."""
    return demand.sum(axis=1) + demand.sum(axis=0)


def compute_tmin(
    adj: dict[str, dict[str, float]],
    node_ids: list[str],
    id_to_idx: dict[str, int],
) -> np.ndarray:
    """
    T_min[i][j] = shortest path distance (km) from i to j on the directed graph.
    Runs Dijkstra from every node using a binary heap (stdlib only).
    Complexity: O(N × (N + E) × log N) ≈ feasible for 1387 nodes / 1745 edges.
    """
    n = len(node_ids)
    # Build adjacency indexed by integer for speed
    int_adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for src, neighbors in adj.items():
        si = id_to_idx[src]
        for tgt, w in neighbors.items():
            int_adj[si].append((id_to_idx[tgt], w))

    INF = float("inf")
    t_min = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(t_min, 0.0)

    for src in range(n):
        dist: list[float] = [INF] * n
        dist[src] = 0.0
        heap: list[tuple[float, int]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, w in int_adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        t_min[src] = dist

    return t_min


def evaluate_existing_routes() -> None:
    """Evaluate the existing Yerevan transport routes using the GA fitness function."""
    print("Loading data …")
    gd = load_graph_data()
    n_stops = len(gd.node_ids)
    print(f"  Nodes: {n_stops} | Demand: {gd.demand.shape} | "
          f"Edges (unique pairs): {sum(len(v) for v in gd.adj.values())}")

    print("Computing activity levels …")
    activity = compute_activity(gd.demand)
    print(f"  Activity range: [{activity.min():.1f}, {activity.max():.1f}]")

    print("Computing T_min via Dijkstra (all-pairs shortest paths) …")
    t0 = time.time()
    t_min = compute_tmin(gd.adj, gd.node_ids, gd.id_to_idx)
    unreachable = int(np.isinf(t_min).sum() - n_stops)  # exclude diagonal
    print(f"  Done in {time.time()-t0:.1f}s | Unreachable pairs: {unreachable}")

    print("Loading existing routes …")
    existing_routes = load_existing_routes(gd.id_to_idx)
    print(f"  Loaded {len(existing_routes)} existing route directions")

    partner_map = load_return_partner_map(RETURN_PARTNER_JSON)
    if partner_map:
        print(
            f"Evaluating with synthetic return routes ({RETURN_PARTNER_JSON.name}, "
            f"{len(partner_map)} partner entries) …"
        )
    else:
        print(
            f"No partner file at {RETURN_PARTNER_JSON} — outbound-only eval "
            "(run line_return_mapping/global_stop_return_map.py)."
        )

    print("Evaluating existing routes …")
    totfit, metrics = eval_route_set(
        existing_routes, gd.demand, t_min, gd, partner_map
    )

    print("Existing routes evaluation:")
    print(f"  Number of routes (outbound): {len(existing_routes)}")
    print(f"  TOTFIT: {totfit:.4f}")
    print(f"  Direct coverage (d0): {metrics['d0p']:.1f}%")
    print(f"  1-transfer coverage (d1): {metrics['d1p']:.1f}%")
    print(f"  2-transfer coverage (d2): {metrics['d2p']:.1f}%")
    print(f"  Unsatisfied demand: {metrics['dunp']:.1f}%")
    print(f"  Average Travel Time (ATT): {metrics['ATT']:.2f} km")
    print(f"  F1 (travel time score): {metrics['F1']:.3f}")
    print(f"  F2 (demand satisfaction): {metrics['F2']:.3f}")
    print(f"  F3 (unsatisfied penalty): {metrics['F3']:.3f}")


if __name__ == "__main__":
    evaluate_existing_routes()