#!/usr/bin/env python3
"""
Evaluate existing Yerevan transport routes using the same fitness function
as the GA-based route optimizer.
"""

from __future__ import annotations

import copy
import csv
import heapq
import json
import random
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

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
        for direction_key in ("line"):
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


# ── EVAL — Fitness Evaluation ─────────────────────────────────────────────────

def _build_stop_on_routes(
    route_set: RouteSet,
    id_to_idx: dict[str, int],
    n_stops: int,
) -> np.ndarray:
    """
    Binary membership matrix (n_stops × R, float32).
    sor[s, r] = 1 iff stop s is on route r.
    """
    r_count = len(route_set)
    sor = np.zeros((n_stops, r_count), dtype=np.float32)
    for r_idx, route in enumerate(route_set):
        for stop_id in route:
            s = id_to_idx.get(stop_id)
            if s is not None:
                sor[s, r_idx] = 1.0
    return sor


def _route_cumulative_ivt(
    route: Route,
    adj: dict[str, dict[str, float]],
    id_to_idx: dict[str, int],
) -> tuple[list[int], np.ndarray]:
    """
    Compute IVT matrix for a single route (bidirectional).
    Returns (stop_indices, ivt_matrix) where ivt[a,b] = |cum[b] - cum[a]|.
    """
    n = len(route)
    stop_idxs = [id_to_idx[sid] for sid in route if sid in id_to_idx]
    valid_route = [sid for sid in route if sid in id_to_idx]
    n = len(valid_route)
    if n < 2:
        return stop_idxs, np.zeros((n, n))

    cum = np.zeros(n)
    for i in range(1, n):
        # Try forward edge, fallback to reverse edge weight
        w = adj.get(valid_route[i - 1], {}).get(valid_route[i])
        if w is None:
            w = adj.get(valid_route[i], {}).get(valid_route[i - 1], 0.0)
        cum[i] = cum[i - 1] + w

    ivt = np.abs(cum[:, None] - cum[None, :])
    return stop_idxs, ivt


def eval_route_set(
    route_set: RouteSet,
    demand: np.ndarray,
    t_min: np.ndarray,
    gd: GraphData,
) -> tuple[float, dict]:
    """
    Evaluate one route set. Returns (TOTFIT, metrics_dict).

    Coverage approach:
      sor  (N × R) stop-on-routes binary matrix
      d0_reach[i,j]  = sor @ sor.T > 0               (same route)
      route_overlap   = sor.T @ sor                    (R × R shared-stop counts)
      d1_reach[i,j]  = sor @ route_overlap_offdiag @ sor.T > 0  (one transfer)
      d2_reach[i,j]  = sor @ route_2hop @ sor.T > 0   (two transfers)

    All matmuls have inner dimension R=70 — efficient regardless of N.
    """
    n_stops = len(gd.node_ids)
    total_demand = float(demand.sum())
    if total_demand == 0:
        raise ValueError("Demand matrix is all zeros.")

    sor = _build_stop_on_routes(route_set, gd.id_to_idx, n_stops)  # (N, R)

    # ── Demand coverage ───────────────────────────────────────────────────────
    d0_reach: np.ndarray = (sor @ sor.T) > 0          # (N, N) bool
    np.fill_diagonal(d0_reach, False)

    # Route overlap: how many stops each pair of routes shares
    route_overlap = sor.T @ sor                        # (R, R) float32
    route_offdiag = route_overlap.copy()
    np.fill_diagonal(route_offdiag, 0.0)
    route_offdiag_bool = (route_offdiag > 0).astype(np.float32)

    d1_reach: np.ndarray = (sor @ route_offdiag_bool @ sor.T) > 0
    d1_reach &= ~d0_reach
    np.fill_diagonal(d1_reach, False)

    route_2hop = (route_offdiag_bool @ route_offdiag_bool)
    np.fill_diagonal(route_2hop, 0.0)
    route_2hop_bool = (route_2hop > 0).astype(np.float32)

    d2_reach: np.ndarray = (sor @ route_2hop_bool @ sor.T) > 0
    d2_reach &= ~d0_reach & ~d1_reach
    np.fill_diagonal(d2_reach, False)

    served_mask = d0_reach | d1_reach | d2_reach
    np.fill_diagonal(served_mask, False)

    # Exclude diagonal (self-demand) from all fractions
    off_diag_demand = total_demand - float(np.diag(demand).sum())

    d0  = float(demand[d0_reach].sum())  / off_diag_demand
    d1  = float(demand[d1_reach].sum())  / off_diag_demand
    d2  = float(demand[d2_reach].sum())  / off_diag_demand

    unserved_mask = ~served_mask
    np.fill_diagonal(unserved_mask, False)
    dun = float(demand[unserved_mask].sum()) / off_diag_demand
    dun = max(0.0, dun)

    # ── F2: demand satisfaction score (Eq. 6–7) ──────────────────────────────
    # dT = α·d0 + β·d1 + γ·d2  (Eq. 7)
    dT = ALPHA * d0 + BETA_W * d1 + GAMMA * d2
    dT_max = ALPHA + BETA_W + GAMMA  # max possible dT
    F2 = K2 * (dT / dT_max)

    # ── F3: unsatisfied demand penalty (Eq. 8) ────────────────────────────────
    # F3 = K3 at dun=0, F3 = 0 at dun=1
    F3 = K3 * (1.0 - dun) ** 2

    # ── F1: in-vehicle travel time score (Eq. 4–5) ────────────────────────────
    # Build IVT matrix for direct (d0) pairs
    ivt_mat = np.full((n_stops, n_stops), np.inf, dtype=np.float64)
    for route in route_set:
        valid = [sid for sid in route if sid in gd.id_to_idx]
        if len(valid) < 2:
            continue
        stop_idxs, ivt_route = _route_cumulative_ivt(valid, gd.adj, gd.id_to_idx)
        ix = np.array(stop_idxs)
        ivt_mat[np.ix_(ix, ix)] = np.minimum(
            ivt_mat[np.ix_(ix, ix)], ivt_route
        )

    # For d1/d2 pairs: approximate IVT = T_min + transfer penalty(ies)
    # Only set where IVT is still inf (d0 route already set finite values)
    ivt_mat[d1_reach] = t_min[d1_reach] + U_TRANSFER
    ivt_mat[d2_reach] = t_min[d2_reach] + 2.0 * U_TRANSFER

    # Only score served pairs with finite IVT (inf appears on unreachable pairs)
    finite_mask = served_mask & np.isfinite(ivt_mat) & np.isfinite(t_min)

    served_demand = demand[finite_mask]
    ivt_served    = ivt_mat[finite_mask]
    tmin_served   = t_min[finite_mask]

    # x = IVT - T_min  (excess travel time, Eq. 4)
    x = np.clip(ivt_served - tmin_served, 0.0, None)
    # f_ij = K1 * (1 - x/xm)^2 for x <= xm, else 0  (simplified Eq. 4)
    f_vals = np.where(x <= X_M, K1 * (1.0 - x / X_M) ** 2, 0.0)

    denom_f1 = float(served_demand.sum())
    F1 = float((served_demand * f_vals).sum() / denom_f1) if denom_f1 > 0 else 0.0

    # ATT: demand-weighted average IVT over finite-IVT served pairs (for reporting)
    att = float((served_demand * ivt_served).sum() / denom_f1) if denom_f1 > 0 else np.inf

    # ── TOTFIT (Eq. 3) ────────────────────────────────────────────────────────
    o1, o2, o3 = OMEGA
    totfit = o1 * F1 + o2 * F2 + o3 * F3

    metrics = {
        "d0p":    100.0 * d0,
        "d1p":    100.0 * d1,
        "d2p":    100.0 * d2,
        "dunp":   100.0 * dun,
        "ATT":    att,
        "F1":     F1,
        "F2":     F2,
        "F3":     F3,
        "TOTFIT": totfit,
    }
    return totfit, metrics


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

    print("Evaluating existing routes …")
    totfit, metrics = eval_route_set(existing_routes, gd.demand, t_min, gd)

    print("Existing routes evaluation:")
    print(f"  Number of routes: {len(existing_routes)}")
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