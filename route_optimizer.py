#!/usr/bin/env python3
"""
GA-based Transit Route Network Designer
Implements: Chakroborty & Dwivedi (2002), Engineering Optimization 34(1), 83-100.

Inputs  (all under data/):
  transitGraph.json            1387 nodes, 6375 directed multigraph edges
  demand_matrix.npy            (1387, 1387) log1p-scaled OD demand
  node_index.json              nodeId -> matrix row/col index
  allYerevanTransportLines.json  74 existing routes (seeds one population member)

Outputs:
  best_routes.json             optimised route set (list of stop-ID sequences)
  optimization_log.csv         per-generation metrics
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

# Optional live plotting — renders optimization_progress.png after each generation
try:
    from plot_results import render as _render_plot
    _PLOT_OUT = Path("optimization_progress.png")
    def _save_plot(log_path: Path) -> None:
        try:
            _render_plot(log_path, _PLOT_OUT)
        except Exception:
            pass  # never crash the optimizer because of plotting
except ImportError:
    def _save_plot(log_path: Path) -> None:  # type: ignore[misc]
        pass

# Optional route map — renders an interactive HTML map from a route set
try:
    from visualize_routes import load_graph as _load_graph_for_map, build_html as _build_html
    def _save_route_map(route_set: RouteSet, metrics: dict, totfit: float, out_html: Path, label: str) -> None:  # type: ignore[misc]
        try:
            nodes, _ = _load_graph_for_map(DATA_DIR / "transitGraph.json")
            html = _build_html(route_set, nodes, metrics, totfit, label)
            out_html.parent.mkdir(parents=True, exist_ok=True)
            out_html.write_text(html, encoding="utf-8")
        except Exception:
            pass
except ImportError:
    def _save_route_map(route_set: RouteSet, metrics: dict, totfit: float, out_html: Path, label: str) -> None:  # type: ignore[misc]
        pass

# ── Type aliases ──────────────────────────────────────────────────────────────
Route = list[str]          # ordered stop-ID sequence
RouteSet = list[Route]     # R routes forming one "solution"

# ── Parameters ────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")

R               = 70       # routes per route set
N_POP           = 20       # population size
M_COPIES        = 5        # copies per string in MODIFY (pool size = N_POP × M_COPIES)
K_INS           = 200      # INS cardinality (top-K most-active nodes)
MAX_NODES_ROUTE = 25       # max stops per route (M)
MAX_LEN_ROUTE   = 30.0     # max route length in km (L)
U_TRANSFER      = 5.0      # transfer penalty (same units as edge weights, km here)
MAX_GENERATIONS  = 100     # GA iterations
CROSSOVER_PROB   = 0.5     # inter-string crossover probability
MUTATION_PROB    = 0.01    # per-node mutation probability
CHECKPOINT_EVERY = 10      # save routes snapshot every N generations (0 = disabled)

# Fitness weights (ω1, ω2, ω3)
OMEGA           = (1.0, 1.0, 1.0)
# Max-score constants for F1, F2, F3
K1 = K2 = K3   = 10.0
# Demand-satisfaction weights for dT (α ≥ β ≥ γ)
ALPHA, BETA_W, GAMMA = 1.0, 0.5, 0.25
# Acceptable upper limit of IVT excess over T_min (km)
X_M             = 15.0

RANDOM_SEED     = 42


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
        for direction_key in ("line", "returnLine"):
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


# ── IRSG — Initial Route Set Generation ──────────────────────────────────────

def _direction_bias(
    nodes: dict,
    prev_id: str | None,
    current_id: str,
    candidate_id: str,
) -> float:
    """
    Biasing term d_k (Eq. 2): penalise backtracking by comparing the proposed
    direction vector with the current travel direction via cosine similarity.
    Mapped to (0.05, 1.0] so backtracking nodes are unlikely but not impossible.
    """
    if prev_id is None:
        return 1.0

    pc = nodes[prev_id]["coords"]
    cc = nodes[current_id]["coords"]
    nc = nodes[candidate_id]["coords"]

    dx_cur = cc["lng"] - pc["lng"]
    dy_cur = cc["lat"] - pc["lat"]
    dx_new = nc["lng"] - cc["lng"]
    dy_new = nc["lat"] - cc["lat"]

    norm_cur = (dx_cur**2 + dy_cur**2) ** 0.5
    norm_new = (dx_new**2 + dy_new**2) ** 0.5

    if norm_cur < 1e-9 or norm_new < 1e-9:
        return 1.0

    cosine = (dx_cur * dx_new + dy_cur * dy_new) / (norm_cur * norm_new)
    # Map [-1, 1] → [0.05, 1.0]
    return max(0.05, (cosine + 1.0) / 2.0)


def _build_one_route(
    gd: GraphData,
    activity: np.ndarray,
    rng: random.Random,
    first_node_id: str,
) -> Route:
    """Grow a single route from first_node_id using IRSG node-selection (Eq. 2)."""
    route: Route = [first_node_id]
    route_set: set[str] = {first_node_id}
    current_id = first_node_id
    prev_id: str | None = None
    total_len = 0.0

    while True:
        vns = [
            (tgt, w)
            for tgt, w in gd.adj_list.get(current_id, [])
            if tgt not in route_set
        ]
        if not vns:
            break

        weights = []
        for tgt, w in vns:
            a_k = float(activity[gd.id_to_idx[tgt]])
            d_k = _direction_bias(gd.nodes, prev_id, current_id, tgt)
            weights.append(max(1e-9, d_k * a_k))

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        # Check termination criteria before committing
        chosen_tgt, chosen_edge_w = rng.choices(vns, weights=probs, k=1)[0]

        if len(route) >= MAX_NODES_ROUTE:
            break
        if total_len + chosen_edge_w > MAX_LEN_ROUTE:
            break

        route.append(chosen_tgt)
        route_set.add(chosen_tgt)
        total_len += chosen_edge_w
        prev_id = current_id
        current_id = chosen_tgt

    return route


def irsg(
    gd: GraphData,
    activity: np.ndarray,
    rng: random.Random,
    n_route_sets: int,
) -> list[RouteSet]:
    """
    Generate n_route_sets initial route sets, each containing R routes.
    Returns list[RouteSet].
    """
    n = len(gd.node_ids)
    sorted_by_activity = sorted(range(n), key=lambda i: activity[i], reverse=True)

    population: list[RouteSet] = []

    for _ in range(n_route_sets):
        ins: list[int] = list(sorted_by_activity[:K_INS])
        ins_set: set[int] = set(ins)

        route_set: RouteSet = []
        for _ in range(R):
            ins_list = list(ins_set)
            acts = [activity[i] for i in ins_list]
            total_a = sum(acts)
            probs = (
                [a / total_a for a in acts]
                if total_a > 0
                else [1.0 / len(ins_list)] * len(ins_list)
            )

            chosen_idx: int = rng.choices(ins_list, weights=probs, k=1)[0]
            first_node = gd.node_ids[chosen_idx]

            ins_set.discard(chosen_idx)
            if len(ins_set) < K_INS // 2:
                for new_idx in sorted_by_activity:
                    if new_idx not in ins_set:
                        ins_set.add(new_idx)
                    if len(ins_set) >= K_INS:  # check AFTER potential add
                        break

            route = _build_one_route(gd, activity, rng, first_node)
            route_set.append(route)

        population.append(route_set)

    return population


def seed_from_existing(existing: list[Route], rng: random.Random) -> RouteSet:
    """Build one RouteSet by sampling R routes from the existing parsed routes."""
    if len(existing) >= R:
        return rng.sample(existing, R)
    result = list(existing)
    while len(result) < R:
        result.append(rng.choice(existing))
    return result


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


# ── MODIFY — GA Operators ─────────────────────────────────────────────────────

def crossover_inter(
    rs1: RouteSet,
    rs2: RouteSet,
    rng: random.Random,
) -> tuple[RouteSet, RouteSet]:
    """
    Inter-string crossover (Figure 4): randomly pick a demarcation line
    and exchange route-suffixes between two parent route sets.
    """
    r1, r2 = len(rs1), len(rs2)
    if r1 < 2 or r2 < 2:
        return copy.deepcopy(rs1), copy.deepcopy(rs2)

    cut = rng.randint(1, min(r1, r2) - 1)
    child1: RouteSet = [r[:] for r in rs1[:cut]] + [r[:] for r in rs2[cut:]]
    child2: RouteSet = [r[:] for r in rs2[:cut]] + [r[:] for r in rs1[cut:]]
    return child1, child2


def crossover_intra(rs: RouteSet, rng: random.Random) -> RouteSet:
    """
    Intra-string crossover (Figure 5): pick two routes sharing a common node,
    exchange tails at that node. Returns a modified shallow copy.
    """
    rs = [r[:] for r in rs]
    if len(rs) < 2:
        return rs

    indices = list(range(len(rs)))
    rng.shuffle(indices)

    for attempt in range(min(20, len(rs) * (len(rs) - 1) // 2)):
        r1_idx = rng.choice(indices)
        r2_idx = rng.choice(indices)
        if r1_idx == r2_idx:
            continue

        r1, r2 = rs[r1_idx], rs[r2_idx]
        common = list(set(r1) & set(r2))
        if not common:
            continue

        node_k = rng.choice(common)
        k1 = r1.index(node_k)
        k2 = r2.index(node_k)

        new_r1 = r1[: k1 + 1] + r2[k2 + 1:]
        new_r2 = r2[: k2 + 1] + r1[k1 + 1:]

        if len(new_r1) >= 2 and len(new_r2) >= 2:
            rs[r1_idx] = new_r1
            rs[r2_idx] = new_r2
            break

    return rs


def _repair_gap(
    src: str,
    dst: str,
    adj: dict[str, dict[str, float]],
) -> list[str]:
    """
    Dijkstra bridge from src to dst (for mutation feasibility repair).
    Returns a path [src, ..., dst] or [] if unreachable within 8 hops.
    """
    dist: dict[str, float] = {src: 0.0}
    prev: dict[str, str | None] = {src: None}
    heap = [(0.0, src)]
    visited: set[str] = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == dst:
            path: list[str] = []
            node: str | None = dst
            while node is not None:
                path.append(node)
                node = prev.get(node)
            return path[::-1]
        if len(visited) > 8:
            break
        for v, w in adj.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return []


def mutate(
    rs: RouteSet,
    gd: GraphData,
    rng: random.Random,
) -> RouteSet:
    """
    Mutation operator: for each node in each route, with probability
    MUTATION_PROB, replace it with a random adjacent neighbour. If the
    substitution breaks path continuity, insert bridge nodes via short
    Dijkstra search (Section 3.3).
    """
    rs = [r[:] for r in rs]

    for r_idx, route in enumerate(rs):
        if len(route) < 2:
            continue

        new_route: Route = []
        i = 0
        while i < len(route):
            node = route[i]
            if rng.random() < MUTATION_PROB:
                neighbors = [t for t, _ in gd.adj_list.get(node, [])]
                route_set = set(new_route) | set(route[i + 1:])
                candidates = [n for n in neighbors if n not in route_set]
                if not candidates:
                    candidates = neighbors
                if candidates:
                    replacement = rng.choice(candidates)
                    # Repair: if there's a next node, ensure replacement can reach it
                    if i + 1 < len(route):
                        next_node = route[i + 1]
                        if next_node not in gd.adj.get(replacement, {}):
                            bridge = _repair_gap(replacement, next_node, gd.adj)
                            if bridge and len(new_route) + len(bridge) <= MAX_NODES_ROUTE:
                                new_route.extend(bridge[:-1])
                                # bridge[-1] == next_node, will be added in next iter
                            else:
                                new_route.append(node)  # revert mutation
                                i += 1
                                continue
                        new_route.append(replacement)
                        i += 1
                        continue
            new_route.append(node)
            i += 1

        if len(new_route) >= 2:
            rs[r_idx] = new_route

    return rs


def tournament_select(
    population: list[RouteSet],
    fitnesses: list[float],
    n_select: int,
    m: int,
    rng: random.Random,
) -> tuple[list[RouteSet], list[float]]:
    """
    Tournament selection (Section 3.3): group all N×m strings into batches
    of 2m, pick the highest-fitness string from each batch, keep 2 copies
    → total N strings for next generation. Returns (selected, fitnesses).
    """
    combined = list(zip(population, fitnesses))
    rng.shuffle(combined)

    selected: list[RouteSet] = []
    sel_fits: list[float] = []
    batch_size = 2 * m

    for start in range(0, len(combined), batch_size):
        group = combined[start: start + batch_size]
        if not group:
            break
        winner_rs, winner_fit = max(group, key=lambda x: x[1])
        # Two copies of the winner (as per Section 3.3)
        selected.append([r[:] for r in winner_rs])
        selected.append([r[:] for r in winner_rs])
        sel_fits.extend([winner_fit, winner_fit])

    return selected[:n_select], sel_fits[:n_select]


# ── Main ──────────────────────────────────────────────────────────────────────

def _save_checkpoint(gen: int, route_set: RouteSet, totfit: float, metrics: dict) -> None:
    """
    Write three files atomically at every checkpoint:
      checkpoints/gen_NNNN.json      — route set snapshot
      checkpoints/gen_NNNN_map.html  — interactive map of that snapshot
      best_routes.json               — always reflects the latest best (survives Ctrl+C)
    """
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    payload = {
        "generation":  gen,
        "best_TOTFIT": totfit,
        "metrics":     metrics,
        "routes":      route_set,
    }

    json_path = ckpt_dir / f"gen_{gen:04d}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    map_path = ckpt_dir / f"gen_{gen:04d}_map.html"
    _save_route_map(route_set, metrics, totfit, map_path, f"checkpoint gen {gen}")

    # Keep best_routes.json up-to-date so it's always valid even if the run is interrupted
    best_path = Path("best_routes.json")
    with open(best_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  [checkpoint] {json_path}  {map_path}  {best_path}")


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

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

    print("Loading existing routes for population seeding …")
    existing_routes = load_existing_routes(gd.id_to_idx)
    print(f"  Parsed {len(existing_routes)} existing route directions")

    print(f"Generating {N_POP-1} random initial route sets via IRSG …")
    t0 = time.time()
    population: list[RouteSet] = irsg(gd, activity, rng, N_POP - 1)
    seeded = seed_from_existing(existing_routes, rng)
    population.insert(0, seeded)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Evaluating initial population …")
    fitnesses: list[float] = []
    best_fit = -np.inf
    best_rs: RouteSet = population[0]
    best_metrics: dict = {}

    for i, rs in enumerate(population):
        fit, metrics = eval_route_set(rs, gd.demand, t_min, gd)
        fitnesses.append(fit)
        if fit > best_fit:
            best_fit = fit
            best_rs = rs
            best_metrics = metrics
        label = "seeded" if i == 0 else f"IRSG-{i}"
        print(f"  [{label}] TOTFIT={fit:.3f}  d0={metrics['d0p']:.1f}%  "
              f"d1={metrics['d1p']:.1f}%  ATT={metrics['ATT']:.2f}")

    log_path = Path("optimization_log.csv")
    log_fields = ["generation", "best_TOTFIT", "d0p", "d1p", "d2p", "dunp",
                  "ATT", "F1", "F2", "F3"]

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        # ── GA Main Loop ──────────────────────────────────────────────────────
        for gen in range(MAX_GENERATIONS):
            t0 = time.time()

            # 1. Expand: M_COPIES deep copies of each string
            expanded: list[RouteSet] = [
                copy.deepcopy(rs)
                for rs in population
                for _ in range(M_COPIES)
            ]

            # 2. Intra-string crossover on every copy
            expanded = [crossover_intra(rs, rng) for rs in expanded]

            # 3. Inter-string crossover between adjacent pairs (prob = CROSSOVER_PROB)
            for i in range(0, len(expanded) - 1, 2):
                if rng.random() < CROSSOVER_PROB:
                    expanded[i], expanded[i + 1] = crossover_inter(
                        expanded[i], expanded[i + 1], rng
                    )

            # 4. Mutation
            expanded = [mutate(rs, gd, rng) for rs in expanded]

            # 5. Evaluate expanded pool
            exp_fitnesses: list[float] = []
            exp_best_metrics: dict = {}
            for rs in expanded:
                fit, metrics = eval_route_set(rs, gd.demand, t_min, gd)
                exp_fitnesses.append(fit)
                if fit > best_fit:
                    best_fit = fit
                    best_rs = [r[:] for r in rs]
                    best_metrics = metrics

            # 6. Tournament selection → next generation (reuse already-computed fitnesses)
            population, fitnesses = tournament_select(
                expanded, exp_fitnesses, N_POP, M_COPIES, rng
            )

            # Best of this generation (from expanded pool)
            gen_best_idx = int(np.argmax(exp_fitnesses))
            gen_best_fit = exp_fitnesses[gen_best_idx]
            _, gen_metrics = eval_route_set(expanded[gen_best_idx], gd.demand, t_min, gd)

            elapsed = time.time() - t0
            m = gen_metrics
            print(
                f"Gen {gen+1:3d}/{MAX_GENERATIONS} | "
                f"TOTFIT={gen_best_fit:.4f} | "
                f"d0={m['d0p']:.1f}% d1={m['d1p']:.1f}% d2={m['d2p']:.1f}% "
                f"un={m['dunp']:.1f}% | ATT={m['ATT']:.2f} | {elapsed:.1f}s"
            )

            writer.writerow({
                "generation": gen + 1,
                "best_TOTFIT": gen_best_fit,
                "d0p": m["d0p"],
                "d1p": m["d1p"],
                "d2p": m["d2p"],
                "dunp": m["dunp"],
                "ATT": m["ATT"],
                "F1": m["F1"],
                "F2": m["F2"],
                "F3": m["F3"],
            })
            log_file.flush()
            _save_plot(log_path)

            # ── Periodic route checkpoint ─────────────────────────────────────
            if CHECKPOINT_EVERY > 0 and (gen + 1) % CHECKPOINT_EVERY == 0:
                _save_checkpoint(gen + 1, best_rs, best_fit, best_metrics)

    # ── Save final best route set ─────────────────────────────────────────────
    output = {
        "n_routes":    len(best_rs),
        "best_TOTFIT": best_fit,
        "metrics":     best_metrics,
        "routes":      best_rs,
    }
    out_path = Path("best_routes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    map_path = DATA_DIR / "routes_map.html"
    _save_route_map(best_rs, best_metrics, best_fit, map_path, "final best")

    print(f"\nOptimisation complete.")
    print(f"  Best TOTFIT : {best_fit:.4f}")
    print(f"  d0={best_metrics.get('d0p',0):.1f}%  "
          f"d1={best_metrics.get('d1p',0):.1f}%  "
          f"d2={best_metrics.get('d2p',0):.1f}%  "
          f"un={best_metrics.get('dunp',0):.1f}%")
    print(f"  ATT         : {best_metrics.get('ATT', 0):.2f}")
    print(f"Routes saved → {out_path}")
    print(f"Map    saved → {map_path}")
    print(f"Log    saved → {log_path}")


if __name__ == "__main__":
    main()
