#!/usr/bin/env python3
"""
GA-based Transit Route Network Designer
Implements: Chakroborty & Dwivedi (2002), Engineering Optimization 34(1), 83-100.

Inputs  (all under data/):
  transitGraph.json            1387 nodes, 6375 directed multigraph edges
  demand_matrix.npy            (1387, 1387) log1p-scaled OD demand
  node_index.json              nodeId -> matrix row/col index
  allYerevanTransportLines.json  74 existing routes (optional seed for one population member)

CLI (see --help): --output-dir, --seed, --no-seed-existing.

Outputs (under OUTPUT_DIR, overridable with --output-dir):
  best_routes.json             optimised route set (list of stop-ID sequences)
  optimization_log.csv         per-generation metrics
  optimization_progress.png    optional chart from plot_results
  checkpoints/                 periodic snapshots + maps
  routes_map.html              final interactive map
"""

from __future__ import annotations

import argparse
import copy
import csv
import heapq
import json
import math
import random
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np


class OptimizerCliArgs(NamedTuple):
    """Parsed CLI options for a single optimisation run."""

    output_dir: Path
    seed: int
    seed_with_existing_routes: bool

# Optional live plotting — renders optimization_progress.png after each generation
try:
    from plot_results import render as _render_plot

    def _save_plot(log_path: Path) -> None:
        try:
            plot_out = log_path.with_name("optimization_progress.png")
            _render_plot(log_path, plot_out)
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
# All run artefacts (log, checkpoints, best JSON, plots, maps). Override via --output-dir.
OUTPUT_DIR      = Path(".")

R               = 74       # routes per route set
N_POP           = 20       # population size
ELITE_COUNT     = 2        # top individuals carried unchanged; must be < N_POP
M_COPIES        = 5        # copies per string in MODIFY (pool size = N_POP × M_COPIES)
K_INS           = 500      # INS cardinality (top-K most-active nodes)
MAX_NODES_ROUTE = 63       # max stops per route (M)
MIN_NODES_ROUTE = 20       # min stops per route (M)

MAX_LEN_ROUTE    = 55      # max route length in km (L)
U_TRANSFER       = 5.0     # transfer penalty (same units as edge weights, km here)
MAX_GENERATIONS  = 1000    # GA iterations
CROSSOVER_PROB   = 0.5     # inter-string crossover probability
MUTATION_PROB    = 0    # per-node mutation probability
CHECKPOINT_EVERY = 10      # save routes snapshot every N generations (0 = disabled)

# Minimum great-circle distance between any two distinct stops on ONE route [km].
# 0.1 km == 100 m. Applied in good_route(...): two different nodes closer than this
# cannot appear on the same line (distinct IDs only — duplicate IDs are rejected separately).
# Note: Maps often draw straight chords between consecutive stops; those segments can cross
# without visiting an intermediate stop (graph path vs straight-line geometry).
MIN_PAIRWISE_STOP_SEP_KM = 0.1

# If True, reject a route when any two non-consecutive chord segments (straight lines between
# consecutive stops in map order) intersect in the (lng, lat) plane — same convention as typical
# Leaflet polylines over a small city extent. Stops may all differ and be ≥100 m apart but chords
# can still cross (e.g. opposite sides of an interchange).
REJECT_POLYLINE_SELF_INTERSECTION = True

# Fitness weights (ω1, ω2, ω3)
OMEGA           = (1.0, 1.0, 1.0)
# Max-score constants for F1, F2, F3
K1 = K2 = K3   = 10.0
# Demand-satisfaction weights for dT (α ≥ β ≥ γ)
ALPHA, BETA_W, GAMMA = 1.0, 0.5, 0.25
# Acceptable upper limit of IVT excess over T_min (km)
X_M             = 15.0

RANDOM_SEED     = 42

haversine_distance = {}

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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # distance in km

def calculate_haversine_of_all_nodes(gd: GraphData) -> None:
    for node1 in gd.nodes:
        for node2 in gd.nodes:
            if node1 != node2:
                haversine_distance[(node1, node2)] = haversine(gd.nodes[node1]['coords']['lat'], 
                gd.nodes[node1]['coords']['lng'], 
                gd.nodes[node2]['coords']['lat'],
                gd.nodes[node2]['coords']['lng'])

def _node_lng_lat(node_id: str, gd: GraphData) -> tuple[float, float]:
    c = gd.nodes[node_id]["coords"]
    return (float(c["lng"]), float(c["lat"]))


_SEG_INTER_PAR_EPS = 1e-15  # parallel denominator threshold (degrees²-scale)
_SEG_INTER_T_EPS = 1e-12    # parametric intersection tolerance along segments


def _collinear_seg_overlap_on_ab(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    """Return True if segments AB and CD overlap on the same line with positive length."""
    dx_ab = bx - ax
    dy_ab = by - ay
    len2 = dx_ab * dx_ab + dy_ab * dy_ab
    if len2 < 1e-22:
        return False

    def dot_from_a(px: float, py: float) -> float:
        return (px - ax) * dx_ab + (py - ay) * dy_ab

    t1_lo, t1_hi = 0.0, len2
    tc, td = dot_from_a(cx, cy), dot_from_a(dx, dy)
    t2_lo, t2_hi = min(tc, td), max(tc, td)
    overlap = min(t1_hi, t2_hi) - max(t1_lo, t2_lo)
    scale = math.sqrt(len2)
    return overlap > max(1e-9 * scale, 1e-12)


def _chord_segments_intersect(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    """
    True iff closed segments AB and CD intersect (including collinear overlap).
    Uses segment parameters t, u in [0,1] for P=A+t(B-A), Q=C+u(D-C).
    """
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) < _SEG_INTER_PAR_EPS:
        scale_ab = max(math.hypot(rx, ry), 1e-15)
        cross_c = rx * (cy - ay) - ry * (cx - ax)
        cross_d = rx * (dy - ay) - ry * (dx - ax)
        tol = 1e-9 * scale_ab
        if abs(cross_c) > tol or abs(cross_d) > tol:
            return False
        return _collinear_seg_overlap_on_ab(ax, ay, bx, by, cx, cy, dx, dy)

    t = ((cx - ax) * sy - (cy - ay) * sx) / denom
    u = ((cx - ax) * ry - (cy - ay) * rx) / denom
    return (
        -_SEG_INTER_T_EPS <= t <= 1.0 + _SEG_INTER_T_EPS
        and -_SEG_INTER_T_EPS <= u <= 1.0 + _SEG_INTER_T_EPS
    )


def _route_polyline_self_intersects(route: Route, gd: GraphData) -> bool:
    """
    True if the polyline connecting consecutive stops has a self-intersection
    (two non-consecutive edges crossing or overlapping in lng–lat space).
    """
    n = len(route)
    if n < 4:
        return False
    pts: list[tuple[float, float]] = [_node_lng_lat(nid, gd) for nid in route]
    for i in range(n - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        for j in range(i + 2, n - 1):
            cx, cy = pts[j]
            dx, dy = pts[j + 1]
            if _chord_segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
    return False


def good_route(route: Route, gd: GraphData) -> bool:
    """
    Feasibility for a single line (ordered stop list).

    - Each graph stop ID may appear at most once (no revisiting the same station).
    - Any two different stops must be at least MIN_PAIRWISE_STOP_SEP_KM apart
      (Haversine), so two distinct nodes within ~100 m cannot both be on the line.
    - Optionally: consecutive-stop chords may not self-intersect (see
      REJECT_POLYLINE_SELF_INTERSECTION).
    """
    if len(route) < MIN_NODES_ROUTE or len(route) > MAX_NODES_ROUTE:
        return False

    if len(route) != len(set(route)):
        return False

    n_route = len(route)
    for i in range(n_route):
        for j in range(i + 1, n_route):
            a, b = route[i], route[j]
            if haversine_distance[(a, b)] < MIN_PAIRWISE_STOP_SEP_KM:
                return False

    if REJECT_POLYLINE_SELF_INTERSECTION and _route_polyline_self_intersects(route, gd):
        return False

    return True

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
    return max(0, (cosine + 1.0) / 2.0)


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
            # print(d_k, a_k, d_k * a_k, sep="------")

        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        # print(*probs)
        # print("*"*20)

        # Check termination criteria before committing
        chosen_tgt, chosen_edge_w = rng.choices(vns, weights=probs, k=1)[0]

        if len(route) >= MAX_NODES_ROUTE:
            break
        if total_len + chosen_edge_w > MAX_LEN_ROUTE:
            break

        # if not good_route(route, gd):
         #   continue

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
        while len(route_set) < R:
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
            if not good_route(route, gd):
                ins_set.add(chosen_idx)
                continue
            # print(len(route_set))
            route_set.append(route)
    
        population.append(route_set)

    return population


def seed_from_existing(
    existing: list[Route],
    rng: random.Random,
    gd: GraphData,
) -> RouteSet:
    """
    Build one RouteSet by sampling R routes from parsed real lines.
    Only routes that pass good_route(...) are eligible (no duplicate stops / min spacing).
    """
    valid = [r for r in existing if good_route(r, gd)]
    if not valid:
        raise ValueError(
            "No existing route direction passes good_route (unique stops + min pairwise "
            f"separation {MIN_PAIRWISE_STOP_SEP_KM} km). Use --no-seed-existing or relax limits."
        )
    if len(valid) >= R:
        return rng.sample(valid, R)
    result = list(valid)
    pool = valid
    while len(result) < R:
        result.append(rng.choice(pool))
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

def crossover_intra(rs: RouteSet, rng: random.Random, gd: GraphData) -> RouteSet:
    """
    Intra-string crossover (Figure 5): pick two routes sharing a common node,
    exchange tails at that node. Returns a modified shallow copy.

    Splices are skipped if any distinct stop pair across the two splice branches
    (tail vs head on each side) is closer than MIN_PAIRWISE_STOP_SEP_KM.
    """
    rs = [r[:] for r in rs]
    if len(rs) < 2:
        return rs

    indices = list(range(len(rs)))
    rng.shuffle(indices)

    for attempt in range(40):
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

        if good_route(new_r1, gd) and good_route(new_r2, gd):
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

        original: Route = route[:]
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

        if len(new_route) >= 2 and good_route(new_route, gd):
            rs[r_idx] = new_route
        else:
            rs[r_idx] = original

    return rs


def tournament_select(
    population: list[RouteSet],
    fitnesses: list[float],
    n_select: int,
    m: int,
    rng: random.Random,
    exclude_indices: set[int] | None = None,
) -> tuple[list[RouteSet], list[float]]:
    """
    Tournament selection (Section 3.3): group all N×m strings into batches
    of 2m, pick the highest-fitness string from each batch, keep 2 copies
    → total N strings for next generation. Returns (selected, fitnesses).

    exclude_indices: pool indices to omit (e.g. elite slots already taken).
    """
    combined = [
        (population[i], fitnesses[i])
        for i in range(len(population))
        if exclude_indices is None or i not in exclude_indices
    ]
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

def _save_checkpoint(
    gen: int,
    route_set: RouteSet,
    totfit: float,
    metrics: dict,
    output_dir: Path,
    *,
    random_seed: int,
    seed_with_existing_routes: bool,
) -> None:
    """
    Write three files atomically at every checkpoint:
      {output_dir}/checkpoints/gen_NNNN.json      — route set snapshot
      {output_dir}/checkpoints/gen_NNNN_map.html  — interactive map of that snapshot
      {output_dir}/best_routes.json               — latest checkpoint payload (survives Ctrl+C)
    """
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generation":               gen,
        "best_TOTFIT":              totfit,
        "metrics":                  metrics,
        "routes":                   route_set,
        "random_seed":              random_seed,
        "seed_with_existing_routes": seed_with_existing_routes,
    }

    json_path = ckpt_dir / f"gen_{gen:04d}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    map_path = ckpt_dir / f"gen_{gen:04d}_map.html"
    _save_route_map(route_set, metrics, totfit, map_path, f"checkpoint gen {gen}")

    # Keep best_routes.json up-to-date so it's always valid even if the run is interrupted
    best_path = output_dir / "best_routes.json"
    with open(best_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  [checkpoint] {json_path}  {map_path}  {best_path}")


def _parse_args() -> OptimizerCliArgs:
    parser = argparse.ArgumentParser(
        description="GA-based transit route network optimizer.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--run-dir",
        type=Path,
        default=OUTPUT_DIR,
        metavar="DIR",
        help=(
            "Directory for checkpoints/, optimization_log.csv, best_routes.json, "
            "optimization_progress.png, and routes_map.html (created if missing). "
            f"Default: {OUTPUT_DIR!s}"
        ),
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=RANDOM_SEED,
        metavar="N",
        help=(
            "Seed for `random` and NumPy (`numpy.random`). "
            f"Default: {RANDOM_SEED}."
        ),
    )
    parser.add_argument(
        "--no-seed-existing",
        action="store_true",
        help=(
            "Do not prepend the real-world route network (from allYerevanTransportLines.json) "
            "as the first individual; initial population is entirely IRSG-generated."
        ),
    )
    ns = parser.parse_args()
    out_dir: Path = ns.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return OptimizerCliArgs(
        output_dir=out_dir,
        seed=ns.seed,
        seed_with_existing_routes=not ns.no_seed_existing,
    )


def main() -> None:
    cli = _parse_args()
    output_dir = cli.output_dir
    print(f"Writing results under: {output_dir}")
    print(f"Random seed: {cli.seed} (random + NumPy)")

    rng = random.Random(cli.seed)
    np.random.seed(cli.seed)

    if ELITE_COUNT > 0:
        if ELITE_COUNT >= N_POP:
            raise ValueError("ELITE_COUNT must be < N_POP")
        if (N_POP - ELITE_COUNT) % 2 != 0:
            raise ValueError("N_POP - ELITE_COUNT must be even (tournament emits winner pairs)")

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

    print("Calculating haversine of all nodes …")
    calculate_haversine_of_all_nodes(gd)

    if cli.seed_with_existing_routes:
        print("Loading existing routes for population seeding …")
        existing_routes = load_existing_routes(gd.id_to_idx)
        n_ok_existing = sum(1 for r in existing_routes if good_route(r, gd))
        print(
            f"  Parsed {len(existing_routes)} existing route directions; "
            f"{n_ok_existing} pass good_route (unique stops + ≥{MIN_PAIRWISE_STOP_SEP_KM} km apart)"
        )

        print(f"Generating {N_POP - 1} random initial route sets via IRSG …")
        t0 = time.time()
        population = irsg(gd, activity, rng, N_POP - 1)
        population.insert(0, seed_from_existing(existing_routes, rng, gd))
        print(f"  Done in {time.time()-t0:.1f}s")
    else:
        print("Initial population: IRSG only (--no-seed-existing).")
        t0 = time.time()
        population = irsg(gd, activity, rng, N_POP)
        print(f"  Generated {N_POP} route sets in {time.time()-t0:.1f}s")

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
        if cli.seed_with_existing_routes and i == 0:
            label = "seeded-existing"
        else:
            label = f"IRSG-{i}"
        _save_checkpoint(
            i,
            rs,
            fit,
            metrics,
            output_dir,
            random_seed=cli.seed,
            seed_with_existing_routes=cli.seed_with_existing_routes,
        )
        print(f"  [{label}] TOTFIT={fit:.3f}  d0={metrics['d0p']:.1f}%  "
              f"d1={metrics['d1p']:.1f}%  ATT={metrics['ATT']:.2f}")

    log_path = output_dir / "optimization_log.csv"
    log_fields = ["generation", "best_TOTFIT", "d0p", "d1p", "d2p", "dunp",
                  "ATT", "F1", "F2", "F3"]

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        # ── GA Main Loop ──────────────────────────────────────────────────────
        for gen in range(MAX_GENERATIONS):
            t0 = time.time()
            
            expanded: list[RouteSet] = []
            # 1. Expand: M_COPIES deep copies of each string
            for _ in range(M_COPIES):
                for rs in population:
                    expanded.append(copy.deepcopy(rs))

            # 2–4. Variation operators (skip copies of elite population slots 0..ELITE_COUNT-1)
            def _exp_is_elite_slot(i: int) -> bool:
                return ELITE_COUNT > 0 and (i % N_POP) < ELITE_COUNT

            expanded = [
                rs if _exp_is_elite_slot(i) else crossover_intra(rs, rng, gd)
                for i, rs in enumerate(expanded)
            ]

            for i in range(0, len(expanded) - 1, 2):
                if _exp_is_elite_slot(i) or _exp_is_elite_slot(i + 1):
                    continue
                if rng.random()< CROSSOVER_PROB:
                    expanded[i], expanded[i + 1] = crossover_inter(
                        expanded[i], expanded[i + 1], rng
                    )

            expanded = [
                rs if _exp_is_elite_slot(i) else mutate(rs, gd, rng)
                for i, rs in enumerate(expanded)
            ]

            # 5. Evaluate expanded pool
            exp_fitnesses: list[float] = []
            exp_metrics_list: list[dict] = []
            for rs in expanded:
                fit, metrics = eval_route_set(rs, gd.demand, t_min, gd)
                exp_fitnesses.append(fit)
                exp_metrics_list.append(metrics)
                if fit > best_fit:
                    best_fit = fit
                    best_rs = [r[:] for r in rs]
                    best_metrics = metrics

            # 6. Elitism + tournament → next generation
            if ELITE_COUNT <= 0:
                population, fitnesses = tournament_select(
                    expanded, exp_fitnesses, N_POP, M_COPIES, rng
                )
            else:
                order = np.argsort(exp_fitnesses)
                elite_indices = [int(order[-1 - k]) for k in range(ELITE_COUNT)]
                exclude_set = set(elite_indices)
                elites_rs = [copy.deepcopy(expanded[j]) for j in elite_indices]
                elites_fit = [exp_fitnesses[j] for j in elite_indices]

                rest, rest_fit = tournament_select(
                    expanded,
                    exp_fitnesses,
                    N_POP - ELITE_COUNT,
                    M_COPIES,
                    rng,
                    exclude_indices=exclude_set,
                )
                population = elites_rs + rest
                fitnesses = elites_fit + rest_fit

            # Best of this generation (from expanded pool)
            gen_best_idx = int(np.argmax(exp_fitnesses))
            gen_best_fit = exp_fitnesses[gen_best_idx]
            gen_metrics = exp_metrics_list[gen_best_idx]

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
                _save_checkpoint(
                    gen + 1,
                    best_rs,
                    best_fit,
                    best_metrics,
                    output_dir,
                    random_seed=cli.seed,
                    seed_with_existing_routes=cli.seed_with_existing_routes,
                )

    # ── Save final best route set ─────────────────────────────────────────────
    output = {
        "n_routes":                   len(best_rs),
        "best_TOTFIT":                best_fit,
        "metrics":                    best_metrics,
        "routes":                     best_rs,
        "random_seed":                cli.seed,
        "seed_with_existing_routes":  cli.seed_with_existing_routes,
    }
    out_path = output_dir / "best_routes.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    map_path = output_dir / "routes_map.html"
    _save_route_map(best_rs, best_metrics, best_fit, map_path, "final best")

    print(f"\nOptimisation complete.")
    print(f"  Output dir  : {output_dir}")
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
