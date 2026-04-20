"""
Build a gravity-model demand matrix for Yerevan bus transit nodes.

demand[i][j] = population[i] * activities[j] / dist_km(i, j) ** BETA   (i != j)
demand[i][i] = 0

population[i]  = sum of residential building populations within RADIUS_M of stop i
activities[j]  = count of OSM activity POIs within RADIUS_M of stop j
dist_km(i, j)  = Haversine distance in km between stops i and j

Dependencies: numpy, Pillow (PIL) — stdlib json, csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
RADIUS_M: Final[float] = 400.0   # pedestrian catchment radius in metres
BETA: Final[float] = 2.0          # gravity decay exponent
MIN_DIST_KM: Final[float] = 0.1  # minimum distance floor in km (prevents 1/dist² blow-up for co-located stops)
LOG_TRANSFORM: Final[bool] = True # apply log1p after gravity model
NORMALIZE: Final[bool] = False    # if True, scale final matrix to [0, 1]
BATCH_SIZE: Final[int] = 64       # stops processed per batch during aggregation

DATA_DIR = Path(__file__).parent.parent / "data"
TRANSIT_GRAPH_PATH = DATA_DIR / "transitGraph.json"
BUILDINGS_PATH     = DATA_DIR / "building_population.json"
ACTIVITIES_PATH    = DATA_DIR / "activities.json"

OUT_NPY        = DATA_DIR / "demand_matrix.npy"
OUT_INDEX_JSON = DATA_DIR / "node_index.json"
OUT_CSV        = DATA_DIR / "demand_matrix.csv"
OUT_JSON       = DATA_DIR / "demand_matrix.json"
OUT_PNG        = DATA_DIR / "demand_matrix.png"

_EARTH_R_KM: Final[float] = 6_371.0   # km
_RADIUS_KM:  Final[float] = RADIUS_M / 1_000.0


# ---------------------------------------------------------------------------
# Haversine helpers
# ---------------------------------------------------------------------------

def _haversine_batch(
    lat_a: np.ndarray,  # shape (A,) or (A, 1)  radians
    lng_a: np.ndarray,  # shape (A,) or (A, 1)
    lat_b: np.ndarray,  # shape (B,)             radians
    lng_b: np.ndarray,  # shape (B,)
) -> np.ndarray:
    """Return (A, B) distance matrix in km via Haversine formula."""
    lat_a = np.asarray(lat_a)[:, None]
    lng_a = np.asarray(lng_a)[:, None]
    lat_b = np.asarray(lat_b)[None, :]
    lng_b = np.asarray(lng_b)[None, :]

    dlat = lat_a - lat_b
    dlng = lng_a - lng_b
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat_a) * np.cos(lat_b) * np.sin(dlng / 2.0) ** 2
    )
    return 2.0 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _haversine_square(lats: np.ndarray, lngs: np.ndarray) -> np.ndarray:
    """Return (N, N) symmetric Haversine distance matrix in km."""
    lat_r = np.radians(lats)
    lng_r = np.radians(lngs)
    return _haversine_batch(lat_r, lng_r, lat_r, lng_r)


# ---------------------------------------------------------------------------
# Spatial aggregation  (batched to avoid peak RAM > ~200 MB)
# ---------------------------------------------------------------------------

def aggregate_to_stops(
    stop_lats_deg: np.ndarray,
    stop_lngs_deg: np.ndarray,
    point_lats_deg: np.ndarray,
    point_lons_deg: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    For each stop sum `weights` (or count) of points within RADIUS_M.
    Processes stops in batches of BATCH_SIZE to limit memory.
    """
    n_stops = len(stop_lats_deg)
    result = np.zeros(n_stops, dtype=np.float64)

    slat_r = np.radians(stop_lats_deg)
    slng_r = np.radians(stop_lngs_deg)
    plat_r = np.radians(point_lats_deg)
    plon_r = np.radians(point_lons_deg)

    for start in range(0, n_stops, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_stops)
        # dist shape: (batch, n_points)
        dist = _haversine_batch(slat_r[start:end], slng_r[start:end], plat_r, plon_r)
        mask = dist <= _RADIUS_KM          # (batch, n_points) bool
        if weights is not None:
            result[start:end] = mask.dot(weights)
        else:
            result[start:end] = mask.sum(axis=1).astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_transit_graph(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with path.open() as f:
        graph = json.load(f)
    nodes = graph["nodes"]
    node_ids = list(nodes.keys())
    lats = np.array([nodes[nid]["coords"]["lat"] for nid in node_ids], dtype=np.float64)
    lngs = np.array([nodes[nid]["coords"]["lng"] for nid in node_ids], dtype=np.float64)
    return node_ids, lats, lngs


def load_buildings(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open() as f:
        buildings = json.load(f)
    lats = np.array([float(b["lat"])        for b in buildings], dtype=np.float64)
    lons = np.array([float(b["lon"])        for b in buildings], dtype=np.float64)
    pops = np.array([float(b["population"]) for b in buildings], dtype=np.float64)
    return lats, lons, pops


def load_activities(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        pois = json.load(f)
    # Some entries are OSM way-stubs {"type": "way"} with no coordinates — skip them.
    node_pois = [p for p in pois if "lat" in p and "lon" in p]
    lats = np.array([float(p["lat"]) for p in node_pois], dtype=np.float64)
    lons = np.array([float(p["lon"]) for p in node_pois], dtype=np.float64)
    return lats, lons


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_csv(demand: np.ndarray, node_ids: list[str], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + node_ids)
        for i, nid in enumerate(node_ids):
            writer.writerow([nid] + [f"{v:.6g}" for v in demand[i]])


def _write_sparse_json(demand: np.ndarray, node_ids: list[str], path: Path) -> None:
    sparse: dict[str, dict[str, float]] = {}
    for i, nid_i in enumerate(node_ids):
        row: dict[str, float] = {}
        for j, nid_j in enumerate(node_ids):
            v = demand[i, j]
            if v > 0.0:
                row[nid_j] = round(float(v), 6)
        if row:
            sparse[nid_i] = row
    with path.open("w") as f:
        json.dump(sparse, f, indent=2)


# Plasma colormap: 5 control points mapped to [0, 64, 128, 192, 255]
_PLASMA_STOPS = np.array([
    [13,   8,  135],   # 0.00 — deep purple
    [126,  3,  168],   # 0.25 — violet
    [203,  71,  119],  # 0.50 — rose
    [248, 149,   64],  # 0.75 — orange
    [240, 249,  33],   # 1.00 — yellow
], dtype=np.float32)


def _apply_plasma(norm: np.ndarray) -> np.ndarray:
    """
    Map a float32 array in [0, 1] to uint8 RGB via plasma colormap.
    norm shape: (H, W) → output shape: (H, W, 3).
    """
    n_seg = len(_PLASMA_STOPS) - 1
    scaled = norm * n_seg                       # [0, n_seg]
    seg = np.clip(scaled.astype(np.int32), 0, n_seg - 1)
    frac = (scaled - seg)[..., None]           # fractional part, shape (H, W, 1)

    c0 = _PLASMA_STOPS[seg]                    # (H, W, 3)
    c1 = _PLASMA_STOPS[np.minimum(seg + 1, n_seg)]
    rgb = c0 + frac * (c1 - c0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _write_heatmap(
    demand: np.ndarray,
    stop_lats: np.ndarray,
    stop_lngs: np.ndarray,
    path: Path,
) -> None:
    """
    Save a log-scaled plasma heatmap of the demand matrix.

    Nodes are sorted north→south (descending lat) on both axes so that
    the image has geographic meaning: top-left = northernmost origin,
    top-left corner = north-to-north demand.
    """
    # Sort by latitude descending (north → south)
    geo_order = np.argsort(-stop_lats)
    mat = demand[np.ix_(geo_order, geo_order)]

    # Log1p scale — compresses the huge dynamic range
    log_mat = np.log1p(mat)

    max_val = log_mat.max()
    if max_val > 0.0:
        norm = (log_mat / max_val).astype(np.float32)
    else:
        norm = log_mat.astype(np.float32)

    rgb = _apply_plasma(norm)                  # (N, N, 3)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(str(path))
    print(f"  {path}  ({img.width}×{img.height} px, log1p-scaled plasma)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_demand_matrix() -> None:
    print("Loading transit graph …")
    node_ids, stop_lats, stop_lngs = load_transit_graph(TRANSIT_GRAPH_PATH)
    n = len(node_ids)
    print(f"  {n} nodes loaded.")

    print("Loading buildings …")
    b_lats, b_lons, b_pops = load_buildings(BUILDINGS_PATH)
    print(f"  {len(b_lats)} buildings, total population {int(b_pops.sum()):,}")

    print("Loading activity POIs …")
    a_lats, a_lons = load_activities(ACTIVITIES_PATH)
    print(f"  {len(a_lats)} POI nodes loaded.")

    print(f"Aggregating populations within {RADIUS_M:.0f} m of each stop …")
    pop_vec = aggregate_to_stops(stop_lats, stop_lngs, b_lats, b_lons, weights=b_pops)
    print(f"  Stops with population > 0: {(pop_vec > 0).sum()} / {n}")

    print(f"Aggregating activity counts within {RADIUS_M:.0f} m of each stop …")
    act_vec = aggregate_to_stops(stop_lats, stop_lngs, a_lats, a_lons, weights=None)
    print(f"  Stops with activities > 0: {(act_vec > 0).sum()} / {n}")

    print("Computing Haversine distance matrix …")
    dist_km = _haversine_square(stop_lats, stop_lngs)
    # Diagonal set to inf so division yields 0; restored explicitly after.
    np.fill_diagonal(dist_km, np.inf)
    # Clamp off-diagonal distances to MIN_DIST_KM to prevent 1/dist² blow-up
    # for co-located or near-duplicate stops (inf on diagonal is unaffected).
    np.maximum(dist_km, MIN_DIST_KM, out=dist_km)
    print(f"  Min distance floor applied: {MIN_DIST_KM} km")

    print(f"Applying gravity model (beta={BETA}) …")
    numerator = np.outer(pop_vec, act_vec)         # (N, N)
    demand = numerator / (dist_km ** BETA)
    np.fill_diagonal(demand, 0.0)

    if LOG_TRANSFORM:
        demand = np.log1p(demand)
        print("  log1p transform applied.")

    if NORMALIZE:
        max_val = float(demand.max())
        if max_val > 0.0:
            demand /= max_val
        print("  Matrix normalised to [0, 1].")

    nonzero = demand[demand > 0]
    print("\nMatrix stats:")
    print(f"  shape       : {demand.shape}")
    print(f"  non-zero    : {len(nonzero):,}  ({100 * len(nonzero) / demand.size:.1f}%)")
    if len(nonzero):
        print(f"  min  (>0)   : {nonzero.min():.4g}")
        print(f"  p25         : {float(np.percentile(nonzero, 25)):.4g}")
        print(f"  median      : {float(np.median(nonzero)):.4g}")
        print(f"  p75         : {float(np.percentile(nonzero, 75)):.4g}")
        print(f"  p99         : {float(np.percentile(nonzero, 99)):.4g}")
    print(f"  max         : {demand.max():.4g}")
    print(f"  mean (all)  : {demand.mean():.4g}")
    if len(nonzero) > 1:
        print(f"  max/min ratio (>0): {nonzero.max()/nonzero.min():.2g}")

    print("\nSaving outputs …")

    np.save(str(OUT_NPY), demand)
    print(f"  {OUT_NPY}")

    node_index = {nid: idx for idx, nid in enumerate(node_ids)}
    with OUT_INDEX_JSON.open("w") as f:
        json.dump(node_index, f, indent=2)
    print(f"  {OUT_INDEX_JSON}")

    _write_csv(demand, node_ids, OUT_CSV)
    print(f"  {OUT_CSV}")

    _write_sparse_json(demand, node_ids, OUT_JSON)
    print(f"  {OUT_JSON}")

    _write_heatmap(demand, stop_lats, stop_lngs, OUT_PNG)

    print("\nDone.")


if __name__ == "__main__":
    build_demand_matrix()
