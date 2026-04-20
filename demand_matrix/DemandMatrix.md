# Yerevan Bus Transit — Demand Matrix

Capstone project dataset and preprocessing pipeline for Yerevan municipal bus network.
The goal is to compute an OD (origin–destination) demand matrix across all bus stops,
which will be used as input for a transit optimization algorithm.

---

## Repository Contents

| File | Size | Description |
|---|---|---|
| `data/transitGraph.json` | 379 KB | Bus stop nodes + directed edges (the transit network graph) |
| `data/building_population.json` | 5.5 MB | Residential buildings with estimated population |
| `data/activities.json` | 1.5 MB | OSM POI nodes representing activity destinations |
| `data/demand_matrix.npy` | 9.3 MB | Dense float64 demand matrix, shape (1387, 1387) |
| `data/demand_matrix.json` | 35 MB | Same matrix, sparse JSON (zero entries omitted), indented |
| `data/demand_matrix.csv` | 8.9 MB | Same matrix as CSV, rows/cols labelled by node ID |
| `data/node_index.json` | 23 KB | `{ nodeId: row_index }` mapping for the matrix |
| `data/demand_matrix.png` | 2.2 MB | Plasma heatmap of the demand matrix |
| `demand_matrix/build_demand_matrix.py` | — | Main pipeline script |
| `demand_matrix/visualize_demand.py` | — | Matplotlib heatmap visualization of the demand matrix |

Paths above are relative to the capstone project root.

---

## Data Sources

### `data/transitGraph.json`
Scraped dataset of **Yerevan bus stops and routes**, collected from the city's public
transit data. Built by `transit_graph/build_transit_graph.py`. Nodes represent physical
bus stops; edges represent direct route connections between stops.

```
nodes: 1,387 bus stops
edges: 6,375 directed connections (multigraph)
```

Each node:
```json
{
  "id": "1733309221",
  "name": "Arshakunyats Avenue, 59",
  "coords": { "lng": 44.497831354, "lat": 40.14751156 },
  "activityLevel": 2,
  "lines": [14, 33]
}
```

Each edge entry maps a source node ID to a list of outgoing segment objects.
`weight` is the real distance in km between the two stops.

---

### `data/building_population.json`
Residential building footprints collected from **OpenStreetMap** via Overpass API,
covering central and suburban Yerevan.

```
buildings: 41,635
estimated total population: 1,526,121
```

Each entry:
```json
{
  "id": 31662732,
  "lat": "40.179538",
  "lon": "44.516315",
  "area_sqm": 424,
  "floors": 4,
  "population": 58
}
```

**Population estimation formula** (derived from the dataset — consistent to ±0.1%):

```
population = round(area_sqm * floors / 29.34)
```

29.34 m²/person reflects Soviet-era housing density norms typical for Yerevan.
`area_sqm` is the building footprint area (ground floor only); gross floor area
is `area_sqm * floors`.

---

### `data/activities.json`
OSM **POI (Point of Interest) nodes** — shops, offices, schools, amenities,
leisure facilities — representing activity destinations that attract bus trips.

```
total entries: 20,783
valid (with coordinates): 19,554
invalid (OSM way-stubs, type="way", no coords): 1,229  ← filtered out
```

Each valid entry:
```json
{ "type": "node", "lat": 40.1767756, "lon": 44.5124699 }
```

**Remaining limitation:** 48 stops in peripheral villages (Kasakh, Proshyan,
Musaler, Kharberd outskirts, etc.) still have zero activity POIs within 400 m.
These appear as zero columns in the demand matrix, meaning no demand is directed
*toward* those stops. This reflects genuine low-activity suburban zones in OSM,
not a collection error.

---

## Demand Matrix

### Formula

The pipeline applies three sequential steps:

**Step 1 — Gravity model (raw)**
```
raw[i][j] = population[i] * activities[j] / max(dist_km(i,j), MIN_DIST_KM) ^ beta    (i ≠ j)
raw[i][i] = 0
```

**Step 2 — Minimum distance floor**

Before dividing, all pairwise distances are clamped to `MIN_DIST_KM = 0.1 km`.
This prevents `1/dist²` blow-up for co-located or near-duplicate stops
(e.g. two directions of the same stop 30 m apart on Mashtots Ave.).

**Step 3 — log1p transform**
```
demand[i][j] = log(1 + raw[i][j])
```

Compresses the power-law distributed gravity model output into a near-normal
distribution. Preserves relative ordering and is trivially invertible
(`expm1`). The diagonal stays 0 since `log1p(0) = 0`.

| Parameter | Value | Meaning |
|---|---|---|
| `population[i]` | sum of building populations within 400 m of stop i | trip origin supply |
| `activities[j]` | count of POI nodes within 400 m of stop j | trip destination attractiveness |
| `dist_km(i, j)` | Haversine distance in km between stops i and j | travel cost |
| `MIN_DIST_KM` | 0.1 km | minimum distance floor to cap `1/dist²` |
| `BETA` | 2.0 | gravity decay exponent (standard for urban transit) |
| `RADIUS_M` | 400 m | pedestrian catchment radius (standard walking distance) |
| `LOG_TRANSFORM` | True | apply log1p after gravity model |

This is a **singly-unconstrained gravity model**. Row and column totals are not
balanced — `demand[i][j]` and `demand[j][i]` are generally not equal.

### Statistics (after min-distance floor + log1p)

| Metric | Value |
|---|---|
| Shape | 1387 × 1387 |
| Total cells | 1,923,769 |
| Non-zero cells | ~95% |
| Min (> 0) | 0.052 |
| p25 | 6.57 |
| Median | 8.33 |
| p75 | 9.96 |
| p99 | 14.39 |
| Max | 21.82 |
| Mean (non-zero) | 7.90 |
| max/min ratio | ~420 |

Before the min-distance floor and log1p were applied, the raw gravity model
produced values in [0.05, 3.46 × 10¹⁰] — a ratio of ~10¹¹. The two-step
normalisation brings this down to ~420×, with the interquartile range
(6.6–10.0) tightly clustered.

### Remaining limitation

**48 zero-column stops** — peripheral destinations (Kasakh, Proshyan, Musaler,
Kharberd outskirts, etc.) have no OSM POI nodes within 400 m. No demand is
directed *toward* these stops. This reflects genuine low-activity suburban zones
in OSM, not a collection error. Fixable by re-fetching activity data for those
coordinates or applying a minimum activity floor of 1.

---

## Reproducing the Outputs

### Requirements

```
Python 3.10+
numpy       (apt: python3-numpy)
Pillow      (apt: python3-pil)
matplotlib  (apt: python3-matplotlib)  # for visualize_demand.py only
```

### Pipeline

```bash
# Run from the capstone project root

# 1. Build transit graph (prerequisite)
python3 transit_graph/build_transit_graph.py

# 2. Build demand matrix and plasma heatmap
python3 demand_matrix/build_demand_matrix.py

# 3. (Optional) Render matplotlib heatmap
python3 demand_matrix/visualize_demand.py
```

Outputs written to `data/`:
- `demand_matrix.npy` — NumPy binary, load with `numpy.load("data/demand_matrix.npy")`
- `node_index.json` — maps each OSM node ID string to its matrix row/column index
- `demand_matrix.csv` — human-readable, rows and columns labelled by node ID
- `demand_matrix.json` — sparse dict, zero entries omitted
- `demand_matrix.png` — heatmap, log1p-scaled, plasma colormap, nodes sorted north→south

### Tunable parameters (top of `demand_matrix/build_demand_matrix.py`)

```python
RADIUS_M      = 400.0   # catchment radius in metres
BETA          = 2.0     # gravity decay exponent
MIN_DIST_KM   = 0.1     # minimum distance floor in km
LOG_TRANSFORM = True    # apply log1p after gravity model
NORMALIZE     = False   # set True to additionally scale matrix to [0, 1]
```

### Loading the matrix in Python

```python
import json
import numpy as np

demand = np.load("data/demand_matrix.npy")       # shape (1387, 1387), log1p-scaled

with open("data/node_index.json") as f:
    node_index = json.load(f)                    # { "nodeId": row_int }

row_to_node = {v: k for k, v in node_index.items()}

# Demand from stop "1733309221" to all others
i = node_index["1733309221"]
outgoing = demand[i]                             # shape (1387,)

# To recover raw gravity model values:
raw = np.expm1(demand)
```
