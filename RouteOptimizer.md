# Route Optimizer — GA-Based Transit Route Network Design

Implementation of **Chakroborty & Dwivedi (2002)**, *"Optimal Route Network Design for Transit Systems Using Genetic Algorithms"*, Engineering Optimization 34(1), pp. 83–100.

Applied to the **Yerevan municipal bus network**: 1,387 stops, 74 routes (145 directions), 1,387 × 1,387 OD demand matrix.

---

## Files

| File | Role |
|---|---|
| `route_optimizer.py` | Main optimizer — run this |
| `plot_results.py` | Renders `optimization_progress.png` from the CSV log |
| `visualize_routes.py` | Renders an interactive Leaflet map from any route JSON |
| `data/transitGraph.json` | 1,387 nodes, 6,375 directed multigraph edges |
| `data/demand_matrix.npy` | (1387 × 1387) log1p-scaled OD demand matrix |
| `data/node_index.json` | `{ nodeId: row_index }` for matrix lookup |
| `data/allYerevanTransportLines.json` | 74 existing routes — seeds one population member |

---

## How It Works

The algorithm evolves a set of bus routes by iterating three steps:

```
IRSG → EVAL → [ MODIFY → EVAL ] × generations
```

### Step 1 — IRSG: Initial Route Set Generation

Generates `N_POP` starting route sets, each containing `R` routes.
One member is seeded directly from `allYerevanTransportLines.json` (the real Yerevan network as a baseline); the rest are built stochastically.

Each route is grown stop-by-stop:

**First stop** — sampled from the top-`K_INS` most-active stops. Activity `a_i` is the total demand involving stop `i` as origin or destination:

```
a[i] = demand[i, :].sum() + demand[:, i].sum()
p_j  = a_j / Σ a_k   (Eq. 1)
```

**Next stops** — the Vicinity Node Set (VNS) is the set of direct graph neighbours not already on the route. Next stop `k` is sampled as:

```
p_k ∝ d_k × a_k   (Eq. 2)
```

`d_k` is a **direction bias**: cosine similarity of the proposed step with the current travel direction, mapped to `[0.05, 1.0]` to penalise backtracking without fully forbidding it.

**Termination** — route stops growing when any of these is hit:
- stops on route ≥ `MAX_NODES_ROUTE`
- accumulated length ≥ `MAX_LEN_ROUTE` km
- VNS is empty

---

### Step 2 — EVAL: Route Set Fitness

Each route set is scored as a whole:

```
TOTFIT(r) = ω₁·F1(r) + ω₂·F2(r) + ω₃·F3(r)   (Eq. 3)
```

**Coverage** is computed using a binary stop-membership matrix `sor` of shape `(N_stops × R)`.
All matrix multiplies have inner dimension R = 70, so they are fast regardless of N_stops = 1,387:

| Coverage | Computation |
|---|---|
| Direct (d0) | `(sor @ sor.T) > 0` — both stops on the same route |
| 1 transfer (d1) | `sor @ route_overlap_offdiag @ sor.T > 0` — connected via a shared stop on two different routes |
| 2 transfers (d2) | `sor @ route_2hop @ sor.T > 0` — connected through two transfer points |
| Unsatisfied (dun) | demand not reachable within 2 transfers |

**F1 — travel time score (Eq. 4–5)**

```
x      = IVT(i,j) − T_min(i,j)          (excess over shortest path)
f(i,j) = K1 × (1 − x / x_m)²           if x ≤ x_m, else 0
F1     = Σ d(i,j)·f(i,j) / Σ d(i,j)    (demand-weighted average)
```

IVT is exact for d0 pairs (cumulative edge weights along route); approximated as `T_min + U` (d1) and `T_min + 2U` (d2). T_min is precomputed once via Dijkstra from all 1,387 nodes (~0.8 s).

**F2 — demand satisfaction score (Eq. 6–7)**

```
dT = α·d0 + β·d1 + γ·d2      (α ≥ β ≥ γ, higher weight on direct trips)
F2 = K2 × dT / (α + β + γ)
```

**F3 — unsatisfied demand penalty (Eq. 8)**

```
F3 = K3 × (1 − dun)²         (K3 when fully served, 0 when nothing is served)
```

---

### Step 3 — MODIFY: Genetic Operators

**Intra-string crossover (Figure 5)** — two routes within the same route set that share a common stop exchange their tails at that stop. Up to 20 random pairs are tried per string; only accepted when both children have ≥ 2 stops.

**Inter-string crossover (Figure 4)** — two different route sets swap route-suffix halves at a random demarcation line. Applied with probability `CROSSOVER_PROB`.

**Mutation (Section 3.3)** — each stop is independently replaced with a random adjacent neighbour at probability `MUTATION_PROB`. If the replacement breaks path continuity (no direct edge to the next stop), a short Dijkstra bridge is inserted (limited to 8 hops to prevent bloat).

**Reproduction — Tournament selection (Section 3.3)** — the expanded pool of `N_POP × M_COPIES` strings is partitioned into random groups of `2 × M_COPIES`. The highest-fitness string from each group is kept twice, discarding the rest. This produces exactly `N_POP` strings for the next generation.

---

## Usage

Run from the capstone project root (the directory that contains `data/`):

```bash
cd /home/samvel/Desktop/capstone
python3 route_optimizer.py
```

### Dependencies

```
Python 3.10+
numpy
Pillow      (for optimization_progress.png — plot_results.py)
```

No scipy, no folium, no external solvers. The route map uses Leaflet via CDN.

### Expected runtime

| Phase | Time |
|---|---|
| Data loading | ~1 s |
| T_min — all-pairs Dijkstra, 1387 nodes | ~0.8 s |
| IRSG — generate 20 initial route sets | ~0.1 s |
| Per EVAL call | ~0.25 s |
| Full run — 100 generations × 100 strings | ~40 min |

For a quick development check reduce `MAX_GENERATIONS`:

```python
MAX_GENERATIONS = 5    # smoke test (~3 min)
MAX_GENERATIONS = 20   # moderate (~8 min)
MAX_GENERATIONS = 100  # full run  (~40 min, default)
```

---

## Outputs

Every output is written to the capstone root unless noted:

| File | When written | Content |
|---|---|---|
| `optimization_log.csv` | After every generation | Per-generation metrics |
| `optimization_progress.png` | After every generation | 4-panel PNG progress chart |
| `checkpoints/gen_NNNN.json` | Every `CHECKPOINT_EVERY` generations | Best route set snapshot |
| `checkpoints/gen_NNNN_map.html` | Every `CHECKPOINT_EVERY` generations | Interactive Leaflet map of that snapshot |
| `best_routes.json` | End of run | Final best route set |
| `data/routes_map.html` | End of run | Interactive Leaflet map of the final routes |

### `best_routes.json` schema

```json
{
  "n_routes": 70,
  "best_TOTFIT": 12.34,
  "metrics": {
    "d0p":  18.5,
    "d1p":  47.2,
    "d2p":  15.1,
    "dunp": 19.2,
    "ATT":  14.3,
    "F1": 4.1, "F2": 7.3, "F3": 8.1,
    "TOTFIT": 19.5
  },
  "routes": [
    ["stopId_A", "stopId_B", "stopId_C", ...],
    ...
  ]
}
```

### `optimization_log.csv` columns

| Column | Meaning |
|---|---|
| `generation` | Generation number (1-indexed) |
| `best_TOTFIT` | Best fitness in this generation's expanded pool |
| `d0p` | % demand served directly (no transfer) |
| `d1p` | % demand served with 1 transfer |
| `d2p` | % demand served with 2 transfers |
| `dunp` | % demand unsatisfied |
| `ATT` | Demand-weighted average IVT (km) over all served pairs |
| `F1`, `F2`, `F3` | Individual fitness component scores |

### Console output

```
Loading data …
  Nodes: 1387 | Demand: (1387, 1387) | Edges (unique pairs): 1745
Computing activity levels …
  Activity range: [2680.5, 30858.2]
Computing T_min via Dijkstra (all-pairs shortest paths) …
  Done in 0.80s | Unreachable pairs: 221352
Loading existing routes for population seeding …
  Parsed 145 existing route directions
Generating 19 random initial route sets via IRSG …
Evaluating initial population …
  [seeded] TOTFIT=10.524  d0=6.9%  d1=43.2%  ATT=17.66
  [IRSG-1] TOTFIT=4.578   d0=2.2%  d1=10.6%  ATT=16.62
  ...
Gen   1/100 | TOTFIT=10.5140 | d0=7.0% d1=44.1% d2=16.8% un=32.1% | ATT=17.5 | 24.3s
  [checkpoint] checkpoints/gen_0010.json  checkpoints/gen_0010_map.html
Gen   2/100 | TOTFIT=10.6213 | ...
```

---

## Parameters

All set at the top of `route_optimizer.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `R` | `70` | Routes per route set |
| `N_POP` | `20` | Population size |
| `M_COPIES` | `5` | Copies per string (pool = N_POP × M_COPIES = 100) |
| `K_INS` | `200` | INS size — top-K nodes used for first-stop selection |
| `MAX_NODES_ROUTE` | `25` | Maximum stops per route |
| `MAX_LEN_ROUTE` | `30.0` | Maximum route length in km |
| `U_TRANSFER` | `5.0` | Transfer penalty (km-equivalent, added per transfer) |
| `MAX_GENERATIONS` | `100` | GA iterations |
| `CHECKPOINT_EVERY` | `10` | Save route snapshot every N generations (0 = off) |
| `CROSSOVER_PROB` | `0.5` | Inter-string crossover probability per pair |
| `MUTATION_PROB` | `0.01` | Per-stop mutation probability |
| `OMEGA` | `(1, 1, 1)` | Weights ω₁ ω₂ ω₃ for F1 F2 F3 |
| `K1`, `K2`, `K3` | `10.0` | Maximum score for each fitness component |
| `ALPHA`, `BETA_W`, `GAMMA` | `1.0, 0.5, 0.25` | Weights for d0/d1/d2 in dT |
| `X_M` | `15.0` | IVT excess ceiling for F1 (km) |
| `RANDOM_SEED` | `42` | RNG seed |

---

## Algorithm Flow

```
┌──────────────────────────────────────────────────────────────┐
│  Precompute (once, ~1 s total)                               │
│  • activity[i] = demand[i,:].sum() + demand[:,i].sum()       │
│  • T_min[i,j]  via Dijkstra from all 1,387 nodes            │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  IRSG — build N_POP initial route sets                       │
│  • 1 seeded from existing 74 Yerevan routes (baseline)       │
│  • N_POP-1 grown stochastically via Eq. 1 & 2               │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│  EVAL — score all N_POP route sets → fitnesses               │
└─────────────────────────┬────────────────────────────────────┘
                          │
         ┌────────────────▼──────────────────────┐
         │  for gen in range(MAX_GENERATIONS):   │◄──────┐
         │                                       │       │
         │  1. Expand  N_POP × M_COPIES copies   │       │
         │  2. Intra-string crossover (all)      │       │
         │  3. Inter-string crossover (pairs)    │       │
         │  4. Mutation (per stop, MUTATION_PROB)│       │
         │  5. EVAL all expanded strings         │       │
         │  6. Tournament select → N_POP         │       │
         │  7. Write CSV row + PNG chart         │       │
         │  8. If gen % CHECKPOINT_EVERY == 0:   │       │
         │     • checkpoints/gen_NNNN.json       │       │
         │     • checkpoints/gen_NNNN_map.html   │       │
         └────────────────┬──────────────────────┘       │
                          │                              │
                    more gens? ───────────────────────►──┘
                          │
                    done
                          │
         ┌────────────────▼──────────────────────┐
         │  Write best_routes.json               │
         │  Write data/routes_map.html           │
         └───────────────────────────────────────┘
```

---

## Reference

Chakroborty, P. & Dwivedi, T. (2002). Optimal route network design for transit systems using genetic algorithms. *Engineering Optimization*, **34**(1), 83–100.
