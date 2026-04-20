# Visualization

Three visualization tools are produced by or work alongside `route_optimizer.py`. All require only stdlib + Pillow (chart) or a browser (map).

---

## Output files at a glance

| File | Written | What it shows |
|---|---|---|
| `optimization_progress.png` | After every generation | 4-panel PNG progress chart |
| `optimization_log.csv` | After every generation | Raw numbers — open in any spreadsheet |
| `data/routes_map.html` | End of run | Interactive map of the final best routes |
| `checkpoints/gen_NNNN.json` | Every `CHECKPOINT_EVERY` gens | Best route set snapshot |
| `checkpoints/gen_NNNN_map.html` | Every `CHECKPOINT_EVERY` gens | Interactive map of that snapshot |

---

## 1 — Progress chart (`optimization_progress.png`)

Rendered automatically by `route_optimizer.py` after every generation using Pillow. Open it in any image viewer while the optimizer runs — press F5 or use a viewer that auto-reloads.

```bash
eog optimization_progress.png          # GNOME — F5 to reload
gwenview optimization_progress.png     # KDE — auto-reloads
xdg-open optimization_progress.png    # default viewer
```

### Four panels

| Panel | What to watch |
|---|---|
| **Fitness (TOTFIT)** | Primary convergence signal — should trend upward |
| **Demand coverage %** | `d0%` (direct) rising and `Unsatisfied%` falling is progress |
| **F1 / F2 / F3 components** | Shows which sub-objective is driving or bottlenecking improvement |
| **Average Travel Time** | Lower is better — routes becoming more efficient |

### Render manually at any time

```bash
python3 plot_results.py                          # reads optimization_log.csv
python3 plot_results.py path/to/other_log.csv   # specific log file
python3 plot_results.py --watch                  # re-render every 5 s (live mode)
```

Or from Python:

```python
from plot_results import render
from pathlib import Path

render(Path("optimization_log.csv"), Path("optimization_progress.png"))
```

---

## 2 — Interactive route map (`data/routes_map.html`)

`visualize_routes.py` reads a route JSON and `data/transitGraph.json`, then writes a self-contained HTML file. Leaflet.js loads from CDN — no Python packages required.

```bash
# From the final best route set
python3 visualize_routes.py

# From a checkpoint
python3 visualize_routes.py checkpoints/gen_0050.json

# Custom output path
python3 visualize_routes.py checkpoints/gen_0050.json -o my_map.html

# Open in browser
xdg-open data/routes_map.html
```

The map is also generated automatically inside `route_optimizer.py`:
- at every checkpoint → `checkpoints/gen_NNNN_map.html`
- at the end of the run → `data/routes_map.html`

### Map elements

| Element | Detail |
|---|---|
| **Route polylines** | One colour per route — 70 routes use golden-ratio HSV spacing so adjacent routes are visually distinct |
| **Stop circles** | Sized and coloured by how many routes serve that stop: yellow = hub (4+), blue = shared (2–3), grey = single |
| **Hover tooltips** | Routes: index, stop count, first → last stop name. Stops: OSM name, ID, route count |
| **Legend (bottom-left)** | Click any row to show/hide that route on the map |
| **Stats panel (top-right)** | Routes, covered stops, TOTFIT, d0%/d1%/d2%/un%, ATT |

---

## 3 — Checkpoints

Every `CHECKPOINT_EVERY` generations (default: 10), two files are written:

```
checkpoints/
  gen_0010.json          ← route set + metrics JSON
  gen_0010_map.html      ← interactive map for that generation
  gen_0020.json
  gen_0020_map.html
  ...
```

### JSON schema

```json
{
  "generation": 10,
  "best_TOTFIT": 11.23,
  "metrics": {
    "d0p": 8.1, "d1p": 45.2, "d2p": 14.3, "dunp": 32.4,
    "ATT": 16.8, "F1": 4.2, "F2": 3.9, "F3": 3.1
  },
  "routes": [
    ["stopId_1", "stopId_2", "stopId_3", ...],
    ...
  ]
}
```

### Load a checkpoint in Python

```python
import json

with open("checkpoints/gen_0050.json") as f:
    ckpt = json.load(f)

print(ckpt["metrics"])
routes = ckpt["routes"]   # list of 70 stop-ID sequences
```

### Compare two checkpoints

```python
import json

early = json.load(open("checkpoints/gen_0010.json"))
late  = json.load(open("checkpoints/gen_0100.json"))

print(f"gen 10  → TOTFIT={early['best_TOTFIT']:.3f}  d0={early['metrics']['d0p']:.1f}%")
print(f"gen 100 → TOTFIT={late['best_TOTFIT']:.3f}   d0={late['metrics']['d0p']:.1f}%")
```

### Change checkpoint frequency

Edit in `route_optimizer.py`:

```python
CHECKPOINT_EVERY = 10   # default
CHECKPOINT_EVERY = 5    # more frequent
CHECKPOINT_EVERY = 1    # every generation
CHECKPOINT_EVERY = 0    # disabled
```

---

## 4 — Reading the CSV directly

```python
import csv

rows = list(csv.DictReader(open("optimization_log.csv")))

totfit = [float(r["best_TOTFIT"]) for r in rows]
d0p    = [float(r["d0p"])         for r in rows]
dunp   = [float(r["dunp"])        for r in rows]

print(f"Generations  : {len(rows)}")
print(f"TOTFIT       : {totfit[0]:.4f} → {totfit[-1]:.4f}")
print(f"Direct d0%   : {d0p[0]:.1f}% → {d0p[-1]:.1f}%")
print(f"Unsatisfied% : {dunp[0]:.1f}% → {dunp[-1]:.1f}%")
```
