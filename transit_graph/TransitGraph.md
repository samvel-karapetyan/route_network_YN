# Yerevan Transit Graph

## Source Data

| File | Role |
|---|---|
| `data/allYerevanTransportLines.json` | Raw route definitions — 74 routes (bus, trolleybus, minibus) |
| `data/transitGraph.json` | Derived directed multigraph used in all analysis |
| `data/transit_map.html` | Interactive Folium map of the graph |

Paths above are relative to the capstone project root.

### Raw Format (`data/allYerevanTransportLines.json`)

Each entry is a route object:

```json
{
  "lineNumber": 14,
  "lineType": "bus",
  "line": [
    { "id": "...", "coordinates": [lng, lat], "name": "Stop Name" },
    { "distance": 0.567 },
    { "id": "...", "coordinates": [lng, lat], "name": "Next Stop" },
    ...
  ],
  "returnLine": [ /* same interleaved format, opposite direction */ ]
}
```

Key structural detail: every route has **both `line` (forward)** and **`returnLine` (return)** arrays. They are independent stop sequences — the return direction is not simply a reversal of the forward stops. Both must be parsed to get the full network.

---

## Graph Structure (`data/transitGraph.json`)

### Nodes

Each node is a unique bus/trolleybus/minibus stop:

```json
"1733309221": {
  "id": "1733309221",
  "name": "Arshakunyats Avenue, 59",
  "coords": { "lng": 44.497831354, "lat": 40.14751156 },
  "activityLevel": 2,
  "lines": [14, 33]
}
```

- `activityLevel` — total number of route-passes through this stop (sum across both directions of all lines)
- `lines` — distinct line numbers that serve this stop

### Edges

Directed adjacency list — each entry is a list of outgoing segments:

```json
"1733309221": [
  { "target": "1733306811", "weight": 0.567004, "line": 14, "lineType": "bus", "direction": "return" },
  { "target": "1733306811", "weight": 0.567004, "line": 33, "lineType": "bus", "direction": "return" }
]
```

- `weight` — real distance in km between the two stops (from source data)
- `direction` — `"forward"` or `"return"`, which direction of the route this edge belongs to
- Multiple edges between the same stop-pair are kept (multigraph), one per route that serves that segment

---

## Statistics

| Metric | Value |
|---|---|
| Total routes | 74 (63 bus, 5 trolleybus, 6 minibus) |
| Nodes (unique stops) | 1,387 |
| Total edges (multigraph) | 6,375 |
| Unique stop-pairs | 1,745 |
| Forward edges | 3,305 |
| Return edges | 3,070 |
| Isolated stops (no outgoing edge) | 21 |
| Average out-degree | 4.60 |
| Max out-degree | 26 |
| Total network distance | ~2,837 km (sum of all edge weights) |

### Edges by Transport Type

| Type | Edges |
|---|---|
| Bus | 5,700 |
| Trolleybus | 403 |
| Minibus | 272 |

### Out-Degree Distribution

Most stops have low out-degree (1–2), meaning they sit on a single route corridor. High-degree stops are transfer hubs where many lines converge.

| Out-degree | # Stops |
|---|---|
| 1 | 431 |
| 2 | 246 |
| 3 | 109 |
| 4–6 | 248 |
| 7–10 | 143 |
| 11–15 | 120 |
| 16–26 | 67 |

---

## Key Hubs

Stops with the highest out-degree — these are the most critical transfer points in the network:

| Stop | Out-degree | Lines served |
|---|---|---|
| Khachatur Abovyan Square | 26 | 1,5,10,14,15,16,20,22,23,26,29,30,35,36,40,41,42,44,46,50,53,54,55,62,77,99 |
| Agricultural University | 26 | same 26 lines |
| Medical University | 24 | 24 lines |
| Garegin Nzhdeh Square | 24 | 21 lines |
| Mashtots / Tumanyan | 22 | 21 lines |
| Gay / Marshal Khudyakov | 22 | 22 lines |

Abovyan Square and Agricultural University are the highest-degree nodes in the network — removing either would disrupt 26 routes simultaneously.

---

## Known Data Issues

### 1. Isolated Stops (21 nodes)

These stops appear in route definitions but have no outgoing edge — they are terminus stops at the end of a `returnLine` sequence. Examples:

- Харберд (line 6)
- Mika Stadium (lines 6, 11, 42, 66)
- Nubarashen (lines 10, 48, 97)

These are correct: a terminus is the last stop on a direction, so it has no "next stop" to point to.

### 2. Duplicate Stop Names, Different IDs

Multiple physical stops share the same name but have distinct OSM node IDs. Example:

| ID | Name | Lines |
|---|---|---|
| `1733306811` | Hayreniq | 44 |
| `1787749186` | Hayreniq | 48, 2 |

These are separate stops in the same neighborhood. No transfer edge is generated between them automatically because they never appear adjacent in any route. If pedestrian transfer edges are needed, a proximity threshold (e.g. < 300 m) can be used to add them.

Use `data/transitGraph.json` for all downstream work.

---

## Scripts

| Script | Purpose |
|---|---|
| `transit_graph/build_transit_graph.py` | Parse `data/allYerevanTransportLines.json` → `data/transitGraph.json` |
| `transit_graph/visualize_graph.py` | Render `data/transitGraph.json` → `data/transit_map.html` |
