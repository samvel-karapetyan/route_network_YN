#!/usr/bin/env python3
"""
Render optimised routes on an interactive Leaflet map.

Reads a best_routes.json (or any checkpoints/gen_NNNN.json) and
data/transitGraph.json, then writes a self-contained HTML file.
No third-party Python packages required — Leaflet is loaded from CDN.

Usage:
    python3 visualize_routes.py                          # best_routes.json → data/routes_map.html
    python3 visualize_routes.py checkpoints/gen_0050.json
    python3 visualize_routes.py checkpoints/gen_0050.json -o my_map.html
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DATA_DIR   = Path("data")
GRAPH_PATH = DATA_DIR / "transitGraph.json"
DEFAULT_IN = Path("best_routes.json")
DEFAULT_OUT = DATA_DIR / "routes_map.html"


# ── Colour generation ─────────────────────────────────────────────────────────

def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """h in [0,1), s and v in [0,1]."""
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    sector = i % 6
    r, g, b = [
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ][sector]
    return int(r * 255), int(g * 255), int(b * 255)


def make_palette(n: int) -> list[str]:
    """
    Generate n visually distinct hex colours using the golden-ratio hue method.
    Alternates between two luminosity bands so adjacent routes are easier to tell apart.
    """
    golden = 0.618033988749895
    colours: list[str] = []
    h = 0.1
    for i in range(n):
        s = 0.75 if i % 2 == 0 else 0.55
        v = 0.95 if i % 2 == 0 else 0.80
        r, g, b = _hsv_to_rgb(h % 1.0, s, v)
        colours.append(f"#{r:02x}{g:02x}{b:02x}")
        h += golden
    return colours


# ── Data loading ──────────────────────────────────────────────────────────────

def load_routes(path: Path) -> tuple[list[list[str]], dict, float]:
    with open(path) as f:
        data = json.load(f)
    routes: list[list[str]] = data["routes"]
    metrics: dict = data.get("metrics", {})
    totfit: float = data.get("best_TOTFIT", 0.0)
    return routes, metrics, totfit


def load_graph(path: Path) -> tuple[dict, dict]:
    with open(path) as f:
        g = json.load(f)
    return g["nodes"], g["edges"]


# ── Route geometry ────────────────────────────────────────────────────────────

def route_length_km(route: list[str], edges: dict) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        for e in edges.get(route[i], []):
            if e["target"] == route[i + 1]:
                total += e["weight"]
                break
    return total


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d12; font-family: monospace; }}
  #map {{ width: 100vw; height: 100vh; }}

  #stats {{
    position: fixed; top: 12px; right: 12px; z-index: 1000;
    background: rgba(10,10,18,0.93);
    color: #dde; padding: 14px 18px; border-radius: 10px;
    border: 1px solid #4fc3f7; min-width: 260px;
    font-size: 13px; line-height: 1.7;
    box-shadow: 0 4px 24px rgba(0,0,0,0.6);
  }}
  #stats b.head {{ color: #4fc3f7; font-size: 15px; }}
  #stats .val {{ color: #fff; font-weight: bold; }}
  #stats .good {{ color: #69f0ae; }}
  #stats .warn {{ color: #ffb74d; }}
  #stats .bad  {{ color: #ff5252; }}

  #legend {{
    position: fixed; bottom: 30px; left: 12px; z-index: 1000;
    background: rgba(10,10,18,0.90);
    color: #ccc; padding: 10px 14px; border-radius: 8px;
    border: 1px solid #333; font-size: 12px;
    max-height: 55vh; overflow-y: auto; min-width: 180px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }}
  #legend .l-title {{ color: #4fc3f7; font-size: 13px; margin-bottom: 6px; }}
  .l-row {{ display: flex; align-items: center; gap: 8px;
            cursor: pointer; padding: 2px 4px; border-radius: 4px; }}
  .l-row:hover {{ background: rgba(255,255,255,0.07); }}
  .l-swatch {{ width: 26px; height: 4px; border-radius: 2px; flex-shrink: 0; }}
  .l-label {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 130px; }}
  .l-row.hidden {{ opacity: 0.35; }}

  .leaflet-control-zoom {{ border: 1px solid #333 !important; }}

  #toolbar {{
    position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
    z-index: 1000; display: flex; gap: 8px;
  }}
  .tb-btn {{
    background: rgba(10,10,18,0.92); color: #dde;
    border: 1px solid #444; border-radius: 6px;
    padding: 6px 14px; font-family: monospace; font-size: 12px;
    cursor: pointer; transition: background 0.15s, border-color 0.15s;
    white-space: nowrap;
  }}
  .tb-btn:hover  {{ background: rgba(79,195,247,0.18); border-color: #4fc3f7; }}
  .tb-btn.active {{ background: rgba(79,195,247,0.25); border-color: #4fc3f7; color: #4fc3f7; }}
</style>
</head>
<body>
<div id="map"></div>

<div id="toolbar">
  <button class="tb-btn" id="btn-show-all">Show all routes</button>
  <button class="tb-btn" id="btn-hide-all">Hide all routes</button>
  <button class="tb-btn active" id="btn-toggle-stops">Hide stops</button>
</div>
<div id="stats">
  <b class="head">Optimised Route Network</b><br>
  <span style="font-size:11px;color:#888">{source_label}</span>
  <br>
  Routes:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span class="val">{n_routes}</span><br>
  Stops covered: <span class="val">{n_covered}</span> / {n_total}<br>
  TOTFIT:&nbsp;&nbsp;&nbsp;&nbsp; <span class="val">{totfit:.4f}</span><br>
  <br>
  Direct d0%:&nbsp;&nbsp;&nbsp; <span class="good">{d0p:.1f}%</span><br>
  1-transfer d1%: <span class="good">{d1p:.1f}%</span><br>
  2-transfer d2%: <span class="warn">{d2p:.1f}%</span><br>
  Unsatisfied%:&nbsp; <span class="bad">{dunp:.1f}%</span><br>
  Avg travel time: <span class="val">{att:.2f} km</span>
</div>
<div id="legend">
  <div class="l-title">Routes (click to toggle)</div>
  <div id="legend-rows"></div>
</div>

<script>
const ROUTES  = {routes_json};
const NODES   = {nodes_json};
const PALETTE = {palette_json};

const map = L.map('map', {{ zoomControl: true }}).setView(
  [{center_lat}, {center_lng}], 13
);

L.tileLayer(
  'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: 'abcd', maxZoom: 19,
  }}
).addTo(map);

/* ── Stop markers ─────────────────────────────────────────────────────────── */
const stopGroup = L.featureGroup().addTo(map);
const stopUsage = {{}};   // stopId -> count of routes using it

ROUTES.forEach(route => {{
  route.forEach(sid => {{ stopUsage[sid] = (stopUsage[sid] || 0) + 1; }});
}});

Object.entries(NODES).forEach(([sid, node]) => {{
  const usage = stopUsage[sid] || 0;
  if (usage === 0) return;           // only show stops that are on some route
  const r   = Math.min(2 + usage * 1.2, 9);
  const col = usage >= 4 ? '#ffdd00' : usage >= 2 ? '#4fc3f7' : '#b0bec5';
  L.circleMarker(
    [node.coords.lat, node.coords.lng],
    {{ radius: r, color: col, weight: 1, fillColor: col, fillOpacity: 0.85 }}
  )
  .bindTooltip(
    `<b>${{node.name}}</b><br>ID: ${{sid}}<br>Used by ${{usage}} route(s)`,
    {{ sticky: true }}
  )
  .addTo(stopGroup);
}});

/* ── Route polylines ──────────────────────────────────────────────────────── */
const routeLayers = [];
const legendRows  = document.getElementById('legend-rows');

ROUTES.forEach((route, idx) => {{
  const color = PALETTE[idx % PALETTE.length];
  const coords = route
    .filter(sid => NODES[sid])
    .map(sid => [NODES[sid].coords.lat, NODES[sid].coords.lng]);

  if (coords.length < 2) return;

  const layer = L.polyline(coords, {{
    color, weight: 3, opacity: 0.80,
  }})
  .bindTooltip(
    `<b>Route ${{idx + 1}}</b><br>${{route.length}} stops<br>` +
    `${{route[0] && NODES[route[0]] ? NODES[route[0]].name : '?'}}` +
    ` → ` +
    `${{route[route.length-1] && NODES[route[route.length-1]] ? NODES[route[route.length-1]].name : '?'}}`,
    {{ sticky: true }}
  )
  .addTo(map);

  routeLayers.push({{ layer, color, visible: true }});

  /* Legend row */
  const row = document.createElement('div');
  row.className = 'l-row';
  row.innerHTML =
    `<span class="l-swatch" style="background:${{color}}"></span>` +
    `<span class="l-label">Route ${{idx + 1}} (${{route.length}} stops)</span>`;
  row.addEventListener('click', () => {{
    const entry = routeLayers[idx];
    if (entry.visible) {{ map.removeLayer(entry.layer); row.classList.add('hidden'); }}
    else               {{ entry.layer.addTo(map);       row.classList.remove('hidden'); }}
    entry.visible = !entry.visible;
  }});
  legendRows.appendChild(row);
}});

/* ── Toolbar buttons ──────────────────────────────────────────────────────── */
const allRows = document.querySelectorAll('.l-row');

document.getElementById('btn-show-all').addEventListener('click', () => {{
  routeLayers.forEach((entry, idx) => {{
    if (!entry.visible) {{
      entry.layer.addTo(map);
      entry.visible = true;
      allRows[idx] && allRows[idx].classList.remove('hidden');
    }}
  }});
}});

document.getElementById('btn-hide-all').addEventListener('click', () => {{
  routeLayers.forEach((entry, idx) => {{
    if (entry.visible) {{
      map.removeLayer(entry.layer);
      entry.visible = false;
      allRows[idx] && allRows[idx].classList.add('hidden');
    }}
  }});
}});

let stopsVisible = true;
const btnStops = document.getElementById('btn-toggle-stops');
btnStops.addEventListener('click', () => {{
  if (stopsVisible) {{
    map.removeLayer(stopGroup);
    btnStops.textContent = 'Show stops';
    btnStops.classList.remove('active');
  }} else {{
    stopGroup.addTo(map);
    btnStops.textContent = 'Hide stops';
    btnStops.classList.add('active');
  }}
  stopsVisible = !stopsVisible;
}});
</script>
</body>
</html>
"""


def build_html(
    routes: list[list[str]],
    nodes: dict,
    metrics: dict,
    totfit: float,
    source_label: str,
) -> str:
    palette = make_palette(len(routes))

    # Stops that appear in at least one route
    covered = {sid for route in routes for sid in route if sid in nodes}

    # Map centre
    lats = [n["coords"]["lat"] for n in nodes.values()]
    lngs = [n["coords"]["lng"] for n in nodes.values()]
    center_lat = sum(lats) / len(lats)
    center_lng = sum(lngs) / len(lngs)

    # Slim node dict: only coords + name (keep HTML small)
    slim_nodes = {
        nid: {"coords": n["coords"], "name": n["name"]}
        for nid, n in nodes.items()
    }

    return _HTML_TEMPLATE.format(
        title        = f"Optimised Routes — {source_label}",
        source_label = source_label,
        n_routes     = len(routes),
        n_covered    = len(covered),
        n_total      = len(nodes),
        totfit       = totfit,
        d0p          = metrics.get("d0p",  0.0),
        d1p          = metrics.get("d1p",  0.0),
        d2p          = metrics.get("d2p",  0.0),
        dunp         = metrics.get("dunp", 0.0),
        att          = metrics.get("ATT",  0.0),
        center_lat   = round(center_lat, 6),
        center_lng   = round(center_lng, 6),
        routes_json  = json.dumps(routes),
        nodes_json   = json.dumps(slim_nodes),
        palette_json = json.dumps(palette),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    # Parse -o / --output flag
    out_path = DEFAULT_OUT
    if "-o" in args:
        i = args.index("-o")
        out_path = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    elif "--output" in args:
        i = args.index("--output")
        out_path = Path(args[i + 1])
        args = args[:i] + args[i + 2:]

    in_path = Path(args[0]) if args else DEFAULT_IN

    if not in_path.exists():
        print(f"Route file not found: {in_path}")
        print("Run route_optimizer.py first to generate best_routes.json,")
        print("or pass a checkpoint: python3 visualize_routes.py checkpoints/gen_0010.json")
        sys.exit(1)

    if not GRAPH_PATH.exists():
        print(f"Graph file not found: {GRAPH_PATH}")
        sys.exit(1)

    print(f"Loading routes from {in_path} …")
    routes, metrics, totfit = load_routes(in_path)

    print(f"Loading graph from {GRAPH_PATH} …")
    nodes, _ = load_graph(GRAPH_PATH)

    source_label = in_path.name
    if "generation" in json.load(open(in_path)):
        gen = json.load(open(in_path))["generation"]
        source_label = f"checkpoint gen {gen}"

    print(f"Building map ({len(routes)} routes, {len(nodes)} stops) …")
    html = build_html(routes, nodes, metrics, totfit, source_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    covered = len({sid for r in routes for sid in r if sid in nodes})
    print(f"Done.")
    print(f"  Routes   : {len(routes)}")
    print(f"  Covered  : {covered} / {len(nodes)} stops")
    print(f"  TOTFIT   : {totfit:.4f}")
    print(f"  d0={metrics.get('d0p',0):.1f}%  d1={metrics.get('d1p',0):.1f}%  "
          f"d2={metrics.get('d2p',0):.1f}%  un={metrics.get('dunp',0):.1f}%")
    print(f"  Map saved → {out_path}")
    print(f"  Open with: xdg-open {out_path}")


if __name__ == "__main__":
    main()
