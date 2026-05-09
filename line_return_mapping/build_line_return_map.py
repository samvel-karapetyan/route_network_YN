#!/usr/bin/env python3
"""
For each route in allYerevanTransportLines.json, map every outbound (`line`)
stop id to the nearest return (`returnLine`) stop id (Haversine).

Outputs:
  output/mapping.json      — { lineNo: { outboundId: returnId } } only
  output/line_return_map.html — Leaflet + OpenStreetMap visualization

  python3 line_return_mapping/build_line_return_map.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_TRANSPORT = REPO_ROOT / "data" / "allYerevanTransportLines.json"
OUTPUT_DIR = HERE / "output"


def parse_stops(raw_line: list) -> list[dict[str, Any]]:
    stops: list[dict[str, Any]] = []
    for item in raw_line:
        if isinstance(item, dict) and "id" in item:
            stops.append(item)
    return stops


def haversine_km(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lng / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def nearest_return_id(fwd: dict[str, Any], ret_stops: list[dict[str, Any]]) -> str:
    lng1, lat1 = fwd["coordinates"]
    best_id = ret_stops[0]["id"]
    best_d = math.inf
    for s in ret_stops:
        lng2, lat2 = s["coordinates"]
        d = haversine_km(lng1, lat1, lng2, lat2)
        if d < best_d:
            best_d = d
            best_id = s["id"]
    return str(best_id)


def to_latlng_seq(stops: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(s["coordinates"][1]), float(s["coordinates"][0])] for s in stops]


def analyze_routes(
    transport_path: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    """
    Returns:
      mapping — id→id only per line
      viz_lines — geometry + pair coords for HTML (same pairing)
    """
    with open(transport_path, encoding="utf-8") as f:
        routes: list[dict[str, Any]] = json.load(f)

    mapping: dict[str, dict[str, str]] = {}
    viz_lines: list[dict[str, Any]] = []

    for entry in routes:
        line_num = entry.get("lineNumber")
        if line_num is None:
            continue
        key = str(line_num)
        raw_fwd = entry.get("line") or []
        raw_ret = entry.get("returnLine") or []
        fwd_stops = parse_stops(raw_fwd)
        ret_stops = parse_stops(raw_ret)
        if not fwd_stops or not ret_stops:
            continue

        ret_by_id = {str(s["id"]): s for s in ret_stops}
        line_map: dict[str, str] = {}
        pairs: list[dict[str, Any]] = []

        for stop_f in fwd_stops:
            oid = str(stop_f["id"])
            rid = nearest_return_id(stop_f, ret_stops)
            line_map[oid] = rid
            lng_o, lat_o = stop_f["coordinates"]
            sr = ret_by_id[rid]
            lng_r, lat_r = sr["coordinates"]
            pairs.append(
                {
                    "outbound_id": oid,
                    "return_id": rid,
                    "outbound_latlng": [lat_o, lng_o],
                    "return_latlng": [lat_r, lng_r],
                }
            )

        mapping[key] = line_map
        viz_lines.append(
            {
                "lineNumber": line_num,
                "lineType": entry.get("lineType", ""),
                "outbound_latlng": to_latlng_seq(fwd_stops),
                "return_latlng": to_latlng_seq(ret_stops),
                "pairs": pairs,
            }
        )

    return mapping, viz_lines


def _palette(n: int) -> list[str]:
    golden = 0.618033988749895
    colours: list[str] = []
    h = 0.07
    for i in range(n):
        s = 0.8 if i % 2 else 0.55
        v = 0.9 if i % 2 else 0.75
        hh = (h + i * golden) % 1.0
        c = hh * 6
        x = 1 - abs((c % 2) - 1)
        r, g, b = 0.0, 0.0, 0.0
        seg = int(c)
        if seg == 0:
            r, g, b = v, x * v, 0
        elif seg == 1:
            r, g, b = x * v, v, 0
        elif seg == 2:
            r, g, b = 0, v, x * v
        elif seg == 3:
            r, g, b = 0, x * v, v
        elif seg == 4:
            r, g, b = x * v, 0, v
        else:
            r, g, b = v, 0, x * v
        colours.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colours


def render_html(viz_lines: list[dict[str, Any]], out_html: Path) -> None:
    colours = _palette(max(len(viz_lines), 1))
    num_to_color = {
        str(v["lineNumber"]): colours[i % len(colours)]
        for i, v in enumerate(viz_lines)
    }
    payload = {"lines": viz_lines, "colors": num_to_color}
    data_json = json.dumps(payload, ensure_ascii=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Outbound → return stop ID pairs (nearest)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossorigin=""/>
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .legend {{
      position: absolute; bottom: 24px; right: 12px; z-index: 1000;
      background: rgba(255,255,255,0.94); padding: 10px 12px; font: 13px/1.4 system-ui, sans-serif;
      border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.25); max-width: 300px;
    }}
    .legend h4 {{ margin: 0 0 6px 0; font-size: 14px; }}
    .legend span {{ display: inline-block; width: 14px; height: 14px; margin-right: 6px;
      vertical-align: middle; border-radius: 3px; }}
    .legend .toggle-all {{
      display: flex; align-items: center; gap: 8px; margin: 10px 0 6px 0;
      cursor: pointer; font-weight: 600; user-select: none;
    }}
    .legend .toggle-all input {{ cursor: pointer; width: 16px; height: 16px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="legend">
    <h4>Stop ID mapping (outbound → return)</h4>
    <label class="toggle-all"><input type="checkbox" id="show-all-routes" checked /> Show all route lines</label>
    <div><span style="background:#2563eb"></span> Solid = outbound <code>line</code></div>
    <div><span style="background:#ea580c"></span> Dashed = <code>returnLine</code></div>
    <div><span style="background:#94a3b8"></span> Link = nearest return stop</div>
    <p style="margin:8px 0 0 0;color:#444">Click markers for IDs. Toggle layers top-right.</p>
  </div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
          crossorigin=""></script>
  <script>
const DATA = {data_json};
const map = L.map('map').setView([40.18, 44.51], 12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}}).addTo(map);

const overlays = {{}};
const allBounds = [];
const allGroups = [];

function addLine(entry) {{
  const color = DATA.colors[String(entry.lineNumber)] || '#3388ff';
  const g = L.featureGroup();

  L.polyline(entry.outbound_latlng, {{ color: color, weight: 4, opacity: 0.88 }}).addTo(g);
  L.polyline(entry.return_latlng, {{
    color: '#c2410c', weight: 3, opacity: 0.75, dashArray: '8 6'
  }}).addTo(g);

  entry.pairs.forEach(p => {{
    L.polyline([p.outbound_latlng, p.return_latlng], {{
      color: '#64748b', weight: 1, opacity: 0.55, dashArray: '4 4'
    }}).addTo(g);
    const popup =
      '<b>Outbound</b> <code>' + p.outbound_id + '</code><br/>' +
      '→ <b>return</b> <code>' + p.return_id + '</code>';
    L.circleMarker(p.outbound_latlng, {{
      radius: 4, color: color, fillColor: color, fillOpacity: 0.75, weight: 1
    }}).bindPopup(popup).addTo(g);
  }});

  g.addTo(map);
  allGroups.push(g);
  try {{ allBounds.push(g.getBounds()); }} catch (e) {{}}
  overlays[entry.lineType + ' ' + entry.lineNumber + ' (' + entry.pairs.length + ')'] = g;
}}

DATA.lines.forEach(addLine);
L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

document.getElementById('show-all-routes').addEventListener('change', function () {{
  const on = this.checked;
  allGroups.forEach(g => {{ if (on) map.addLayer(g); else map.removeLayer(g); }});
}});
if (allBounds.length) {{
  let b = allBounds[0];
  for (let i = 1; i < allBounds.length; i++) b = b.extend(allBounds[i]);
  map.fitBounds(b, {{ padding: [28, 28] }});
}}
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mapping, viz_lines = analyze_routes(DEFAULT_TRANSPORT)

    json_path = OUTPUT_DIR / "mapping.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=True)

    html_path = OUTPUT_DIR / "line_return_map.html"
    render_html(viz_lines, html_path)

    n_pairs = sum(len(v) for v in mapping.values())
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {html_path.relative_to(REPO_ROOT)}")
    print(f"Lines: {len(mapping)} | Outbound→return id pairs: {n_pairs}")
    print("Open line_return_map.html in a browser (needs network for map tiles).")


if __name__ == "__main__":
    main()
