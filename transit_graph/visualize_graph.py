"""
Visualize Yerevan transit graph (transitGraph2.json) on an interactive map.

Source: rebuilt from allYerevanTransportLines.json, both forward and return
directions included per route.

Layers (toggleable):
  • Bus routes        – blue lines
  • Trolleybus routes – green lines
  • Minibus routes    – orange lines
  • Bus stops         – circles colored/sized by activity level
"""

import json
from collections import defaultdict
from pathlib import Path

import folium
from folium.plugins import MiniMap
from branca.colormap import LinearColormap

DATA_PATH   = Path(__file__).parent.parent / "data" / "transitGraph.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "transit_map.html"

LINE_TYPE_STYLE: dict[str, dict] = {
    "bus":        {"color": "#4fc3f7", "weight": 2.0, "opacity": 0.55},
    "trolleybus": {"color": "#69f0ae", "weight": 2.5, "opacity": 0.60},
    "minibus":    {"color": "#ffb74d", "weight": 2.0, "opacity": 0.55},
}


def load_graph(path: Path) -> tuple[dict, dict]:
    with open(path) as f:
        data = json.load(f)
    return data["nodes"], data["edges"]


def compute_stats(nodes: dict, edges: dict) -> dict:
    n_edges = sum(len(v) for v in edges.values())
    type_counts: dict[str, int] = defaultdict(int)
    for elist in edges.values():
        for e in elist:
            type_counts[e["lineType"]] += 1
    max_activity = max((n["activityLevel"] for n in nodes.values()), default=1)
    isolated = sum(1 for nid in nodes if nid not in edges)
    unique_pairs = len({
        (src, e["target"])
        for src, elist in edges.items()
        for e in elist
    })
    return {
        "n_nodes": len(nodes),
        "n_edges": n_edges,
        "unique_pairs": unique_pairs,
        "type_counts": dict(type_counts),
        "max_activity": max_activity,
        "isolated": isolated,
    }


def build_map(nodes: dict, edges: dict, stats: dict) -> folium.Map:
    center_lat = sum(n["coords"]["lat"] for n in nodes.values()) / len(nodes)
    center_lng = sum(n["coords"]["lng"] for n in nodes.values()) / len(nodes)

    fmap = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )
    fmap.add_child(MiniMap())

    groups: dict[str, folium.FeatureGroup] = {
        lt: folium.FeatureGroup(name=f"{lt.title()} routes", show=True)
        for lt in LINE_TYPE_STYLE
    }

    for src_id, edge_list in edges.items():
        src_node = nodes.get(src_id)
        if src_node is None:
            continue
        src_lat = src_node["coords"]["lat"]
        src_lng = src_node["coords"]["lng"]

        for edge in edge_list:
            tgt_node = nodes.get(str(edge["target"]))
            if tgt_node is None:
                continue
            lt = edge.get("lineType", "bus")
            style = LINE_TYPE_STYLE.get(lt, {"color": "#ccc", "weight": 1.5, "opacity": 0.4})
            direction = edge.get("direction", "forward")
            folium.PolyLine(
                locations=[
                    [src_lat, src_lng],
                    [tgt_node["coords"]["lat"], tgt_node["coords"]["lng"]],
                ],
                color=style["color"],
                weight=style["weight"],
                opacity=style["opacity"],
                tooltip=(
                    f"Line {edge['line']} ({lt}, {direction})<br>"
                    f"{src_node['name']} → {tgt_node['name']}<br>"
                    f"dist: {edge['weight']:.3f} km"
                ),
            ).add_to(groups[lt])

    for g in groups.values():
        g.add_to(fmap)

    max_act = stats["max_activity"]
    colormap = LinearColormap(
        colors=["#b0bec5", "#4fc3f7", "#69f0ae", "#ffdd00", "#ff4d00"],
        vmin=1,
        vmax=max_act,
        caption="Stop activity level (# line-passes through stop)",
    )
    colormap.add_to(fmap)

    stop_group = folium.FeatureGroup(name="Bus stops", show=True)
    for node_id, node in nodes.items():
        act = node["activityLevel"]
        lines_str = ", ".join(str(ln) for ln in sorted(node.get("lines", [])))
        folium.CircleMarker(
            location=[node["coords"]["lat"], node["coords"]["lng"]],
            radius=3 + act * 0.9,
            color=colormap(act),
            fill=True,
            fill_color=colormap(act),
            fill_opacity=0.9,
            tooltip=(
                f"<b>{node['name']}</b><br>"
                f"ID: {node_id}<br>"
                f"Activity: {act}  lines: [{lines_str}]"
            ),
        ).add_to(stop_group)
    stop_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)

    tc = stats["type_counts"]
    stats_html = f"""
    <div style="
        position: fixed; top: 10px; right: 10px; z-index: 1000;
        background: rgba(12,12,18,0.92); color: #e0e0e0;
        padding: 14px 18px; border-radius: 10px;
        font-family: monospace; font-size: 13px;
        border: 1px solid #69f0ae; min-width: 240px;
    ">
        <b style="color:#69f0ae; font-size:15px;">Yerevan Transit Graph</b><br>
        <span style="font-size:11px; color:#aaa;">74 routes · both directions</span>
        <br><br>
        Stops (nodes):     <b style="color:#fff">{stats['n_nodes']}</b><br>
        Total edges:       <b style="color:#fff">{stats['n_edges']}</b><br>
        Unique stop-pairs: <b style="color:#fff">{stats['unique_pairs']}</b><br>
        Isolated stops:    <b style="color:#ff7043">{stats['isolated']}</b><br>
        <br>
        <span style="color:#4fc3f7">■</span> Bus:        <b>{tc.get('bus', 0)}</b> edges<br>
        <span style="color:#69f0ae">■</span> Trolleybus: <b>{tc.get('trolleybus', 0)}</b> edges<br>
        <span style="color:#ffb74d">■</span> Minibus:    <b>{tc.get('minibus', 0)}</b> edges<br>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(stats_html))
    return fmap


def main() -> None:
    nodes, edges = load_graph(DATA_PATH)
    stats = compute_stats(nodes, edges)

    for k, v in stats.items():
        print(f"{k}: {v}")

    fmap = build_map(nodes, edges, stats)
    fmap.save(str(OUTPUT_PATH))
    print(f"\nMap saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
