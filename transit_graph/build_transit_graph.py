"""
Build a proper transit graph from allYerevanTransportLines.json.

Node  = unique bus/trolleybus/minibus stop
Edge  = directed segment between consecutive stops on the same line,
        carrying the real km distance from the source data.

Multiple routes may share the same stop-pair; all are preserved so the
graph is a true multigraph (edges list per source node).

Output: transitGraph2.json  (same top-level shape as transitGraph.json
        but richer: nodes carry 'lines' list, edges carry 'line'/'lineType').
"""

import json
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).parent.parent / "data" / "allYerevanTransportLines.json"
DST = Path(__file__).parent.parent / "data" / "transitGraph.json"


def parse_line(raw_line: list) -> tuple[list[dict], list[float]]:
    """Split the interleaved stop/distance array into two parallel lists."""
    stops: list[dict] = []
    distances: list[float] = []
    for item in raw_line:
        if "id" in item:
            stops.append(item)
        elif "distance" in item:
            distances.append(item["distance"])
    return stops, distances


def process_direction(
    raw_line: list,
    line_num: int,
    line_type: str,
    direction: str,
    nodes: dict[str, dict],
    edges: dict[str, list[dict]],
) -> None:
    """Register stops and directed edges for one direction of a route."""
    stops, distances = parse_line(raw_line)
    for stop in stops:
        sid = stop["id"]
        lng, lat = stop["coordinates"]
        if sid not in nodes:
            nodes[sid] = {
                "id": sid,
                "name": stop["name"],
                "coords": {"lng": lng, "lat": lat},
                "activityLevel": 0,
                "lines": [],
            }
        node = nodes[sid]
        node["activityLevel"] += 1
        if line_num not in node["lines"]:
            node["lines"].append(line_num)

    for i in range(len(stops) - 1):
        src_id = stops[i]["id"]
        tgt_id = stops[i + 1]["id"]
        dist = distances[i] if i < len(distances) else 0.0
        edges[src_id].append({
            "target": tgt_id,
            "weight": round(dist, 6),
            "line": line_num,
            "lineType": line_type,
            "direction": direction,
        })


def build_graph(transport_lines: list[dict]) -> dict:
    nodes: dict[str, dict] = {}
    edges: dict[str, list[dict]] = defaultdict(list)

    for route in transport_lines:
        line_num: int = route["lineNumber"]
        line_type: str = route["lineType"]

        process_direction(route["line"], line_num, line_type, "forward", nodes, edges)

        return_line = route.get("returnLine", [])
        if return_line:
            process_direction(return_line, line_num, line_type, "return", nodes, edges)

    return {"nodes": nodes, "edges": dict(edges)}


def print_stats(graph: dict) -> None:
    nodes = graph["nodes"]
    edges = graph["edges"]
    n_nodes = len(nodes)
    n_edges = sum(len(v) for v in edges.values())
    degrees = {k: len(v) for k, v in edges.items()}
    max_deg = max(degrees.values(), default=0)
    isolated = sum(1 for nid in nodes if nid not in edges)
    unique_pairs = len(set(
        (src, e["target"])
        for src, elist in edges.items()
        for e in elist
    ))

    print(f"Nodes            : {n_nodes}")
    print(f"Total edges      : {n_edges}  (multigraph – includes parallel routes)")
    print(f"Unique stop-pairs: {unique_pairs}")
    print(f"Max out-degree   : {max_deg}")
    print(f"Isolated nodes   : {isolated}")
    print(f"Avg out-degree   : {n_edges / n_nodes:.2f}")

    type_counts: dict[str, int] = {}
    for elist in edges.values():
        for e in elist:
            t = e["lineType"]
            type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Edges by type    : {type_counts}")


def main() -> None:
    with open(SRC) as f:
        transport_lines = json.load(f)

    print(f"Processing {len(transport_lines)} transport lines …")
    graph = build_graph(transport_lines)
    print_stats(graph)

    with open(DST, "w") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {DST}")


if __name__ == "__main__":
    main()
