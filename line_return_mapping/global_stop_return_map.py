#!/usr/bin/env python3
"""
Global stopId ↔ partner stopId mapping from allYerevanTransportLines.json.

Base rule (same as before): each stop gets a preferred “return-side” partner from
line geometry; multi-line outbound picks the closest return candidate.

Then:
  • Never store stopId → itself.
  • Whenever we record A → B (A ≠ B), we also merge B → A using the same distance.
  • If a stop already has a partner, replace it only when the new candidate is
    strictly closer (Haversine).

Output: output/global_stop_to_return.json — flat { "stopId": "partnerStopId", ... }
        (only stops that ended up with a non-self partner)

  python3 line_return_mapping/global_stop_return_map.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_line_return_map import (
    DEFAULT_TRANSPORT,
    OUTPUT_DIR,
    haversine_km,
    nearest_return_id,
    parse_stops,
)


def _dist(
    coords: dict[str, tuple[float, float]], u: str, v: str
) -> float:
    lng1, lat1 = coords[u]
    lng2, lat2 = coords[v]
    return haversine_km(lng1, lat1, lng2, lat2)


def merge_symmetric_partner(
    m: dict[str, str],
    coords: dict[str, tuple[float, float]],
    a: str,
    b: str,
) -> None:
    """If a ≠ b, update m[a] and m[b] using closer-wins rule (symmetric)."""
    if a == b:
        return
    d_ab = _dist(coords, a, b)
    if a not in m or _dist(coords, a, m[a]) > d_ab:
        m[a] = b
    if b not in m or _dist(coords, b, m[b]) > d_ab:
        m[b] = a


def build_global_mapping(transport_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    with open(transport_path, encoding="utf-8") as f:
        routes: list[dict[str, Any]] = json.load(f)

    all_coords: dict[str, tuple[float, float]] = {}
    return_stop_ids: set[str] = set()
    outbound_candidates: dict[str, list[tuple[str, float]]] = {}

    for entry in routes:
        raw_fwd = entry.get("line") or []
        raw_ret = entry.get("returnLine") or []
        fwd_stops = parse_stops(raw_fwd)
        ret_stops = parse_stops(raw_ret)

        for s in fwd_stops:
            sid = str(s["id"])
            all_coords[sid] = (float(s["coordinates"][0]), float(s["coordinates"][1]))
        for s in ret_stops:
            sid = str(s["id"])
            all_coords[sid] = (float(s["coordinates"][0]), float(s["coordinates"][1]))
            return_stop_ids.add(sid)

        if not fwd_stops or not ret_stops:
            continue

        ret_by_id = {str(s["id"]): s for s in ret_stops}
        for stop_f in fwd_stops:
            oid = str(stop_f["id"])
            rid = nearest_return_id(stop_f, ret_stops)
            sr = ret_by_id[rid]
            d = haversine_km(
                stop_f["coordinates"][0],
                stop_f["coordinates"][1],
                sr["coordinates"][0],
                sr["coordinates"][1],
            )
            outbound_candidates.setdefault(oid, []).append((rid, d))

    if not return_stop_ids:
        raise RuntimeError("No return-line stops found in transport data.")

    # Preferred partner per stop (may be self for nearest-on-return-line fallback)
    preferred: dict[str, str] = {}
    for sid, (lng, lat) in all_coords.items():
        if sid in outbound_candidates:
            pairs = outbound_candidates[sid]
            best_rid, _ = min(pairs, key=lambda x: x[1])
            preferred[sid] = best_rid
        else:
            best_rid: str | None = None
            best_d = float("inf")
            for rid in return_stop_ids:
                lng2, lat2 = all_coords[rid]
                d = haversine_km(lng, lat, lng2, lat2)
                if d < best_d:
                    best_d = d
                    best_rid = rid
            assert best_rid is not None
            preferred[sid] = best_rid

    merged: dict[str, str] = {}
    for sid, partner in preferred.items():
        merge_symmetric_partner(merged, all_coords, sid, partner)

    n_multi = sum(1 for pairs in outbound_candidates.values() if len(pairs) > 1)
    stats = {
        "n_stops_in_map": len(merged),
        "n_preferred_pairs_non_self": sum(
            1 for s, p in preferred.items() if s != p
        ),
        "n_with_multiple_outbound_lines": n_multi,
        "n_return_stop_ids": len(return_stop_ids),
        "n_all_stops": len(all_coords),
    }
    return merged, stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "global_stop_to_return.json"
    mapping, stats = build_global_mapping(DEFAULT_TRANSPORT)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=True)

    repo_root = OUTPUT_DIR.parent.parent
    print(f"Wrote {out_path.relative_to(repo_root)}")
    print(
        f"Entries in map: {stats['n_stops_in_map']} (stops with a non-self partner) | "
        f"all stops in data: {stats['n_all_stops']} | "
        f"non-self preferred before merge: {stats['n_preferred_pairs_non_self']} | "
        f"multi-line outbound: {stats['n_with_multiple_outbound_lines']}"
    )


if __name__ == "__main__":
    main()
