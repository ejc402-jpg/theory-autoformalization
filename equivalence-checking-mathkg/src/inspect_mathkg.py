from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path
import networkx as nx


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_node_list(raw):
    """Handle JSON formatted as a list, {'nodes': [...]}, or {id: {...}}."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "nodes" in raw and isinstance(raw["nodes"], list):
            return raw["nodes"]
        return [{"id": k, **v} if isinstance(v, dict) else {"id": k} for k, v in raw.items()]
    raise ValueError("Unsupported node source JSON structure.")


def build_merged_graph(graph_path: Path, node_source_path: Path | None = None):
    raw_graph = load_json(graph_path)
    nodes = raw_graph.get("nodes", [])
    edges = raw_graph.get("edges", raw_graph.get("links", []))

    # Merge Lean 4 metadata from node source file if provided
    extra_by_id = {}
    if node_source_path and node_source_path.exists():
        for rec in extract_node_list(load_json(node_source_path)):
            nid = rec.get("id")
            if nid:
                extra_by_id[nid] = rec

    merged_nodes = []
    G = nx.DiGraph()
    for n in nodes:
        nid = n["id"]
        merged = dict(n)
        if nid in extra_by_id:
            for field in ("lean4_mathlib_name", "lean4_mathlib_import", "lean4_status"):
                if field in extra_by_id[nid] and field not in merged:
                    merged[field] = extra_by_id[nid][field]
        merged_nodes.append(merged)
        G.add_node(nid, **merged)

    normalized_edges = []
    for e in edges:
        src = e.get("source") or e.get("from") or e.get("u")
        tgt = e.get("target") or e.get("to") or e.get("v")
        if src is None or tgt is None:
            continue
        edge_data = {
            "source": src,
            "target": tgt,
            "type": (
                e.get("edge_type")
                or e.get("type")
                or e.get("relationship")
                or e.get("relation")
                or "unknown"
            ),
            "confidence": float(e.get("confidence", 1.0)),
            "reasoning": e.get("reasoning", ""),
            "is_inverse": bool(e.get("is_inverse", e.get("inverse", False))),
        }
        normalized_edges.append(edge_data)
        G.add_edge(src, tgt, **edge_data)

    return G, merged_nodes, normalized_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path("data/mathkg_graph.json"))
    parser.add_argument("--nodes", type=Path, default=Path("data/mathkg_nodes.json"))
    parser.add_argument("--out", type=Path, default=Path("data/mathkg_merged.json"))
    args = parser.parse_args()

    G, merged_nodes, normalized_edges = build_merged_graph(args.graph, args.nodes)

    domains = Counter(d.get("domain", "Unknown") for _, d in G.nodes(data=True))
    node_types = Counter(d.get("type", "Unknown") for _, d in G.nodes(data=True))
    edge_types = Counter(e["type"] for e in normalized_edges)
    isolated = [G.nodes[n].get("name", n) for n in nx.isolates(G)]
    with_lean = sum(1 for n in merged_nodes if n.get("lean4_mathlib_name"))

    print("=== MathKG Summary ===")
    print(f"Nodes: {G.number_of_nodes()} (with lean4_mathlib_name: {with_lean})")
    print(f"Edges: {len(normalized_edges)} (unique DiGraph edges: {G.number_of_edges()})")
    print(f"Node types: {dict(node_types)}")
    print(f"Domains: {dict(domains)}")
    print("Edge types:")
    for k, v in edge_types.most_common():
        print(f"  - {k}: {v}")
    print(f"Isolated nodes ({len(isolated)}): {isolated}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"nodes": merged_nodes, "edges": normalized_edges}, f, indent=2)
    print(f"\nSaved merged graph to {args.out}")


if __name__ == "__main__":
    main()