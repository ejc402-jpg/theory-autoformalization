from typing import Dict, Set, Tuple
import networkx as nx

DEDUCTIVE_EDGE_TYPES = {
    "prerequisite",
    "applied-in",
    "generalizes",
    "specializes",
    "instance-of",
    "equivalent-to",
}


def filter_deductive_graph(G: nx.DiGraph, min_conf: float = 0.75) -> nx.DiGraph:
    """Keep only deductive/structural edges above confidence threshold."""
    H = nx.DiGraph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        if d.get("type") in DEDUCTIVE_EDGE_TYPES and float(d.get("confidence", 1.0)) >= min_conf:
            H.add_edge(u, v, **d)
    return H


def prerequisite_closure(
    H: nx.DiGraph,
    seed_nodes: Set[str],
    max_hops: int = 1,
    decay: float = 0.7,
) -> Dict[str, float]:
    """
    Expand each node v in seed_nodes along incoming 'prerequisite' and
    'equivalent-to' edges up to max_hops, assigning confidence-weighted scores.
    """
    weights: Dict[str, float] = {n: 1.0 for n in seed_nodes if n in H}
    frontier = set(weights.keys())

    for hop in range(1, max_hops + 1):
        next_frontier = set()
        for v in frontier:
            # Incoming prerequisite edges: u --(prerequisite)--> v means u is used to prove v
            for u, _, d in H.in_edges(v, data=True):
                etype = d.get("type")
                if etype in ("prerequisite", "equivalent-to", "specializes"):
                    w = weights[v] * float(d.get("confidence", 1.0)) * (decay ** hop)
                    if w > weights.get(u, 0.0):
                        weights[u] = w
                        next_frontier.add(u)
            # Also follow outgoing 'equivalent-to' edges
            for _, u, d in H.out_edges(v, data=True):
                if d.get("type") == "equivalent-to":
                    w = weights[v] * float(d.get("confidence", 1.0))
                    if w > weights.get(u, 0.0):
                        weights[u] = w
                        next_frontier.add(u)
        frontier = next_frontier
    return weights


def weighted_jaccard(w_a: Dict[str, float], w_b: Dict[str, float]) -> float:
    all_keys = set(w_a.keys()) | set(w_b.keys())
    if not all_keys:
        return 0.0
    num = sum(min(w_a.get(k, 0.0), w_b.get(k, 0.0)) for k in all_keys)
    den = sum(max(w_a.get(k, 0.0), w_b.get(k, 0.0)) for k in all_keys)
    return num / den if den > 0 else 0.0


def score_prerequisite_closure(
    H: nx.DiGraph,
    nodes_a: Set[str],
    nodes_b: Set[str],
    max_hops: int = 1,
) -> float:
    w_a = prerequisite_closure(H, nodes_a, max_hops=max_hops)
    w_b = prerequisite_closure(H, nodes_b, max_hops=max_hops)
    return weighted_jaccard(w_a, w_b)


def contract_to_target_nodes(
    H: nx.DiGraph,
    source_nodes: Set[str],
    target_nodes: Set[str],
) -> Set[str]:
    """
    If source_nodes contains prerequisite children {M1, M2} of a parent node L2
    in target_nodes, contract those children into L2.
    """
    contracted = set(source_nodes)
    for target in target_nodes:
        if target not in H or target in contracted:
            continue
        prereqs = {
            u for u, _, d in H.in_edges(target, data=True)
            if d.get("type") in ("prerequisite", "equivalent-to")
        }
        overlap = contracted & prereqs
        if overlap:
            contracted -= overlap
            contracted.add(target)
    return contracted


def score_subgraph_contraction(
    H: nx.DiGraph,
    nodes_a: Set[str],
    nodes_b: Set[str],
) -> Tuple[float, Set[str], Set[str]]:
    """Bidirectional contraction followed by Jaccard similarity."""
    c_a = contract_to_target_nodes(H, nodes_a, nodes_b)
    c_b = contract_to_target_nodes(H, nodes_b, nodes_a)
    union = c_a | c_b
    score = len(c_a & c_b) / len(union) if union else 0.0
    return score, c_a, c_b