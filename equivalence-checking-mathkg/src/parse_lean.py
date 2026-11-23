import re
from typing import Dict, List, Set, Tuple
import networkx as nx

LEAN_KEYWORDS = {
    "theorem", "lemma", "def", "example", "by", "have", "let", "show", "from",
    "intro", "intros", "exact", "apply", "rw", "rwa", "simp", "simp_all",
    "ring", "ring_nf", "linarith", "nlinarith", "omega", "norm_num", "positivity",
    "cases", "rcases", "obtain", "induction", "with", "where", "fun", "match",
    "if", "then", "else", "calc", "constructor", "use", "refine", "trivial",
    "assumption", "contradiction", "exfalso", "push_neg", "by_contra", "by_cases",
    "specialize", "generalize", "clear", "revert", "rename_i", "ext", "congr",
    "field_simp", "aesop", "tauto", "decide", "Prop", "Type", "Nat", "Int",
    "Real", "Rat", "Bool", "Set", "Finset", "Matrix", "Polynomial", "sorry"
}


def strip_lean_comments(code: str) -> str:
    code = re.sub(r"/-.*?-/", " ", code, flags=re.DOTALL)
    code = re.sub(r"--[^\n]*", " ", code)
    return code


def extract_lean_identifiers(proof_code: str) -> Set[str]:
    """Extract candidate Mathlib declaration names from a Lean 4 proof body."""
    cleaned = strip_lean_comments(proof_code)
    # Remove theorem header up to ':= by' or ':=' if present
    if ":=" in cleaned:
        cleaned = cleaned.split(":=", 1)[1]

    # Match dotted identifiers like IsAntichain.sperner or Matrix.aeval_self_charpoly
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*\b", cleaned)
    identifiers = set()
    for tok in tokens:
        if tok in LEAN_KEYWORDS:
            continue
        # Skip single-character local variable names (h, x, n, p, q)
        if len(tok) <= 2 and "." not in tok:
            continue
        identifiers.add(tok)
    return identifiers


class MathKGMapper:
    """Maps Lean 4 identifiers to MathKG node IDs."""

    def __init__(self, G: nx.DiGraph):
        self.G = G
        self.exact_map: Dict[str, str] = {}
        self.suffix_map: Dict[str, List[str]] = {}
        self.keyword_map: Dict[str, List[str]] = {}

        for nid, data in G.nodes(data=True):
            lean_name = (data.get("lean4_mathlib_name") or "").strip()
            if lean_name:
                self.exact_map[lean_name] = nid
                suffix = lean_name.split(".")[-1]
                self.suffix_map.setdefault(suffix, []).append(nid)

            for kw in data.get("keywords", []):
                k = kw.lower().replace(" ", "_").replace("-", "_")
                if len(k) >= 4:
                    self.keyword_map.setdefault(k, []).append(nid)

    def map_proof(self, proof_code: str) -> Tuple[Set[str], Dict[str, float]]:
        identifiers = extract_lean_identifiers(proof_code)
        matched_nodes: Set[str] = set()
        direct_hits = 0
        fallback_hits = 0

        for ident in identifiers:
            if ident in self.exact_map:
                matched_nodes.add(self.exact_map[ident])
                direct_hits += 1
                continue

            suffix = ident.split(".")[-1]
            if suffix in self.suffix_map and len(self.suffix_map[suffix]) == 1:
                matched_nodes.add(self.suffix_map[suffix][0])
                direct_hits += 1
                continue

            # Keyword substring fallback
            ident_low = ident.lower()
            for kw, nids in self.keyword_map.items():
                if kw in ident_low and len(nids) == 1:
                    matched_nodes.add(nids[0])
                    fallback_hits += 1
                    break

        total = len(identifiers)
        stats = {
            "num_identifiers": total,
            "direct_hits": direct_hits,
            "fallback_hits": fallback_hits,
            "coverage": (direct_hits + fallback_hits) / total if total > 0 else 0.0,
        }
        return matched_nodes, stats