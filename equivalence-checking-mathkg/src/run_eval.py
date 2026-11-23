import json
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from inspect_mathkg import build_merged_graph
from parse_lean import MathKGMapper
from graph_matcher import (
    filter_deductive_graph,
    score_prerequisite_closure,
    score_subgraph_contraction,
)
from baselines import score_raw_overlap, llm_judge_equivalent


def eval_predictions(y_true, y_pred):
    return {
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=Path("data/mathkg_merged.json"))
    parser.add_argument("--benchmark", type=Path, default=Path("data/toy_benchmark.json"))
    parser.add_argument("--out", type=Path, default=Path("results/toy_results.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()

    G, _, _ = build_merged_graph(args.graph, None)
    H = filter_deductive_graph(G, min_conf=0.75)
    mapper = MathKGMapper(H)

    with open(args.benchmark, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    rows = []
    for item in pairs:
        nodes_a, stats_a = mapper.map_proof(item["proof_a"])
        nodes_b, stats_b = mapper.map_proof(item["proof_b"])

        raw_s = score_raw_overlap(item["proof_a"], item["proof_b"])
        hop1_s = score_prerequisite_closure(H, nodes_a, nodes_b, max_hops=1)
        hop2_s = score_prerequisite_closure(H, nodes_a, nodes_b, max_hops=2)
        cont_s, _, _ = score_subgraph_contraction(H, nodes_a, nodes_b)
        llm_pred = llm_judge_equivalent(item["proof_a"], item["proof_b"]) if args.use_llm else None

        rows.append({
            "id": item["id"],
            "label": item["label"],
            "nodes_a": len(nodes_a),
            "nodes_b": len(nodes_b),
            "avg_coverage": round((stats_a["coverage"] + stats_b["coverage"]) / 2, 3),
            "raw_score": round(raw_s, 3),
            "hop1_score": round(hop1_s, 3),
            "hop2_score": round(hop2_s, 3),
            "contraction_score": round(cont_s, 3),
            "llm_pred": llm_pred,
        })

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    y_true = df["label"].tolist()
    t = args.threshold
    summary = [
        {"Method": "Raw Lemma Overlap", **eval_predictions(y_true, (df["raw_score"] >= t).astype(int))},
        {"Method": "MathKG 1-Hop Closure", **eval_predictions(y_true, (df["hop1_score"] >= t).astype(int))},
        {"Method": "MathKG 2-Hop Closure", **eval_predictions(y_true, (df["hop2_score"] >= t).astype(int))},
        {"Method": "MathKG Contraction", **eval_predictions(y_true, (df["contraction_score"] >= t).astype(int))},
    ]
    if args.use_llm and df["llm_pred"].notna().all():
        summary.append({"Method": "LLM-as-a-Judge", **eval_predictions(y_true, df["llm_pred"].astype(int))})

    print(f"\n=== Results (threshold={t}, avg MathKG node coverage={df['avg_coverage'].mean():.1%}) ===")
    print(pd.DataFrame(summary).to_markdown(index=False))
    print(f"\nPer-pair details saved to {args.out}")


if __name__ == "__main__":
    main()