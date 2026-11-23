from __future__ import annotations
import os
from parse_lean import extract_lean_identifiers


def score_raw_overlap(proof_a: str, proof_b: str) -> float:
    """Jaccard similarity over raw extracted Lean identifiers (no MathKG)."""
    set_a = extract_lean_identifiers(proof_a)
    set_b = extract_lean_identifiers(proof_b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def llm_judge_equivalent(proof_a: str, proof_b: str, model: str = "gpt-4o-mini") -> int | None:
    """
    Optional LLM-as-a-judge baseline. Returns 1 (equivalent), 0 (not equivalent),
    or None if OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = (
        "Determine if the following two Lean 4 proofs represent the same underlying "
        "mathematical proof strategy (e.g., one simply expands a lemma cited by the other). "
        "Reply with ONLY 'YES' or 'NO'.\n\n"
        f"Proof A:\n{proof_a}\n\nProof B:\n{proof_b}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5,
    )
    ans = resp.choices[0].message.content.strip().upper()
    return 1 if ans.startswith("YES") else 0