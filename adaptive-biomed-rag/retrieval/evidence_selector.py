"""Evidence selection: choose a SET of passages, not just the top-k by score.

Greedy relevance + diversity (MMR-style): pick the highest-scoring candidate,
then iteratively add the candidate that best trades relevance against redundancy
with what's already selected. Prevents near-duplicate evidence and improves
coverage of distinct facts.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _toks(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def _score(c: dict) -> float:
    for k in ("rerank_score", "rrf_score", "dense_score", "bm25_score", "graph_score",
              "pageindex_score"):
        if c.get(k) is not None:
            return float(c[k])
    return 0.0


MMR_LAMBDA = 0.7   # weight on relevance vs redundancy with already-selected evidence


def select_evidence(candidates: list[dict], top_k: int = 5, diversity: bool = True,
                    lambda_: float = MMR_LAMBDA) -> list[dict]:
    if not candidates:
        return []
    if not diversity:
        return candidates[:top_k]

    # Normalize base relevance to [0,1] for stable MMR mixing.
    scores = [_score(c) for c in candidates]
    lo, hi = min(scores), max(scores)
    rng = (hi - lo) or 1.0
    rel = {id(c): (s - lo) / rng for c, s in zip(candidates, scores)}
    tok = {id(c): _toks(c["text"]) for c in candidates}

    selected: list[dict] = []
    pool = list(candidates)
    while pool and len(selected) < top_k:
        best, best_val = None, -1e9
        for c in pool:
            redundancy = 0.0
            for s in selected:
                a, b = tok[id(c)], tok[id(s)]
                if a and b:
                    redundancy = max(redundancy, len(a & b) / len(a | b))
            mmr = lambda_ * rel[id(c)] - (1 - lambda_) * redundancy
            if mmr > best_val:
                best, best_val = c, mmr
        selected.append(best)
        pool.remove(best)
    return selected
