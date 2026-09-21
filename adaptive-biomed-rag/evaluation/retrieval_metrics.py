"""Retrieval quality metrics: recall@k, hit@k, MRR@10, nDCG@10.

All functions take a ranked list of retrieved parent-passage ids and the set of
gold parent ids for the question. Ranks are 1-based conceptually; the input list
is assumed already ordered best-first. Pure Python + math, fully offline.
"""
from __future__ import annotations

import math
from typing import Iterable


def _dedupe_preserve(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        s = str(i)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(_dedupe_preserve(retrieved)[:k])
    return len(top & gold) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(_dedupe_preserve(retrieved)[:k])
    return 1.0 if (top & gold) else 0.0


def mrr_at_k(retrieved: list[str], gold: set[str], k: int = 10) -> float:
    if not gold:
        return 0.0
    for rank, rid in enumerate(_dedupe_preserve(retrieved)[:k], start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int = 10) -> float:
    """Binary-relevance nDCG@k (gain 1 for gold, 0 otherwise)."""
    if not gold:
        return 0.0
    ranked = _dedupe_preserve(retrieved)[:k]
    dcg = 0.0
    for i, rid in enumerate(ranked):
        if rid in gold:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-based -> rank i+1
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return (dcg / idcg) if idcg > 0 else 0.0


def per_query_metrics(retrieved: list[str], gold: set[str],
                      ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Compute the standard bundle for a single query."""
    gold = {str(g) for g in gold}
    m: dict[str, float] = {}
    for k in ks:
        m[f"recall@{k}"] = round(recall_at_k(retrieved, gold, k), 4)
        m[f"hit@{k}"] = round(hit_at_k(retrieved, gold, k), 4)
    m["mrr@10"] = round(mrr_at_k(retrieved, gold, 10), 4)
    m["ndcg@10"] = round(ndcg_at_k(retrieved, gold, 10), 4)
    return m


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean each metric key across per-query rows."""
    if not rows:
        return {}
    keys = set().union(*[set(r.keys()) for r in rows])
    agg: dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        agg[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return agg
