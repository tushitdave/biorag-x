"""Retrieval with agentic chunking (question time).

1. Retrieve passages with the flow's search settings on whole passages.
2. Agentic-chunk the top passages (one batched, cached LLM call - chunking/agentic.py).
3. Rank those chunks for the question: MedCPT dense (chunks encoded on the fly with the
   Article Encoder, cached) + BM25 over the same small set, fused with RRF.
4. Rerank and select evidence exactly as the normal pipeline does.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from chunking.agentic import agentic_chunks
from indexes.bm25 import BM25Index
from indexes.dense import encode_medcpt_query
from ingestion.loader import load_chunks
from retrieval.evidence_selector import select_evidence
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank as rerank_fn

N_PASSAGES = 6
_vec_cache: dict[str, np.ndarray] = {}


@lru_cache(maxsize=1)
def _article_encoder():
    from indexes.encoder import ArticleEncoder
    return ArticleEncoder()


@lru_cache(maxsize=1)
def _passage_text() -> dict[str, str]:
    df = load_chunks("passage")
    return dict(zip(df["chunk_id"], df["text"]))


def _chunk_vectors(chunks: list[dict]) -> np.ndarray:
    missing = [c for c in chunks if c["chunk_id"] not in _vec_cache]
    if missing:
        vecs = _article_encoder().encode([c["text"] for c in missing], log=lambda _m: None)
        for c, v in zip(missing, vecs):
            _vec_cache[c["chunk_id"]] = v
    return np.vstack([_vec_cache[c["chunk_id"]] for c in chunks])


def retrieve_agentic(question: str, recipe: dict, reranker_method: str | None = None) -> dict:
    from retrieval.hybrid import retrieve                      # avoid an import cycle
    t0 = time.perf_counter()
    base = {**recipe, "chunking": {"method": "passage"},
            "reranker": {"enabled": False}, "evidence": {"selector": "topk", "top_k": N_PASSAGES}}
    first = retrieve(question, base)
    parents = list(dict.fromkeys(p["parent_passage_id"] for p in first["reranked"]))[:N_PASSAGES]
    texts = _passage_text()
    chunks, meta = agentic_chunks([(pid, texts[pid]) for pid in parents])
    trace = dict(first["trace"]) | {"strategy": "agentic", "agentic": meta,
                                    "agentic_passages": len(parents)}
    if not chunks:
        trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return {"passages": [], "reranked": [], "trace": trace}

    t = time.perf_counter()
    q = encode_medcpt_query(question)
    dense_scores = _chunk_vectors(chunks) @ q
    dense = [dict(chunks[i], dense_score=float(dense_scores[i]))
             for i in np.argsort(-dense_scores)]
    bm25 = BM25Index([c["chunk_id"] for c in chunks], [c["parent_passage_id"] for c in chunks],
                     [c["text"] for c in chunks]).search(question, top_k=len(chunks))
    by_id = {c["chunk_id"]: c for c in chunks}
    bm25 = [dict(by_id[b["chunk_id"]], bm25_score=b["bm25_score"]) for b in bm25]
    fused = reciprocal_rank_fusion([bm25, dense], k=recipe.get("fusion", {}).get("k", 60))
    trace["agentic_rank_ms"] = round((time.perf_counter() - t) * 1000, 1)

    rr = recipe.get("reranker", {})
    method = reranker_method or (rr.get("method", "none") if rr.get("enabled") else "none")
    reranked = rerank_fn(question, fused, method, top_k=len(fused))
    ev = recipe.get("evidence", {})
    k = ev.get("top_k", 5)
    evidence = select_evidence(reranked[:max(10, 2 * k)], top_k=k,
                               diversity=ev.get("selector", "coverage_diversity") != "topk")
    for rank, c in enumerate(evidence):
        c["final_rank"] = rank + 1
    trace["rerank_method"] = method
    trace["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return {"passages": evidence, "reranked": reranked, "trace": trace}
