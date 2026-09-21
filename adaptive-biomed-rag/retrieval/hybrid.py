"""Hybrid retrieval orchestrator.

Runs the configured channels (BM25 lexical, dense vector, graph traversal, PageIndex
tree navigation - see retrieval/structured.py), fuses with RRF,
reranks, and selects an evidence set. Returns rich per-passage score provenance
(bm25 / dense / rrf / rerank / final_rank) matching the API contract so the
frontend can show the score journey, plus per-stage timings. All-local; the only
optional network call is ada-002 query embedding (cached) for that dense model.

Recipe keys read here: chunking.method, embedding.model, ann.{index, ef_search,
nprobe}, retrieval.{channels, top_k}, fusion.k, reranker.{enabled, method,
candidates}, evidence.{selector, top_k}, and top-level ``corpus`` (full | dev).

Query mode (recipe ``query.mode``): ``as_is``; ``synonyms`` expands corpus-defined
abbreviations for the keyword-based channels (retrieval/synonyms.py); ``router`` hands
the whole retrieval to retrieval/router.py, which picks channels per question.

Chunking ``agentic`` is delegated to retrieval/agentic.py (chunks are made at question
time). For ``parent_child`` the children are searched and reranked, then each is
replaced by its whole parent passage (deduplicated) before evidence selection.
"""
from __future__ import annotations

import time
from functools import lru_cache

from indexes.bm25 import get_bm25
from indexes.dense import get_dense
from ingestion.loader import load_chunks
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank as rerank_fn
from retrieval.evidence_selector import select_evidence


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


@lru_cache(maxsize=1)
def _passage_text() -> dict[str, str]:
    df = load_chunks("passage")
    return dict(zip(df["chunk_id"], df["text"]))


def _expand_to_parents(ranked: list[dict]) -> list[dict]:
    """Parent-child: keep each parent's best child, send the parent passage text."""
    texts, seen, out = _passage_text(), set(), []
    for c in ranked:
        pid = c["parent_passage_id"]
        if pid in seen:
            continue
        seen.add(pid)
        out.append(dict(c, child_text=c["text"], text=texts.get(pid, c["text"])))
    return out


def _top_parents(results: list[dict], n: int = 10) -> list[str]:
    return list(dict.fromkeys(r["parent_passage_id"] for r in results))[:n]


def retrieve(question: str, recipe: dict, top_k: int = 10,
             reranker_method: str | None = None) -> dict:
    """Execute retrieval per recipe. Returns a dict with passages + trace.

    ``reranked`` is the whole re-sorted candidate pool (so ranking metrics can look
    past the evidence set); ``passages`` is the selected evidence.
    """
    query_mode = recipe.get("query", {}).get("mode", "as_is")
    if query_mode == "router":
        from retrieval.router import retrieve_routed
        return retrieve_routed(question, recipe, reranker_method)
    strategy = recipe.get("chunking", {}).get("method", "passage")
    if strategy == "agentic":
        from retrieval.agentic import retrieve_agentic
        return retrieve_agentic(question, recipe, reranker_method)

    t0 = time.perf_counter()
    corpus = recipe.get("corpus", "full")
    channels = recipe.get("retrieval", {}).get("channels", ["lexical", "dense"])
    channel_k = recipe.get("retrieval", {}).get("top_k", 40)
    fusion_k = recipe.get("fusion", {}).get("k", 60)
    ann = recipe.get("ann", {})
    index_type = ann.get("index", "flat")
    embed_model = recipe.get("embedding", {}).get("model", "medcpt")
    rr = recipe.get("reranker", {})
    rerank_method = (reranker_method or
                     (rr.get("method", "lexical_overlap") if rr.get("enabled", True) else "none"))
    rerank_pool = rr.get("candidates", rr.get("top_k", 10) * 4)
    ev = recipe.get("evidence", {})
    evidence_k = ev.get("top_k", 5)
    evidence_div = ev.get("selector", "coverage_diversity") != "topk"

    trace = {"channels": channels, "index_type": index_type, "corpus": corpus,
             "rerank_method": rerank_method, "strategy": strategy}
    ranked_lists = []

    keywords = question                  # text for the keyword-based channels
    if query_mode == "synonyms":
        from retrieval import structured
        keywords, added = structured.get_synonyms().expand(question)
        trace["synonyms"] = added
    trace["query_mode"] = query_mode

    if "lexical" in channels:
        t = time.perf_counter()
        bm = get_bm25(strategy, corpus).search(keywords, top_k=channel_k)
        trace["bm25_ms"] = _ms(t)
        trace["bm25_hits"] = len(bm)
        trace["bm25_top"] = _top_parents(bm)
        ranked_lists.append(bm)

    dense_available = True
    if "dense" in channels:
        t = time.perf_counter()
        di = get_dense(strategy, index_type, embed_model, corpus)
        dn = di.search(question, top_k=channel_k, ef_search=ann.get("ef_search"),
                       nprobe=ann.get("nprobe"))
        trace["dense_ms"] = _ms(t)
        trace["dense_hits"] = len(dn)
        trace["dense_top"] = _top_parents(dn)
        trace["dense_index_type"] = di.index_type      # may fall back flat<-hnsw
        trace["embedding_model"] = embed_model
        dense_available = len(dn) > 0
        ranked_lists.append(dn)

    for name in ("graph", "pageindex"):
        if name not in channels:
            continue
        from retrieval import structured
        t = time.perf_counter()
        run = structured.graph_channel if name == "graph" else structured.pageindex_channel
        hits, info = run(keywords, strategy, corpus, channel_k)
        trace[f"{name}_ms"] = _ms(t)
        trace[f"{name}_hits"] = len(hits)
        trace[f"{name}_top"] = _top_parents(hits)
        trace[name] = info
        ranked_lists.append(hits)

    if len(ranked_lists) > 1:
        fused = reciprocal_rank_fusion(ranked_lists, k=fusion_k)
    elif ranked_lists:
        fused = ranked_lists[0]
    else:
        fused = []

    t = time.perf_counter()
    pool = fused[:rerank_pool]
    reranked = rerank_fn(question, pool, rerank_method, top_k=len(pool))
    trace["rerank_ms"] = _ms(t)
    if strategy == "parent_child":
        reranked = _expand_to_parents(reranked)
    evidence = select_evidence(reranked[:max(10, 2 * evidence_k)], top_k=evidence_k,
                               diversity=evidence_div)

    for rank, c in enumerate(evidence):
        c["final_rank"] = rank + 1

    trace["dense_query_embedding_available"] = dense_available
    trace["latency_ms"] = _ms(t0)
    trace["fused_pool"] = len(fused)
    return {"passages": evidence, "reranked": reranked, "trace": trace}
