"""Graph traversal and PageIndex as search channels next to BM25 and dense.

Both rank *passages* (graph/evidence_graph.py, pageindex/tree.py). For a chunking
strategy other than whole passages, each ranked passage is represented by its chunk
that best matches the question (BM25 over that strategy's chunks), so the channel's
list fuses with BM25 / dense lists by RRF like any other. Only passages in the chosen
corpus (full | dev) are returned.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from common import paths
from indexes.bm25 import get_bm25

INDEX_DIRS = {"graph": paths.GRAPH_DIR, "pageindex": paths.PAGEINDEX_DIR, "synonyms": paths.SYNONYM_DIR}


def is_ready(channel: str) -> bool:
    return (INDEX_DIRS[channel] / "manifest.json").exists()


def manifest(channel: str) -> dict:
    try:
        return json.loads((INDEX_DIRS[channel] / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@lru_cache(maxsize=1)
def get_graph():
    from graph.evidence_graph import EvidenceGraph
    return EvidenceGraph(paths.GRAPH_DIR)


@lru_cache(maxsize=1)
def get_synonyms():
    from retrieval.synonyms import SynonymExpander
    return SynonymExpander(paths.SYNONYM_DIR)


@lru_cache(maxsize=1)
def get_tree():
    from pageindex.tree import PageIndexTree
    return PageIndexTree(paths.PAGEINDEX_DIR)


@lru_cache(maxsize=24)
def _by_parent(strategy: str, corpus: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, pid in enumerate(get_bm25(strategy, corpus).parent_ids):
        out.setdefault(pid, []).append(i)
    return out


def _to_units(question: str, ranked: list[dict], strategy: str, corpus: str,
              top_k: int) -> list[dict]:
    """Passage ranking -> this strategy's chunks (best-matching chunk per passage)."""
    bm = get_bm25(strategy, corpus)
    by_parent = _by_parent(strategy, corpus)
    scores = bm.scores(question) if strategy != "passage" else None
    out = []
    for r in ranked:
        idx = by_parent.get(r["parent_passage_id"])
        if not idx:
            continue                                   # not in this corpus / strategy
        i = idx[int(np.argmax(scores[idx]))] if scores is not None else idx[0]
        out.append({**r, "chunk_id": bm.chunk_ids[i], "text": bm.texts[i]})
        if len(out) >= top_k:
            break
    return out


def graph_channel(question: str, strategy: str, corpus: str, top_k: int) -> tuple[list[dict], dict]:
    ranked, info = get_graph().search(question)
    return _to_units(question, ranked, strategy, corpus, top_k), info


def pageindex_channel(question: str, strategy: str, corpus: str,
                      top_k: int) -> tuple[list[dict], dict]:
    tree = get_tree()
    allowed = _by_parent(strategy, corpus)
    read, info = tree.navigate(question, allowed=allowed, top_k=top_k)
    if not read:
        return [], info
    # read the chosen leaves: rank their passages by BM25 (whole passages, this corpus)
    bm = get_bm25("passage", corpus)
    pos = _by_parent("passage", corpus)
    scores = bm.scores(question)
    cands = []
    for order, (leaf, pids) in enumerate(read):
        path = tree.path(leaf)
        for p in pids:
            if p in pos:
                cands.append((float(scores[pos[p][0]]), -order, p, path))
    cands.sort(reverse=True)
    ranked = [{"parent_passage_id": p, "pageindex_score": round(s, 4), "tree_path": path}
              for s, _, p, path in cands]
    info["passages_read"] = len(cands)
    return _to_units(question, ranked, strategy, corpus, top_k), info
