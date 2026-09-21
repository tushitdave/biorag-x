"""Reciprocal Rank Fusion (RRF) for combining ranked lists.

RRF is rank-based (not score-based), so it robustly fuses BM25 and dense
rankings that live on different scales. score = sum over lists of 1/(k + rank).
"""
from __future__ import annotations


CHANNEL_FIELDS = ("bm25_score", "dense_score", "graph_score", "pageindex_score",
                  "graph_path", "tree_path")


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int = 60,
                           id_key: str = "chunk_id") -> list[dict]:
    """Fuse multiple ranked lists of dicts by RRF.

    Merges the per-channel fields (bm25 / dense / graph / pageindex scores and the
    graph path / tree path) onto the fused
    record and adds `rrf_score`. Returns records sorted by rrf_score desc.
    """
    merged: dict[str, dict] = {}
    rrf: dict[str, float] = {}

    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            cid = item[id_key]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in merged:
                merged[cid] = dict(item)
            else:
                for sk in CHANNEL_FIELDS:
                    if sk in item and item[sk] is not None:
                        merged[cid][sk] = item[sk]

    for cid, item in merged.items():
        item["rrf_score"] = rrf[cid]

    return sorted(merged.values(), key=lambda d: d["rrf_score"], reverse=True)
