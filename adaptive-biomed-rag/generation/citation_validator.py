"""Citation validation for grounded answers.

For each claim citation we check two things:
  1. The cited passage_id is actually in the selected evidence set (no phantom
     citations to passages the retriever never surfaced).
  2. The claim's content tokens overlap the cited passage text, so the citation
     is topically anchored — not just a valid id slapped onto an unrelated claim.

Each claim is labelled SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED, and we
compute citation precision (fraction of cited passages that are valid+overlapping)
and recall (fraction of evidence passages actually used by supported claims).
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "with", "on", "at", "by", "as", "that", "this", "these", "those",
    "be", "been", "it", "its", "from", "which", "who", "whom", "can", "may",
}

# Overlap thresholds (fraction of claim content tokens found in the passage).
_SUPPORTED = 0.5
_PARTIAL = 0.2


def _content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP}


def _passage_id(p: dict) -> str:
    return str(p.get("parent_passage_id") or p.get("passage_id") or p.get("chunk_id"))


def validate_claims(claims: list, evidence: list[dict]) -> tuple[list, dict]:
    """Annotate each claim with a support label and return (claims, metrics).

    ``claims`` are contract ``Claim`` objects (mutated in place with .support and
    filtered .citations). Returns citation precision/recall + support histogram.
    """
    ev_index: dict[str, set[str]] = {}
    for p in evidence:
        ev_index[_passage_id(p)] = _content_tokens(p.get("text", ""))
    ev_ids = set(ev_index.keys())

    total_citations = 0
    valid_citations = 0
    used_ev_ids: set[str] = set()
    support_hist = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0}

    for claim in claims:
        claim_tokens = _content_tokens(claim.text)
        best_overlap = 0.0
        kept_citations: list[str] = []
        for cid in claim.citations:
            total_citations += 1
            if cid not in ev_ids:
                continue  # phantom citation -> dropped, not valid
            passage_tokens = ev_index[cid]
            if claim_tokens:
                overlap = len(claim_tokens & passage_tokens) / len(claim_tokens)
            else:
                overlap = 0.0
            best_overlap = max(best_overlap, overlap)
            if overlap >= _PARTIAL:
                valid_citations += 1
                kept_citations.append(cid)
                used_ev_ids.add(cid)
            else:
                kept_citations.append(cid)  # keep id but it lowers precision

        claim.citations = kept_citations
        if not kept_citations:
            claim.support = "UNSUPPORTED"
        elif best_overlap >= _SUPPORTED:
            claim.support = "SUPPORTED"
        elif best_overlap >= _PARTIAL:
            claim.support = "PARTIALLY_SUPPORTED"
        else:
            claim.support = "UNSUPPORTED"
        support_hist[claim.support] += 1

    precision = (valid_citations / total_citations) if total_citations else 0.0
    recall = (len(used_ev_ids) / len(ev_ids)) if ev_ids else 0.0

    metrics = {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "evidence_used": len(used_ev_ids),
        "evidence_total": len(ev_ids),
        "support_histogram": support_hist,
    }
    return claims, metrics


def support_for_passage(claims: list, passage_id: str) -> str:
    """Aggregate the strongest support any claim gives to a specific passage."""
    order = {"UNSUPPORTED": 0, "PARTIALLY_SUPPORTED": 1, "SUPPORTED": 2}
    best = "UNSUPPORTED"
    for claim in claims:
        if passage_id in claim.citations and order[claim.support] > order[best]:
            best = claim.support
    return best
