"""Grounded answer generation -> ChatResponse.

Flow:
  1. Retrieve evidence (via retrieval.hybrid.retrieve) unless supplied.
  2. Build a strict grounded prompt and call common.llm.chat_json.
  3. If the LLM returns structured JSON -> parse + validate citations. An answer
     with no claims (the LLM found nothing citable) is kept and marked insufficient.
  4. If chat_json returns None (disabled / budget / offline cache-miss) or the JSON
     has no answer -> build a DETERMINISTIC EXTRACTIVE answer from the top evidence
     sentences.
  5. Validate citations, compute grounding/coverage/faithfulness, and assemble the
     api.contract.ChatResponse (recording llm_used + llm_source).

No LLM client is ever constructed here — every model call goes through common.llm,
so the budget + cache + master switch are always honored.
"""
from __future__ import annotations

import re

from common.config import load_recipe
from common.llm import chat_json
from retrieval.hybrid import retrieve

from generation.prompts import SYSTEM_PROMPT, build_grounded_prompt
from generation.structured_output import parse_structured_answer
from generation.citation_validator import validate_claims, support_for_passage

from evaluation.rag_metrics import (
    grounding_score, coverage_score, faithfulness_score,
    evidence_status_from_coverage,
)

from api.contract import (
    ChatResponse, Claim, Citation, MetricValue, ProvenanceStep,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "with", "on", "at", "by", "as", "that", "this", "these", "those",
    "what", "which", "who", "how", "why", "when", "where", "does", "do", "did",
}


def _passage_id(p: dict) -> str:
    return str(p.get("parent_passage_id") or p.get("passage_id") or p.get("chunk_id"))


def _content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP}


def _extractive_answer(question: str, evidence: list[dict]) -> tuple[str, list[Claim]]:
    """Deterministic fallback: pick the best-matching sentence(s) from top passages.

    Answer = the single best-overlapping sentence from the top passage plus the
    top sentence from the second passage (if distinct). Claims are one per used
    passage, cited to that passage id.
    """
    q_tokens = _content_tokens(question)
    claims: list[Claim] = []
    answer_parts: list[str] = []

    for p in evidence[:3]:
        pid = _passage_id(p)
        sentences = [s.strip() for s in _SENT_SPLIT.split(p.get("text", "")) if s.strip()]
        if not sentences:
            continue
        # Best sentence by content-token overlap with the question.
        best_sent, best_ov = sentences[0], -1.0
        for s in sentences:
            st = _content_tokens(s)
            ov = (len(q_tokens & st) / len(q_tokens)) if q_tokens else 0.0
            if ov > best_ov:
                best_sent, best_ov = s, ov
        claims.append(Claim(text=best_sent, citations=[pid]))
        if len(answer_parts) < 2:
            answer_parts.append(best_sent)

    if not answer_parts:
        return ("The retrieved evidence does not contain enough information to "
                "answer this question."), []
    return " ".join(answer_parts), claims


def _build_citations(evidence: list[dict], claims: list[Claim]) -> list[Citation]:
    citations: list[Citation] = []
    for p in evidence:
        pid = _passage_id(p)
        scores = {
            "bm25": p.get("bm25_score"),
            "dense": p.get("dense_score"),
            "graph": p.get("graph_score"),
            "pageindex": p.get("pageindex_score"),
            "rrf": p.get("rrf_score"),
            "rerank": p.get("rerank_score"),
            "final_rank": p.get("final_rank"),
        }
        scores = {k: v for k, v in scores.items() if v is not None}
        provenance = [
            ProvenanceStep(level="passage", label=f"Passage {pid}", ref_id=pid),
            ProvenanceStep(level="chunk", label=f"Chunk {p.get('chunk_id')}",
                           ref_id=str(p.get("chunk_id"))),
        ]
        citations.append(Citation(
            passage_id=pid,
            text=p.get("text", ""),
            support=support_for_passage(claims, pid),
            scores=scores,
            provenance=provenance,
        ))
    return citations


def generate_answer(question: str, recipe: str = "hybrid_rerank", top_k: int = 5,
                    retrieval_result: dict | None = None,
                    reranker: str | None = None) -> ChatResponse:
    """Produce a grounded ChatResponse. Uses at most one LLM call (via chat_json)."""
    recipe_dict = load_recipe(recipe)

    if retrieval_result is None:
        retrieval_result = retrieve(question, recipe_dict, top_k=top_k,
                                    reranker_method=reranker)
    evidence = retrieval_result.get("passages", [])[:top_k]
    trace = retrieval_result.get("trace", {})

    # --- Try the LLM (guarded/cached/budgeted); may return None. ---
    prompt = build_grounded_prompt(question, evidence)
    data, _raw, source = chat_json(prompt, system=SYSTEM_PROMPT, max_tokens=700)

    llm_used = source in ("live", "cache")
    if data is not None:
        answer, claims, status = parse_structured_answer(data)
        if not answer:                    # LLM produced unusable output -> fallback
            answer, claims = _extractive_answer(question, evidence)
            status = ""
            llm_used = False
            source = "unusable"
        elif not claims:                  # LLM declined: nothing citable in the evidence
            status = "insufficient"
    else:
        answer, claims = _extractive_answer(question, evidence)
        status = ""

    # --- Validate citations + compute grounding metrics (always, offline). ---
    claims, cite_metrics = validate_claims(claims, evidence)
    g = grounding_score(answer, evidence)
    cov = coverage_score(claims, evidence)
    faith = faithfulness_score(claims)
    if not status:
        status = evidence_status_from_coverage(g, cov)

    citations = _build_citations(evidence, claims)

    grounding = [
        MetricValue(key="grounding", label="Grounding", value=round(g * 100, 1),
                    unit="%", definition="Share of answer tokens found in the evidence.",
                    source="live" if llm_used else "artifact"),
        MetricValue(key="coverage", label="Evidence coverage", value=round(cov * 100, 1),
                    unit="%", definition="Share of evidence passages cited by a claim."),
        MetricValue(key="faithfulness", label="Faithfulness", value=round(faith * 100, 1),
                    unit="%", definition="Share of claims fully supported by their citations."),
        MetricValue(key="citation_precision", label="Citation precision",
                    value=round(cite_metrics["citation_precision"] * 100, 1), unit="%",
                    definition="Share of cited passages that actually support the claim."),
        MetricValue(key="citation_recall", label="Citation recall",
                    value=round(cite_metrics["citation_recall"] * 100, 1), unit="%",
                    definition="Share of evidence passages used by supported claims."),
    ]

    return ChatResponse(
        question=question,
        answer=answer,
        evidence_status=status,
        claims=claims,
        citations=citations,
        grounding=grounding,
        llm_used=llm_used,
        llm_source=source,
        retrieval_trace=trace,
    )
