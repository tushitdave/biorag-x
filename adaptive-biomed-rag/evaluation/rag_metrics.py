"""RAG-specific answer metrics: grounding + coverage from the evidence.

These measure how well the generated answer is anchored to the retrieved
evidence, independent of any gold answer:
  * grounding  : fraction of answer content tokens that appear in the evidence.
  * coverage   : fraction of evidence passages that contribute to the answer.
  * faithfulness (claim-level): fraction of claims that are SUPPORTED.
All are computed offline from the evidence text + generated claims.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "with", "on", "at", "by", "as", "that", "this", "these", "those",
    "be", "been", "it", "its", "from", "which", "who", "can", "may",
}


def _content_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if t.lower() not in _STOP}


def _passage_id(p: dict) -> str:
    return str(p.get("parent_passage_id") or p.get("passage_id") or p.get("chunk_id"))


def grounding_score(answer: str, evidence: list[dict]) -> float:
    """Fraction of answer content tokens present in the union of evidence text."""
    ans = _content_tokens(answer)
    if not ans:
        return 0.0
    ev_tokens: set[str] = set()
    for p in evidence:
        ev_tokens |= _content_tokens(p.get("text", ""))
    return round(len(ans & ev_tokens) / len(ans), 4)


def coverage_score(claims: list, evidence: list[dict]) -> float:
    """Fraction of evidence passages cited by at least one claim."""
    if not evidence:
        return 0.0
    cited: set[str] = set()
    for c in claims:
        cited.update(str(x) for x in getattr(c, "citations", []))
    ev_ids = {_passage_id(p) for p in evidence}
    return round(len(cited & ev_ids) / len(ev_ids), 4)


def faithfulness_score(claims: list) -> float:
    """Fraction of claims labelled SUPPORTED (requires prior validation)."""
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if getattr(c, "support", "") == "SUPPORTED")
    return round(supported / len(claims), 4)


STATUS_SUFFICIENT_GROUNDING = 0.6
STATUS_SUFFICIENT_COVERAGE = 0.4
STATUS_WEAK_GROUNDING = 0.3


def evidence_status_from_coverage(grounding: float, coverage: float) -> str:
    """Deterministic evidence-status label, used when the LLM gives none."""
    if grounding >= STATUS_SUFFICIENT_GROUNDING and coverage >= STATUS_SUFFICIENT_COVERAGE:
        return "sufficient"
    if grounding >= STATUS_WEAK_GROUNDING:
        return "weak"
    return "insufficient"


def rag_answer_metrics(answer: str, claims: list, evidence: list[dict]) -> dict:
    g = grounding_score(answer, evidence)
    cov = coverage_score(claims, evidence)
    faith = faithfulness_score(claims)
    return {
        "grounding": g,
        "coverage": cov,
        "faithfulness": faith,
        "evidence_status": evidence_status_from_coverage(g, cov),
    }
