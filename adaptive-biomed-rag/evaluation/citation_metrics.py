"""Citation-quality aggregation across a set of answered questions.

Wraps the per-answer output of generation.citation_validator into corpus-level
precision/recall/F1 and a support-label distribution.
"""
from __future__ import annotations


def citation_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def aggregate_citation_metrics(per_answer: list[dict]) -> dict:
    """Aggregate per-answer citation metric dicts (from validate_claims).

    Each item is expected to have citation_precision, citation_recall,
    support_histogram. Returns mean precision/recall/F1 + summed histogram.
    """
    if not per_answer:
        return {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "citation_f1": 0.0,
            "support_histogram": {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0},
        }

    p = sum(a.get("citation_precision", 0.0) for a in per_answer) / len(per_answer)
    r = sum(a.get("citation_recall", 0.0) for a in per_answer) / len(per_answer)
    hist = {"SUPPORTED": 0, "PARTIALLY_SUPPORTED": 0, "UNSUPPORTED": 0}
    for a in per_answer:
        for k, v in (a.get("support_histogram") or {}).items():
            hist[k] = hist.get(k, 0) + int(v)

    return {
        "citation_precision": round(p, 4),
        "citation_recall": round(r, 4),
        "citation_f1": round(citation_f1(p, r), 4),
        "support_histogram": hist,
    }
