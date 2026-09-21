"""BioASQ-style answer scoring: token-level F1 (and exact match).

Token F1 is the standard overlap metric for short-form biomedical answers: it
compares the bag of content tokens in the predicted answer against the reference
answer, rewarding overlap without requiring an exact string match.
"""
from __future__ import annotations

import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_ARTICLES = {"a", "an", "the"}


def _normalize(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN.findall(text or "")]
    return [t for t in toks if t not in _ARTICLES]


def token_f1(prediction: str, reference: str) -> float:
    pred = _normalize(prediction)
    ref = _normalize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def best_token_f1(prediction: str, references: list[str]) -> float:
    """Max token-F1 over multiple acceptable reference answers."""
    if not references:
        return 0.0
    return max(token_f1(prediction, r) for r in references)


def aggregate_token_f1(pairs: list[tuple[str, list[str]]]) -> float:
    """Mean best-token-F1 over (prediction, references) pairs."""
    if not pairs:
        return 0.0
    return round(sum(best_token_f1(p, r) for p, r in pairs) / len(pairs), 4)
