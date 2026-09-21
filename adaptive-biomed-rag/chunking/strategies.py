"""Deterministic chunking strategies, ported from the notebooks.

  fixed        NB03  256-token windows, 32 overlap
  recursive    NB03  paragraph -> sentence -> token, <= 256 tokens, merge tiny units (< 32)
  semantic     NB04  break where adjacent-sentence similarity is low (<= 384, merge < 40).
                     NB04 used ada-002 sentence vectors; here the caller supplies
                     local MedCPT sentence similarities (see scripts/build_chunk_indexes.py)
  biomedical   NB04  sentence packing that keeps relation/negation sentences attached (<= 384)
  proposition  NB04  sentences split on clause conjunctions (<= 96 tokens)
  parent_child NB04  96-token children, 16 overlap; generation gets the parent passage
  late         NB04  256/32 spans; vectors pooled from one contextual pass (late chunking)
  adaptive     NB05  rule-based router picks one of the above per passage from its profile

Every function takes (parent_id, text) and returns chunk dicts (see base.make_chunk).
"""
from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np

from chunking.base import (
    PATTERNS, make_chunk, n_tokens, sentences, tokens, windows,
)

# Parameters (words = whitespace tokens). Read by the chunkers and by the
# architecture diagrams, so the documented values are the ones in use.
FIXED_SIZE, FIXED_OVERLAP = 256, 32
RECURSIVE_MAX, RECURSIVE_MIN = 256, 32
SEMANTIC_MAX, SEMANTIC_MIN = 384, 40
BIOMEDICAL_MAX, BIOMEDICAL_STRETCH = 384, 1.15
PROPOSITION_MAX = 96
CHILD_SIZE, CHILD_OVERLAP = 96, 16
LATE_SIZE, LATE_OVERLAP = 256, 32

# Semantic boundaries: the weakest SEMANTIC_PERCENTILE% of adjacent-sentence similarities
# across the corpus become break points (threshold calibrated at build time).
SEMANTIC_PERCENTILE = 20

# Adaptive router thresholds (NB05 route_rule_based).
ROUTE_SHORT_TOKENS = 180
ROUTE_RELATION_DENSITY = 0.015
ROUTE_ENTITY_DENSITY = 0.012
ROUTE_TOPIC_SHIFT = 0.65
ROUTE_LONG_TOKENS = 700


def fixed_chunks(parent_id: str, text: str, size: int = FIXED_SIZE,
                 overlap: int = FIXED_OVERLAP) -> list[dict]:
    toks = tokens(text)
    return [make_chunk(parent_id, "fixed", i, " ".join(toks[a:b]), a, b)
            for i, (a, b) in enumerate(windows(len(toks), size, overlap))]


def recursive_chunks(parent_id: str, text: str, max_tokens: int = RECURSIVE_MAX,
                     min_tokens: int = RECURSIVE_MIN) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    units: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        p_toks = tokens(paragraph)
        if len(p_toks) <= max_tokens:
            units.append(paragraph)
            continue
        sents = sentences(paragraph)
        if len(sents) <= 1:
            units += [" ".join(p_toks[i:i + max_tokens]) for i in range(0, len(p_toks), max_tokens)]
            continue
        current, current_len = [], 0
        for s in sents:
            s_toks = tokens(s)
            if len(s_toks) > max_tokens:
                if current:
                    units.append(" ".join(current))
                    current, current_len = [], 0
                units += [" ".join(s_toks[i:i + max_tokens]) for i in range(0, len(s_toks), max_tokens)]
                continue
            if current and current_len + len(s_toks) > max_tokens:
                units.append(" ".join(current))
                current, current_len = [s], len(s_toks)
            else:
                current.append(s)
                current_len += len(s_toks)
        if current:
            units.append(" ".join(current))

    merged: list[str] = []                      # merge tiny adjacent units where possible
    for unit in units:
        if merged and n_tokens(unit) < min_tokens and n_tokens(f"{merged[-1]} {unit}") <= max_tokens:
            merged[-1] = f"{merged[-1]} {unit}"
        else:
            merged.append(unit)
    return [make_chunk(parent_id, "recursive", i, u) for i, u in enumerate(merged)]


def semantic_chunks(parent_id: str, text: str, adjacent_similarity: Sequence[float] | None,
                    threshold: float, max_tokens: int = SEMANTIC_MAX,
                    min_tokens: int = SEMANTIC_MIN) -> list[dict]:
    """Break between sentences i and i+1 when their similarity < threshold or size overflows.

    ``adjacent_similarity[i]`` is the cosine between sentence i and i+1 (len = n_sentences-1).
    """
    sents = sentences(text)
    if not sents:
        return []
    groups, current = [], [sents[0]]
    for i, s in enumerate(sents[1:]):
        semantic_break = adjacent_similarity is not None and float(adjacent_similarity[i]) < threshold
        size_break = n_tokens(" ".join(current + [s])) > max_tokens
        if semantic_break or size_break:
            groups.append(" ".join(current))
            current = [s]
        else:
            current.append(s)
    groups.append(" ".join(current))

    merged: list[str] = []
    for g in groups:
        if merged and n_tokens(g) < min_tokens and n_tokens(merged[-1] + " " + g) <= max_tokens:
            merged[-1] = merged[-1] + " " + g
        else:
            merged.append(g)
    return [make_chunk(parent_id, "semantic", i, g) for i, g in enumerate(merged)]


def biomedical_chunks(parent_id: str, text: str, max_tokens: int = BIOMEDICAL_MAX) -> list[dict]:
    sents = sentences(text)
    if not sents:
        return []
    groups, current, current_len = [], [], 0
    for s in sents:
        s_len = n_tokens(s)
        relationish = bool(PATTERNS["relation_trigger"].search(s) or PATTERNS["negation"].search(s))
        if current and current_len + s_len > max_tokens:
            if relationish and current_len < int(max_tokens * BIOMEDICAL_STRETCH):
                current.append(s)                # keep the relation with its context
                groups.append(" ".join(current))
                current, current_len = [], 0
            else:
                groups.append(" ".join(current))
                current, current_len = [s], s_len
        else:
            current.append(s)
            current_len += s_len
    if current:
        groups.append(" ".join(current))
    return [make_chunk(parent_id, "biomedical", i, g) for i, g in enumerate(groups)]


_CLAUSE_SPLIT = re.compile(r"\s+(?:and|but|while|whereas|although|because|therefore|however)\s+", re.I)


def proposition_chunks(parent_id: str, text: str, max_tokens: int = PROPOSITION_MAX) -> list[dict]:
    props: list[str] = []
    for s in sentences(text):
        parts = [p.strip(" ,;:") for p in _CLAUSE_SPLIT.split(s) if p.strip(" ,;:")] or [s]
        for p in parts:
            toks = tokens(p)
            if len(toks) <= max_tokens:
                props.append(p)
            else:
                props += [" ".join(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]
    return [make_chunk(parent_id, "proposition", i, p) for i, p in enumerate(props)]


def parent_child_chunks(parent_id: str, text: str, child_size: int = CHILD_SIZE,
                        child_overlap: int = CHILD_OVERLAP) -> list[dict]:
    toks = tokens(text)
    return [make_chunk(parent_id, "parent_child", i, " ".join(toks[a:b]), a, b)
            for i, (a, b) in enumerate(windows(len(toks), child_size, child_overlap))]


def late_spans(parent_id: str, text: str, size: int = LATE_SIZE,
               overlap: int = LATE_OVERLAP) -> list[dict]:
    toks = tokens(text)
    return [make_chunk(parent_id, "late", i, " ".join(toks[a:b]), a, b)
            for i, (a, b) in enumerate(windows(len(toks), size, overlap))]


# --------------------------------------------------------------------------- #
# Adaptive chunking (NB05): profile each passage, route it to one strategy.
# --------------------------------------------------------------------------- #
_BIO_ENTITY = re.compile(r"\b(?:BRCA\d*|EGFR|KRAS|BRAF|p53|COX[- ]?2|aspirin|olaparib|"
                         r"breast cancer|ovarian cancer)\b", re.I)
_BIO_RELATION = re.compile(r"\b(?:inhibits?|activates?|associated with|increases?|decreases?|"
                           r"causes?|reduces?|induces?|interacts with)\b", re.I)
_BIO_NEGATION = re.compile(r"\b(?:no|not|never|without|absence of|did not|does not)\b", re.I)
_WORDS = re.compile(r"[A-Za-z0-9-]+")


def document_profile(text: str) -> dict:
    s, t = sentences(text), tokens(text)
    if not s:
        return {"tokens": 0, "sentences": 0, "entity_density": 0.0, "relation_density": 0.0,
                "negation_density": 0.0, "lexical_shift": 0.0}
    shifts = []
    for a, b in zip(s[:-1], s[1:]):
        A, B = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
        shifts.append(1 - len(A & B) / max(1, len(A | B)))
    return {
        "tokens": len(t),
        "sentences": len(s),
        "entity_density": len(_BIO_ENTITY.findall(text)) / max(1, len(t)),
        "relation_density": len(_BIO_RELATION.findall(text)) / max(1, len(t)),
        "negation_density": len(_BIO_NEGATION.findall(text)) / max(1, len(t)),
        "lexical_shift": float(np.mean(shifts)) if shifts else 0.0,
    }


def route(profile: dict) -> tuple[str, str]:
    """NB05 route_rule_based: (strategy, reason)."""
    if profile["tokens"] <= ROUTE_SHORT_TOKENS:
        return "recursive", "short passage"
    if profile["relation_density"] >= ROUTE_RELATION_DENSITY and profile["negation_density"] > 0:
        return "biomedical", "relation + negation density"
    if profile["entity_density"] >= ROUTE_ENTITY_DENSITY:
        return "biomedical", "high biomedical entity density"
    if profile["lexical_shift"] >= ROUTE_TOPIC_SHIFT:
        return "semantic", "high topic shift"
    if profile["tokens"] >= ROUTE_LONG_TOKENS:
        return "parent_child", "long passage"
    return "recursive", "default structure-preserving route"


def adaptive_chunks(parent_id: str, text: str,
                    semantic_fn: Callable[[str, str], list[dict]]) -> list[dict]:
    """Route the passage, chunk it with the chosen strategy, tag chunks as adaptive."""
    choice, _reason = route(document_profile(text))
    if choice == "semantic":
        chunks = semantic_fn(parent_id, text)
    else:
        chunks = STRATEGIES[choice](parent_id, text)
    return [make_chunk(parent_id, "adaptive", i, c["text"], c["start_token"], c["end_token"])
            | {"routed_to": choice} for i, c in enumerate(chunks)]


STRATEGIES: dict[str, Callable[[str, str], list[dict]]] = {
    "fixed": fixed_chunks,
    "recursive": recursive_chunks,
    "biomedical": biomedical_chunks,
    "proposition": proposition_chunks,
    "parent_child": parent_child_chunks,
    "late": late_spans,
}
