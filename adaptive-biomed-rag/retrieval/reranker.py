"""Rerankers for BioRAG-X.

``medcpt_ce`` is the real MedCPT cross-encoder (reads question and passage
together; local, no API). ``lexical_overlap`` is a zero-dependency word-overlap
reranker. The ``cross_encoder`` and ``colbert`` choices are dependency-free local
proxies: the former scores unigram/bigram interaction and the latter applies
token-level MaxSim. LLM listwise uses the centrally guarded chat client.
"""
from __future__ import annotations

import re
import threading
from difflib import SequenceMatcher
from functools import lru_cache

from common.llm import chat_json

MEDCPT_CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
_ce_lock = threading.Lock()   # one model instance; serialize calls from request + run threads

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def _toks(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "")}


def lexical_overlap_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    q = _toks(query)
    for c in candidates:
        ct = _toks(c["text"])
        inter = len(q & ct)
        # Jaccard-ish overlap, length-normalized so long chunks don't dominate.
        c["rerank_score"] = inter / (len(q) + 1e-9)
    ranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)
    return ranked[:top_k]


def cross_encoder_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Dependency-free interaction proxy using unigram and ordered bigram overlap."""
    q_tokens = [t.lower() for t in _TOKEN.findall(query or "")]
    q = set(q_tokens)
    q_bigrams = set(zip(q_tokens, q_tokens[1:]))
    for c in candidates:
        d_tokens = [t.lower() for t in _TOKEN.findall(c.get("text", ""))]
        d = set(d_tokens)
        d_bigrams = set(zip(d_tokens, d_tokens[1:]))
        unigram = len(q & d) / max(1, len(q))
        bigram = len(q_bigrams & d_bigrams) / max(1, len(q_bigrams))
        phrase = 1.0 if query.lower() in c.get("text", "").lower() else 0.0
        c["rerank_score"] = 0.55 * unigram + 0.35 * bigram + 0.10 * phrase
    return sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:top_k]


def colbert_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """ColBERT-style lexical MaxSim proxy over query and passage tokens."""
    q_tokens = list(_toks(query))
    for c in candidates:
        d_tokens = list(_toks(c.get("text", "")))
        maxima = [
            max((SequenceMatcher(None, qt, dt).ratio() for dt in d_tokens), default=0.0)
            for qt in q_tokens
        ]
        c["rerank_score"] = sum(maxima) / max(1, len(maxima))
    return sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def _medcpt_cross_encoder():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MEDCPT_CROSS_ENCODER)
    model = AutoModelForSequenceClassification.from_pretrained(MEDCPT_CROSS_ENCODER)
    return tok, model.to(device).eval(), device


def medcpt_cross_encoder_rerank(query: str, candidates: list[dict], top_k: int = 10,
                                batch_size: int = 16) -> list[dict]:
    """Score each (question, passage) pair with the MedCPT cross-encoder (logit)."""
    import torch
    tok, model, device = _medcpt_cross_encoder()
    pairs = [[query, c.get("text", "")] for c in candidates]
    scores: list[float] = []
    with _ce_lock, torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            enc = tok(pairs[i:i + batch_size], truncation=True, padding=True,
                      max_length=512, return_tensors="pt").to(device)
            scores += model(**enc).logits.squeeze(dim=1).float().cpu().tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:top_k]


def llm_listwise_rerank(query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Ask the guarded LLM for a candidate ordering; fall back locally if unavailable."""
    pool = candidates[:12]  # bound prompt cost; downstream only needs the top results
    rows = [f"{i}: {c.get('text', '')[:500]}" for i, c in enumerate(pool)]
    prompt = (
        "Rank the candidate IDs by relevance to the biomedical query. "
        "Return JSON only as {\"ranking\":[0,1,...]}.\n"
        f"Query: {query}\nCandidates:\n" + "\n".join(rows)
    )
    data, _raw, _source = chat_json(prompt, max_tokens=250)
    order = data.get("ranking", []) if isinstance(data, dict) else []
    valid = [i for i in order if isinstance(i, int) and 0 <= i < len(pool)]
    if not valid:
        return cross_encoder_rerank(query, candidates, top_k)
    valid.extend(i for i in range(len(pool)) if i not in valid)
    ranked = [pool[i] for i in valid] + candidates[len(pool):]
    for rank, c in enumerate(ranked):
        c["rerank_score"] = 1.0 - rank / max(1, len(ranked))
    return ranked[:top_k]


def rerank(query: str, candidates: list[dict], method: str = "lexical_overlap",
           top_k: int = 10) -> list[dict]:
    if method in ("none", None):
        return candidates[:top_k]
    if method == "lexical_overlap":
        return lexical_overlap_rerank(query, candidates, top_k)
    if method == "cross_encoder":
        return cross_encoder_rerank(query, candidates, top_k)
    if method == "colbert":
        return colbert_rerank(query, candidates, top_k)
    if method == "llm_reranker":
        return llm_listwise_rerank(query, candidates, top_k)
    if method == "medcpt_ce":
        return medcpt_cross_encoder_rerank(query, candidates, top_k)
    return lexical_overlap_rerank(query, candidates, top_k)
