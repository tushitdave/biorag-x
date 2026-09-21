"""Label-free evidence judge for the per-question comparison.

A question typed by a user has no gold passages, so retrieval combinations are
scored by how relevant their evidence is to the question, as judged by the MedCPT
cross-encoder (trained on PubMed search logs; local, no API call). Its logit is
mapped to 0..1 with a sigmoid. Scores are cached per (question, text).
"""
from __future__ import annotations

import hashlib
import math

from retrieval.reranker import _ce_lock, _medcpt_cross_encoder

JUDGE_MODEL = "ncbi/MedCPT-Cross-Encoder"
JUDGE_RERANKER = "medcpt_ce"          # the reranker id that uses the same model
_cache: dict[tuple[str, str], float] = {}
_MAX_CACHE = 50_000


def _key(question: str, text: str) -> tuple[str, str]:
    return question, hashlib.sha1(text.encode("utf-8")).hexdigest()


def relevance(question: str, texts: list[str], batch_size: int = 16) -> list[float]:
    """0..1 relevance of each text to the question (sigmoid of the cross-encoder logit)."""
    import torch
    todo = list(dict.fromkeys(t for t in texts if _key(question, t) not in _cache))
    if todo:
        if len(_cache) > _MAX_CACHE:
            _cache.clear()
        tok, model, device = _medcpt_cross_encoder()
        with _ce_lock, torch.inference_mode():
            for i in range(0, len(todo), batch_size):
                batch = todo[i:i + batch_size]
                enc = tok([[question, t] for t in batch], truncation=True, padding=True,
                          max_length=512, return_tensors="pt").to(device)
                logits = model(**enc).logits.squeeze(dim=1).float().cpu().tolist()
                for t, z in zip(batch, logits):
                    _cache[_key(question, t)] = 1 / (1 + math.exp(-z))
    return [_cache[_key(question, t)] for t in texts]
