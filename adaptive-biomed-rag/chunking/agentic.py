"""Agentic (LLM-guided) chunking at question time - NB05's llm_guided_chunks, batched.

For the passages retrieved for a question, GPT-4o proposes chunk boundaries (sentence
indexes) for every passage in ONE call. Each plan is validated (NB05
validate_llm_plan) and applied; a missing or invalid plan falls back to the recursive
chunker. Plans are cached per passage (data/cache/agentic/), so a passage is segmented
by the LLM at most once, whatever the question. All LLM traffic goes through
common.llm, so the master switch, budget and prompt cache apply.
"""
from __future__ import annotations

import json

from chunking.base import make_chunk, sentences
from chunking.strategies import recursive_chunks
from common import paths
from common.llm import chat_json

CACHE_DIR = paths.CACHE_DIR / "agentic"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_TOKENS = 256

SYSTEM = "You are a precise JSON-only API. Output only valid JSON."


def build_prompt(items: list[tuple[str, list[str]]], max_tokens: int = MAX_TOKENS) -> str:
    blocks = "\n\n".join(
        f"PASSAGE {pid} ({len(sents)} sentences):\n"
        + "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))
        for pid, sents in items
    )
    return f"""You are the BioRAG-X biomedical document segmentation engine.
For EACH passage below, choose where new chunks start. A boundary b means a new
chunk starts at sentence [b]; valid values are 1 .. (number of sentences - 1).

Rules:
1. Preserve the original wording (you only choose boundaries).
2. Do not split biomedical entities or relation statements.
3. Keep negation with the statement it negates.
4. Prefer chunks of at most {max_tokens} words.
5. A short or single-topic passage can stay one chunk (empty boundaries).

Return JSON only:
{{"passages": [{{"id": "<passage id>", "boundaries": [<int>, ...], "reason": "<short>"}}]}}

{blocks}"""


def validate_plan(boundaries, n_sentences: int) -> tuple[bool, list[str]]:
    """NB05 validate_llm_plan: sorted unique ints within 1..n-1."""
    errors = []
    if not isinstance(boundaries, list):
        return False, ["boundaries must be a list"]
    if any(not isinstance(x, int) for x in boundaries):
        errors.append("boundaries must contain integers")
    elif any(x < 1 or x >= n_sentences for x in boundaries):
        errors.append("boundary outside valid range")
    elif boundaries != sorted(set(boundaries)):
        errors.append("boundaries must be sorted and unique")
    return not errors, errors


def _apply(pid: str, sents: list[str], boundaries: list[int]) -> list[str]:
    cuts = [0] + boundaries + [len(sents)]
    return [" ".join(sents[a:b]) for a, b in zip(cuts[:-1], cuts[1:]) if sents[a:b]]


def _cache_file(pid: str):
    return CACHE_DIR / f"{pid}.json"


def agentic_chunks(passages: list[tuple[str, str]]) -> tuple[list[dict], dict]:
    """Chunk (passage_id, text) pairs; returns (chunks, meta about LLM use)."""
    meta = {"passages": len(passages), "llm_planned": 0, "cached": 0, "fallback": 0,
            "single": 0, "llm_calls": 0, "llm_source": "none"}
    plans: dict[str, dict] = {}
    todo: list[tuple[str, list[str]]] = []
    sents_by: dict[str, list[str]] = {}

    for pid, text in passages:
        sents = sentences(text)
        sents_by[pid] = sents
        if len(sents) < 2:
            plans[pid] = {"boundaries": [], "source": "single", "reason": "one sentence"}
        elif _cache_file(pid).exists():
            plans[pid] = json.loads(_cache_file(pid).read_text()) | {"source": "cache"}
        else:
            todo.append((pid, sents))

    if todo:
        data, _raw, source = chat_json(build_prompt(todo), system=SYSTEM,
                                       max_tokens=80 * len(todo) + 100)
        meta["llm_source"] = source
        meta["llm_calls"] = int(source == "live")
        proposed = {}
        if isinstance(data, dict):
            for p in data.get("passages", []) or []:
                if isinstance(p, dict) and "id" in p:
                    proposed[str(p["id"])] = p
        for pid, sents in todo:
            p = proposed.get(pid)
            ok, errors = validate_plan(p.get("boundaries") if p else None, len(sents))
            if data is None:
                plans[pid] = {"boundaries": None, "source": "fallback",
                              "reason": f"LLM unavailable ({source})"}
                continue                                   # not cached: may succeed later
            plan = ({"boundaries": p["boundaries"], "source": "llm",
                     "reason": str(p.get("reason", ""))[:200]} if ok else
                    {"boundaries": None, "source": "fallback",
                     "reason": "invalid plan: " + "; ".join(errors or ["missing"])})
            _cache_file(pid).write_text(json.dumps(plan))
            plans[pid] = plan

    chunks: list[dict] = []
    for pid, text in passages:
        plan = plans[pid]
        src = plan["source"]
        if src == "cache":
            src = "llm" if plan.get("boundaries") is not None else "fallback"
            meta["cached"] += 1
        if plan.get("boundaries") is not None:
            pieces = _apply(pid, sents_by[pid], plan["boundaries"])
        else:
            pieces = [c["text"] for c in recursive_chunks(pid, text)]
        meta["llm_planned"] += src == "llm" and plan["source"] != "cache"
        meta["fallback"] += src == "fallback"
        meta["single"] += src == "single"
        for i, piece in enumerate(pieces):
            chunks.append(make_chunk(pid, "agentic", i, piece)
                          | {"agentic_source": src, "agentic_reason": plan.get("reason", "")})
    return chunks, meta
