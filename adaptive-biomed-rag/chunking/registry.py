"""Every chunking strategy the app offers, and whether its index is ready.

Built strategies live in data/processed/indexes/chunks/<id>/ (scripts/build_chunk_indexes.py);
their live build progress is read from build_status.json. ``passage`` uses the
full-corpus passage index, ``agentic`` runs at question time, and ``semantic_nb04`` is
NB04's ada-002 semantic chunks (3,483 passages only).
"""
from __future__ import annotations

import json

from common import paths

# id -> (label, what it does, where it comes from)
STRATEGIES: dict[str, tuple[str, str, str]] = {
    "passage": ("Whole passage",
                "Each passage is one unit; passages are short (~160 words).", "NB02"),
    "fixed": ("Fixed-size",
              "256-word windows with 32-word overlap; ignores sentence boundaries.", "NB03"),
    "recursive": ("Recursive",
                  "Paragraph, then sentence, then word splits up to 256 words.", "NB03"),
    "semantic": ("Semantic",
                 "Starts a new chunk where adjacent sentences stop being similar (MedCPT).",
                 "NB04"),
    "biomedical": ("Biomedical-aware",
                   "Keeps relation and negation sentences together with their context.", "NB04"),
    "proposition": ("Proposition",
                    "One clause or fact per unit (~12 per passage).", "NB04"),
    "parent_child": ("Parent-child",
                     "Searches 96-word children, sends the whole parent passage to the LLM.",
                     "NB04"),
    "late": ("Late chunking",
             "256-word spans whose vectors come from one pass over the whole passage.", "NB04"),
    "adaptive": ("Adaptive (router)",
                 "Rules pick a chunker for each passage from its profile.", "NB05"),
    "agentic": ("Agentic (LLM)",
                "GPT-4o re-chunks the passages retrieved for your question (1 call).", "NB05"),
    "semantic_nb04": ("Semantic - NB04 ada-002",
                      "NB04's ada-002 semantic chunks; covers 3,483 passages only.", "NB04"),
}

BUILT = ["fixed", "recursive", "semantic", "biomedical", "proposition", "parent_child",
         "late", "adaptive"]


def _build_status() -> dict:
    try:
        return json.loads(paths.CHUNK_BUILD_STATUS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def status(strategy: str) -> dict:
    """{status: ready|building|paused|queued|failed|missing, progress: 0..1 | None, detail: str}."""
    if strategy == "passage":
        ok = paths.MEDCPT_FAISS.exists()
        return {"status": "ready" if ok else "missing", "progress": None, "detail": ""}
    if strategy == "agentic":
        ok = paths.MEDCPT_FAISS.exists()
        return {"status": "ready" if ok else "missing", "progress": None,
                "detail": "runs at question time"}
    if strategy == "semantic_nb04":
        ok = paths.SEMANTIC_CHUNKS.exists() and paths.SEMANTIC_EMBEDDINGS.exists()
        return {"status": "ready" if ok else "missing", "progress": None, "detail": ""}
    if (paths.CHUNK_INDEX_DIR / strategy / "manifest.json").exists():
        return {"status": "ready", "progress": None, "detail": ""}
    st = _build_status().get(strategy, {})
    state = st.get("status", "missing")
    progress = None
    if state in ("building", "paused") and st.get("total"):
        progress = round(st.get("done", 0) / st["total"], 3)
    return {"status": state, "progress": progress, "detail": st.get("error", "")}


def is_ready(strategy: str) -> bool:
    return status(strategy)["status"] == "ready"


def chunk_dir(strategy: str):
    return paths.CHUNK_INDEX_DIR / strategy
