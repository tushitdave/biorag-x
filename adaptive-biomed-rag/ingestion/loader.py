"""Ingestion / data access for BioRAG-X.

Thin layer over the canonical parquet already produced by the notebooks. Loads
are cached in-process (lru_cache) so repeated API calls are cheap. No mutation of
the source data — this is read-only access to the canonical corpus, questions,
gold relationships, and the reusable chunk tables.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from common import paths


@lru_cache(maxsize=1)
def load_passages() -> pd.DataFrame:
    df = pd.read_parquet(paths.PASSAGES_PARQUET)
    df["retrieval_text"] = df["retrieval_text"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def load_questions() -> pd.DataFrame:
    return pd.read_parquet(paths.QUESTIONS_PARQUET)


@lru_cache(maxsize=1)
def load_gold() -> pd.DataFrame:
    return pd.read_parquet(paths.GOLD_PARQUET)


@lru_cache(maxsize=24)
def load_chunks(strategy: str = "passage", corpus: str = "full") -> pd.DataFrame:
    """Load a retrieval-unit table by strategy name.

    ``passage`` is passage-as-is: every usable (non-empty) canonical passage is one
    unit (chunk_id == parent_passage_id). It covers the full corpus and matches the
    MedCPT passage index row-for-row by id. Built strategies (fixed, recursive,
    semantic, ...) come from scripts/build_chunk_indexes.py; ``semantic_nb04`` is
    NB04's ada-002 semantic chunk table (3,483 passages).

    ``corpus="dev"`` keeps only units whose parent passage is in the frozen dev corpus.
    """
    if corpus == "dev":
        df = load_chunks(strategy, "full")
        return df[df["parent_passage_id"].isin(load_dev_corpus_ids())].reset_index(drop=True)

    if strategy == "passage":
        p = load_passages()
        p = p[p["is_usable"] & p["retrieval_text"].str.strip().ne("")]
        ids = p["canonical_passage_id"].astype(str).to_numpy()
        return pd.DataFrame({"chunk_id": ids, "parent_passage_id": ids,
                             "text": p["retrieval_text"].to_numpy()})

    path = (paths.SEMANTIC_CHUNKS if strategy == "semantic_nb04"
            else paths.CHUNK_INDEX_DIR / strategy / "chunks.parquet")
    if not path.exists():
        raise FileNotFoundError(f"No chunk table for strategy '{strategy}' (not built yet?)")
    df = pd.read_parquet(path)
    df["text"] = df["text"].fillna("").astype(str)
    return df


@lru_cache(maxsize=1)
def load_dev_corpus_ids() -> frozenset[str]:
    """Passage ids of the frozen dev corpus (scripts/build_benchmark.py)."""
    return frozenset(pd.read_parquet(paths.DEV_CORPUS_IDS)["canonical_passage_id"].astype(str))


QUESTION_SETS = {"dev_300": paths.DEV_QUESTIONS, "locked_100": paths.LOCKED_QUESTIONS}


@lru_cache(maxsize=2)
def load_question_set(name: str) -> pd.DataFrame:
    """A frozen benchmark question set: dev_300 (tuning) or locked_100 (final check)."""
    return pd.read_parquet(QUESTION_SETS[name])


@lru_cache(maxsize=1)
def load_semantic_embeddings() -> pd.DataFrame:
    """Precomputed ada-002 chunk embeddings (chunk_id -> embedding vector)."""
    return pd.read_parquet(paths.SEMANTIC_EMBEDDINGS)


def corpus_counts() -> dict:
    return {
        "passages": int(len(load_passages())),
        "questions": int(len(load_questions())),
        "gold_relationships": int(len(load_gold())),
    }


def sample_questions(n: int, seed: int = 42, only_usable_gold: bool = True) -> pd.DataFrame:
    """A reproducible question sample for benchmarking (usable gold by default)."""
    q = load_questions()
    if only_usable_gold and "has_usable_gold_evidence" in q.columns:
        q = q[q["has_usable_gold_evidence"] == True]  # noqa: E712
    n = min(n, len(q))
    return q.sample(n=n, random_state=seed).reset_index(drop=True)


def gold_ids_for_question(row) -> list[str]:
    """Return usable gold canonical passage ids for a question row."""
    for col in ("usable_gold_canonical_ids", "gold_canonical_passage_ids"):
        if col in row and row[col] is not None:
            try:
                return [str(x) for x in list(row[col])]
            except TypeError:
                pass
    return []
