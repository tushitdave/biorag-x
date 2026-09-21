"""Central path resolution for BioRAG-X.

Reuses the artifacts already produced by the notebooks (canonical data, chunks,
embeddings, chunking manifests) *in place* rather than copying gigabytes of
parquet. Everything is resolved relative to the repo root so the package works
regardless of the current working directory.
"""
from __future__ import annotations

from pathlib import Path

# adaptive-biomed-rag/common/paths.py -> repo root is two parents up.
PKG_ROOT = Path(__file__).resolve().parents[1]          # .../adaptive-biomed-rag
PROJECT_ROOT = PKG_ROOT.parent                          # .../BioRAG-X

# Existing notebook artifacts (source of truth for data).
NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"
CANONICAL_DIR = NOTEBOOKS_DIR / "data" / "canonical"
CHUNKS_DIR = NOTEBOOKS_DIR / "data" / "chunks"
EMBEDDINGS_DIR = NOTEBOOKS_DIR / "data" / "embeddings"
NB_ARTIFACTS_DIR = NOTEBOOKS_DIR / "artifacts"

# Canonical parquet files.
PASSAGES_PARQUET = CANONICAL_DIR / "passages.parquet"
QUESTIONS_PARQUET = CANONICAL_DIR / "questions.parquet"
GOLD_PARQUET = CANONICAL_DIR / "gold_relationships.parquet"

# Reusable chunk tables (advanced representations from NB04).
SEMANTIC_CHUNKS = CHUNKS_DIR / "semantic_advanced.parquet"
BIOMEDICAL_CHUNKS = CHUNKS_DIR / "biomedical_advanced.parquet"
PROPOSITION_CHUNKS = CHUNKS_DIR / "proposition_advanced.parquet"
PARENT_CHILD_CHUNKS = CHUNKS_DIR / "parent_child_advanced.parquet"
LATE_CHUNKS = CHUNKS_DIR / "late_spans_advanced.parquet"

# Reusable semantic chunk embeddings (ada-002, from NB04 12b).
SEMANTIC_EMBEDDINGS = EMBEDDINGS_DIR / "semantic_chunk_embeddings.parquet"

# Package-local working dirs.
DATA_DIR = PKG_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
METADATA_DIR = DATA_DIR / "metadata"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "processed" / "indexes"
CONFIGS_DIR = PKG_ROOT / "configs"

# Full-corpus MedCPT passage vectors + FAISS index (scripts/build_medcpt_index.py).
DENSE_INDEX_DIR = INDEX_DIR / "dense"
MEDCPT_FAISS = DENSE_INDEX_DIR / "medcpt_flatip.faiss"
MEDCPT_IDS = DENSE_INDEX_DIR / "passage_ids_medcpt.parquet"
MEDCPT_EMBEDDINGS = DENSE_INDEX_DIR / "passage_embeddings_medcpt.npy"

# Chunk-strategy indexes (scripts/build_chunk_indexes.py): one folder per strategy
# with chunks.parquet + embeddings.npy + manifest.json, plus a shared build status.
CHUNK_INDEX_DIR = INDEX_DIR / "chunks"
CHUNK_BUILD_STATUS = CHUNK_INDEX_DIR / "build_status.json"
GRAPH_DIR = INDEX_DIR / "graph"              # evidence graph (scripts/build_graph_pageindex.py)
PAGEINDEX_DIR = INDEX_DIR / "pageindex"      # PageIndex topic tree (same script)
SYNONYM_DIR = INDEX_DIR / "synonyms"         # corpus abbreviation dictionary (same script)

# Frozen benchmark (scripts/build_benchmark.py) and NB01 per-question strata.
DEV_QUESTIONS = BENCHMARK_DIR / "dev_300.parquet"
LOCKED_QUESTIONS = BENCHMARK_DIR / "locked_100.parquet"
DEV_CORPUS_IDS = BENCHMARK_DIR / "dev_corpus_ids.parquet"
NB01_QA_FORENSICS = NB_ARTIFACTS_DIR / "01_dataset_forensics" / "qa_forensics.parquet"

# Experiment run store (SQLite summaries + per-question parquet).
RUNS_DIR = DATA_DIR / "runs"
RUNS_DB = RUNS_DIR / "runs.db"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

for _d in (PROCESSED_DIR, BENCHMARK_DIR, METADATA_DIR, CACHE_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The project .env (contains Azure keys). Loaded by common.llm.
ENV_PATH = PROJECT_ROOT / ".env"


def exists_report() -> dict:
    """Quick availability check for the reusable artifacts (used by /health)."""
    items = {
        "passages": PASSAGES_PARQUET,
        "questions": QUESTIONS_PARQUET,
        "gold": GOLD_PARQUET,
        "semantic_chunks": SEMANTIC_CHUNKS,
        "semantic_embeddings": SEMANTIC_EMBEDDINGS,
        "medcpt_index": MEDCPT_FAISS,
    }
    return {k: v.exists() for k, v in items.items()}
