"""Build chunk tables + MedCPT vectors for every chunking strategy over the full corpus.

Each strategy is written to data/processed/indexes/chunks/<strategy>/
  chunks.parquet   chunk_id, parent_passage_id, strategy, chunk_index, text, n_tokens, ...
  embeddings.npy   float32 [n_chunks, 768], row i <-> chunks row i (MedCPT Article Encoder)
  manifest.json    counts, parameters, timing
and becomes selectable in the app as soon as its manifest exists. Live progress is
written to data/processed/indexes/chunks/build_status.json (read by /capabilities).

semantic  breaks where adjacent sentences are dissimilar. NB04 used ada-002 cosine with
          a 0.72 threshold; here sentence vectors are local MedCPT, whose cosine scale
          differs, so the threshold is calibrated: the 20th percentile of all adjacent-
          sentence similarities (the weakest 20% of transitions become boundaries).
adaptive  NB05 router per passage; its "semantic" route uses the semantic chunker above.
late      one contextual pass per passage, span vectors mean-pooled (late chunking).

Resumable: finished strategies are skipped; encoding checkpoints to shards.

Usage (from adaptive-biomed-rag/):
  python scripts/build_chunk_indexes.py                       # all strategies, full corpus
  python scripts/build_chunk_indexes.py --strategies fixed recursive
  python scripts/build_chunk_indexes.py --limit 300 --out-dir /tmp/chunk_smoke
  python scripts/build_chunk_indexes.py --cool-down 1.0     # half duty: cooler, ~2x slower
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chunking.base import sentences, tokens  # noqa: E402
from chunking.strategies import (  # noqa: E402
    SEMANTIC_PERCENTILE, STRATEGIES, adaptive_chunks, semantic_chunks,
)
from common import paths  # noqa: E402
from indexes.encoder import ArticleEncoder  # noqa: E402

ORDER = ["fixed", "recursive", "biomedical", "parent_child", "semantic", "adaptive",
         "late", "proposition"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class Status:
    """build_status.json: {strategy: {status, stage, done, total, chunks, error, ...}}."""

    def __init__(self, path: Path, strategies: list[str]):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}
        for s in strategies:
            if self.data.get(s, {}).get("status") != "ready":
                self.data[s] = {"status": "queued"}
        self._write()

    def update(self, strategy: str, **fields) -> None:
        self.data.setdefault(strategy, {}).update(fields)
        self._write()

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


def load_passages(limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(paths.PASSAGES_PARQUET,
                         columns=["canonical_passage_id", "retrieval_text", "is_usable"])
    df = df[df["is_usable"] & df["retrieval_text"].fillna("").str.strip().ne("")]
    df = df.reset_index(drop=True)
    return df.head(limit) if limit else df


def sentence_similarities(enc: ArticleEncoder, df: pd.DataFrame, cache: Path) -> list[np.ndarray]:
    """Cosine between each pair of adjacent sentences, per passage (cached)."""
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return list(data["sims"])
    sents = [sentences(t) for t in df["retrieval_text"]]
    flat = [s for ss in sents for s in ss]
    log(f"semantic: encoding {len(flat):,} sentences for boundary detection")
    vecs = enc.encode(flat, batch_size=64, log=log)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    sims, k = [], 0
    for ss in sents:
        v = vecs[k:k + len(ss)]
        sims.append(np.sum(v[:-1] * v[1:], axis=1) if len(ss) > 1 else np.array([], np.float32))
        k += len(ss)
    ragged = np.empty(len(sims), dtype=object)     # one array per passage, lengths differ
    ragged[:] = sims
    np.savez(cache, sims=ragged)
    return sims


def chunk_all(strategy: str, df: pd.DataFrame, semantic_fn) -> pd.DataFrame:
    rows = []
    for pid, text in zip(df["canonical_passage_id"], df["retrieval_text"]):
        if strategy == "semantic":
            rows += semantic_fn(pid, text)
        elif strategy == "adaptive":
            rows += adaptive_chunks(pid, text, semantic_fn)
        else:
            rows += STRATEGIES[strategy](pid, text)
    return pd.DataFrame(rows)


def build(strategy: str, df: pd.DataFrame, enc: ArticleEncoder, out_dir: Path,
          status: Status, semantic_fn, semantic_meta: dict) -> None:
    sdir = out_dir / strategy
    sdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    status.update(strategy, status="building", stage="chunking", started_at=_now(), error="")
    chunks = chunk_all(strategy, df, semantic_fn)
    chunks.to_parquet(sdir / "chunks.parquet", index=False)
    log(f"{strategy}: {len(chunks):,} chunks from {len(df):,} passages "
        f"({len(chunks) / len(df):.2f} per passage)")

    def progress(done: int, total: int) -> None:
        status.update(strategy, stage="embedding", done=done, total=total)

    if strategy == "late":
        text_by_pid = dict(zip(df["canonical_passage_id"], df["retrieval_text"]))
        groups = chunks.groupby("parent_passage_id", sort=False)
        pids = list(groups.groups)
        words = [tokens(text_by_pid[p]) for p in pids]
        spans = [list(zip(g["start_token"], g["end_token"])) for _, g in groups]
        per_passage = enc.encode_late(words, spans, log=log, progress=progress)
        order = np.concatenate([groups.indices[p] for p in pids])
        vecs = np.empty((len(chunks), 768), dtype=np.float32)
        vecs[order] = np.vstack(per_passage)
    else:
        shard_dir = sdir / "_shards"
        shard_dir.mkdir(exist_ok=True)
        vecs = enc.encode(chunks["text"].tolist(), shard_dir=shard_dir, log=log, progress=progress)
        shutil.rmtree(shard_dir)

    np.save(sdir / "embeddings.npy", vecs)
    manifest = {
        "strategy": strategy,
        "created_at": _now(),
        "passages": int(len(df)),
        "chunks": int(len(chunks)),
        "chunks_per_passage": round(len(chunks) / len(df), 3),
        "median_tokens": float(chunks["n_tokens"].median()),
        "encoder": "ncbi/MedCPT-Article-Encoder",
        "model_revision": enc.revision,
        "pooling": "late: mean of contextual token vectors per span" if strategy == "late" else "cls",
        "metric": "inner_product",
        "device": enc.device,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    if strategy in ("semantic", "adaptive"):
        manifest["semantic"] = semantic_meta
    if strategy == "adaptive":
        manifest["routed_to"] = (chunks.drop_duplicates("parent_passage_id")["routed_to"]
                                 .value_counts().to_dict())
    (sdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    status.update(strategy, status="ready", stage="done", chunks=int(len(chunks)),
                  finished_at=_now(), seconds=manifest["seconds"])
    log(f"{strategy}: ready in {manifest['seconds'] / 60:.1f} min")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategies", nargs="*", default=ORDER, choices=ORDER)
    ap.add_argument("--limit", type=int, default=None, help="first N passages (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=paths.CHUNK_INDEX_DIR)
    ap.add_argument("--force", action="store_true", help="rebuild strategies that are ready")
    ap.add_argument("--cool-down", type=float, default=0.0,
                    help="seconds of rest per second of GPU work (1.0 = ~half duty, cooler)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in ORDER if s in args.strategies
            and (args.force or not (args.out_dir / s / "manifest.json").exists())]
    status = Status(args.out_dir / "build_status.json", todo)
    if not todo:
        log("all requested strategies are ready")
        return

    df = load_passages(args.limit)
    enc = ArticleEncoder(cool_down=args.cool_down)
    log(f"{len(df):,} passages; building {todo} on {enc.device}"
        + (f" (cool-down {args.cool_down})" if args.cool_down else ""))

    semantic_meta: dict = {}
    semantic_fn = None
    if {"semantic", "adaptive"} & set(todo):
        sims = sentence_similarities(enc, df, args.out_dir / "_sentence_similarities.npz")
        pooled = np.concatenate([s for s in sims if len(s)])
        threshold = float(np.percentile(pooled, SEMANTIC_PERCENTILE))
        semantic_meta = {"threshold": round(threshold, 4), "percentile": SEMANTIC_PERCENTILE,
                         "sentence_encoder": "ncbi/MedCPT-Article-Encoder (cls, cosine)"}
        sims_by_pid = dict(zip(df["canonical_passage_id"], sims))
        semantic_fn = lambda pid, text: semantic_chunks(pid, text, sims_by_pid[pid], threshold)  # noqa: E731
        log(f"semantic threshold = {threshold:.4f} (p{SEMANTIC_PERCENTILE} of {len(pooled):,} transitions)")

    for s in todo:
        try:
            build(s, df, enc, args.out_dir, status, semantic_fn, semantic_meta)
        except Exception as e:  # keep going; the failure is visible in the status file
            status.update(s, status="failed", error=f"{type(e).__name__}: {e}")
            log(f"{s}: FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
