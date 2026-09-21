"""Build MedCPT passage embeddings and an exact FAISS index over the full corpus.

Encodes every usable (non-empty) canonical passage with ncbi/MedCPT-Article-Encoder
using the official MedCPT recipe: [CLS] pooling, max_length=512, raw vectors scored
by inner product (MedCPT is trained with dot-product similarity, so no normalization).
Empty passages are excluded: they carry no evidence and would all embed to the same
vector.

Outputs in adaptive-biomed-rag/data/processed/indexes/dense/:
  passage_embeddings_medcpt.npy   float32 [n, 768]; row i <-> passage_ids row i
  passage_ids_medcpt.parquet      canonical_passage_id per row
  medcpt_flatip.faiss             faiss.IndexFlatIP over the same vectors (exact search)
  medcpt_manifest.json            model, recipe, counts, versions, timing

Resumable: vectors are written in shards keyed by a fingerprint of the inputs, so an
interrupted run continues where it stopped; shards are removed after success.

Usage (from adaptive-biomed-rag/):
  python scripts/build_medcpt_index.py
  python scripts/build_medcpt_index.py --limit 256 --out-dir /tmp/medcpt_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MAX_LENGTH = 512
DIM = 768
DEFAULT_OUT_DIR = paths.DENSE_INDEX_DIR


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def load_corpus(limit: int | None) -> tuple[pd.DataFrame, int]:
    cols = ["canonical_passage_id", "retrieval_text", "is_usable",
            "source_snapshot_id", "pipeline_version"]
    df = pd.read_parquet(paths.PASSAGES_PARQUET, columns=cols)
    usable = df["is_usable"] & df["retrieval_text"].fillna("").str.strip().ne("")
    excluded = int((~usable).sum())
    df = df[usable].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df, excluded


def fingerprint(ids: list[str], texts: list[str], shard_size: int) -> str:
    h = hashlib.sha256(f"{ARTICLE_MODEL}|{MAX_LENGTH}|{shard_size}".encode())
    for pid, text in zip(ids, texts):
        h.update(pid.encode())
        h.update(hashlib.sha256(text.encode()).digest())
    return h.hexdigest()[:16]


def embed_passages(texts: list[str], device: str, batch_size: int,
                   shard_size: int, shard_dir: Path) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # silence the per-text "longer than 512" warning
    tok = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    revision = getattr(model.config, "_commit_hash", None)

    # Sort by token length so each batch pads to a similar length (big speedup).
    lengths = np.array([len(x) for x in tok(texts, truncation=False)["input_ids"]])
    order = np.argsort(lengths, kind="stable")

    vecs = np.empty((len(texts), DIM), dtype=np.float32)
    n_shards = (len(order) + shard_size - 1) // shard_size
    t0, done_new = time.perf_counter(), 0
    for s in range(n_shards):
        idx = order[s * shard_size:(s + 1) * shard_size]
        shard_file = shard_dir / f"shard_{s:04d}.npy"
        if shard_file.exists():
            vecs[idx] = np.load(shard_file)
            log(f"shard {s + 1}/{n_shards} reused")
            continue
        parts = []
        for b in range(0, len(idx), batch_size):
            batch = [texts[i] for i in idx[b:b + batch_size]]
            enc = tok(batch, truncation=True, padding=True, max_length=MAX_LENGTH,
                      return_tensors="pt").to(device)
            with torch.inference_mode():
                cls = model(**enc).last_hidden_state[:, 0, :]
            parts.append(cls.float().cpu().numpy())
        shard = np.vstack(parts)
        np.save(shard_file, shard)
        vecs[idx] = shard
        done_new += len(idx)
        rate = done_new / (time.perf_counter() - t0)
        remaining = len(order) - (s + 1) * shard_size
        eta = max(0, remaining) / rate if rate else 0
        log(f"shard {s + 1}/{n_shards} done | {rate:.1f} passages/s | ETA {eta / 60:.1f} min")

    stats = {
        "model_revision": revision,
        "truncated_passages": int((lengths > MAX_LENGTH).sum()),
        "median_tokens": float(np.median(lengths)),
        "max_tokens": int(lengths.max()),
    }
    return vecs, stats


def build_faiss(vecs: np.ndarray, out_file: Path) -> None:
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(out_file))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N passages")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--device", default=None, help="mps | cpu (default: mps if available)")
    args = ap.parse_args()

    import torch
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    df, excluded = load_corpus(args.limit)
    ids = df["canonical_passage_id"].astype(str).tolist()
    texts = df["retrieval_text"].astype(str).tolist()
    log(f"{len(ids):,} usable passages to embed ({excluded:,} empty excluded) on {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out_dir / f"_shards_{fingerprint(ids, texts, args.shard_size)}"
    shard_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    vecs, stats = embed_passages(texts, device, args.batch_size, args.shard_size, shard_dir)
    embed_seconds = time.perf_counter() - t0

    emb_file = args.out_dir / "passage_embeddings_medcpt.npy"
    ids_file = args.out_dir / "passage_ids_medcpt.parquet"
    faiss_file = args.out_dir / "medcpt_flatip.faiss"
    np.save(emb_file, vecs)
    pd.DataFrame({"row": np.arange(len(ids)), "canonical_passage_id": ids}).to_parquet(
        ids_file, index=False)
    build_faiss(vecs, faiss_file)

    import faiss
    import transformers
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document_encoder": ARTICLE_MODEL,
        "query_encoder": "ncbi/MedCPT-Query-Encoder",
        "model_revision": stats["model_revision"],
        "pooling": "cls",
        "max_length": MAX_LENGTH,
        "normalized": False,
        "metric": "inner_product",
        "dimension": DIM,
        "faiss_index": "IndexFlatIP (exact)",
        "passage_count": len(ids),
        "excluded_empty_passages": excluded,
        "truncated_passages": stats["truncated_passages"],
        "median_tokens": stats["median_tokens"],
        "max_tokens": stats["max_tokens"],
        "text_column": "retrieval_text",
        "source_snapshot_id": sorted(df["source_snapshot_id"].astype(str).unique().tolist()),
        "pipeline_version": sorted(df["pipeline_version"].astype(str).unique().tolist()),
        "device": device,
        "batch_size": args.batch_size,
        "embed_seconds": round(embed_seconds, 1),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "faiss": faiss.__version__,
            "numpy": np.__version__,
        },
        "files": {
            "embeddings": emb_file.name,
            "ids": ids_file.name,
            "faiss": faiss_file.name,
        },
    }
    (args.out_dir / "medcpt_manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(shard_dir)
    log(f"done: {vecs.shape} in {embed_seconds / 60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
