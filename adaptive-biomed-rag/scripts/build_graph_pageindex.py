"""Build the evidence graph, PageIndex tree and synonym dictionary over all passages (CPU, ~20 s).

All are rules / clustering only: no LLM calls, no new embeddings (PageIndex groups
passages with the MedCPT vectors already in data/processed/indexes/dense/).

Usage (from adaptive-biomed-rag/):
    python scripts/build_graph_pageindex.py                 # both
    python scripts/build_graph_pageindex.py --only graph      # or pageindex | synonyms
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from ingestion.loader import load_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=["graph", "pageindex", "synonyms"])
    args = ap.parse_args()
    df = load_chunks("passage")
    ids, texts = df["chunk_id"].astype(str).tolist(), df["text"].tolist()

    if args.only in (None, "graph"):
        from graph.evidence_graph import build
        t0 = time.perf_counter()
        m = build(list(zip(ids, texts)), paths.GRAPH_DIR)
        print(f"graph built in {time.perf_counter() - t0:.0f}s\n{json.dumps(m, indent=2)}")

    if args.only in (None, "synonyms"):
        from retrieval.synonyms import build
        t0 = time.perf_counter()
        m = build(texts, paths.SYNONYM_DIR)
        print(f"synonyms built in {time.perf_counter() - t0:.0f}s\n{json.dumps(m, indent=2)}")

    if args.only in (None, "pageindex"):
        from pageindex.tree import build
        t0 = time.perf_counter()
        vec_ids = pd.read_parquet(paths.MEDCPT_IDS)["canonical_passage_id"].astype(str).tolist()
        vectors = np.load(paths.MEDCPT_EMBEDDINGS, mmap_mode="r")
        pos = {p: i for i, p in enumerate(vec_ids)}
        keep = [i for i, p in enumerate(ids) if p in pos]
        m = build([ids[i] for i in keep], [texts[i] for i in keep],
                  np.asarray(vectors[[pos[ids[i]] for i in keep]]), paths.PAGEINDEX_DIR)
        print(f"pageindex built in {time.perf_counter() - t0:.0f}s\n{json.dumps(m, indent=2)}")


if __name__ == "__main__":
    main()
