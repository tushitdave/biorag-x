"""Freeze the benchmark: dev_300 (for tuning), locked_100 (final check only), dev corpus.

Questions are drawn from those with usable gold evidence and stratified on NB01's
inferred question type x lexical-overlap quartile, so every type and overlap level
is represented:
  locked_100 : 20 per inferred type (5 per overlap quartile)
  dev_300    : 60 per inferred type (15 per overlap quartile), disjoint from locked
Exact-duplicate questions (same question hash) are kept on one side of the split.

The dev corpus is every usable gold passage of dev_300 plus seeded random distractors
up to --corpus-size passages. Labs run on it; winners are confirmed on the full corpus.

Outputs in adaptive-biomed-rag/data/benchmark/ (refuses to overwrite without --force,
because the locked set must never change silently):
  dev_300.parquet, locked_100.parquet, dev_corpus_ids.parquet, benchmark_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402

TYPES = ["yes_no_candidate", "factoid_candidate", "list_or_factoid_candidate",
         "summary_or_mechanism_candidate", "other"]
QUARTILES = ["low_overlap", "medium_low", "medium_high", "high_overlap"]


def load_pool() -> pd.DataFrame:
    q = pd.read_parquet(paths.QUESTIONS_PARQUET)
    q = q[q["has_usable_gold_evidence"] == True].copy()  # noqa: E712
    f = pd.read_parquet(paths.NB01_QA_FORENSICS)[
        ["question_id", "question_type_inferred", "lexical_overlap_stratum", "difficulty_stratum"]]
    q = q.merge(f, left_on="id", right_on="question_id", how="inner")
    q["gold_ids"] = q["usable_gold_canonical_ids"].map(lambda g: [str(x) for x in g])
    q["n_gold"] = q["gold_ids"].map(len)
    for c in ("question_type_inferred", "lexical_overlap_stratum", "difficulty_stratum"):
        q[c] = q[c].astype(str)
    return q


def stratified_take(pool: pd.DataFrame, per_type: int, rng: np.random.Generator) -> list[int]:
    """Row indices: per_type per inferred type, spread evenly over overlap quartiles."""
    chosen: list[int] = []
    for t in TYPES:
        of_type = pool[pool["question_type_inferred"] == t]
        take: list[int] = []
        for qt in QUARTILES:
            cell = of_type[of_type["lexical_overlap_stratum"] == qt].index.to_numpy()
            k = min(per_type // len(QUARTILES), len(cell))
            take += rng.choice(cell, size=k, replace=False).tolist()
        rest = np.setdiff1d(of_type.index.to_numpy(), take)       # top up thin quartiles
        take += rng.choice(rest, size=per_type - len(take), replace=False).tolist()
        chosen += take
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--corpus-size", type=int, default=5000)
    ap.add_argument("--force", action="store_true", help="overwrite an existing benchmark")
    args = ap.parse_args()

    out = paths.BENCHMARK_DIR
    if paths.LOCKED_QUESTIONS.exists() and not args.force:
        sys.exit(f"Benchmark already frozen in {out}; pass --force to rebuild it.")

    rng = np.random.default_rng(args.seed)
    pool = load_pool()

    locked_idx = stratified_take(pool, 20, rng)
    locked_hashes = set(pool.loc[locked_idx, "question_sha256"])
    remaining = pool.drop(index=locked_idx)
    remaining = remaining[~remaining["question_sha256"].isin(locked_hashes)]
    dev_idx = stratified_take(remaining, 60, rng)

    cols = ["canonical_question_id", "question", "answer", "gold_ids", "n_gold",
            "question_type_inferred", "lexical_overlap_stratum", "difficulty_stratum",
            "question_sha256"]
    locked = pool.loc[locked_idx, cols].reset_index(drop=True)
    dev = pool.loc[dev_idx, cols].reset_index(drop=True)
    assert not set(dev["question_sha256"]) & set(locked["question_sha256"])

    passages = pd.read_parquet(paths.PASSAGES_PARQUET,
                               columns=["canonical_passage_id", "is_usable", "retrieval_text"])
    usable = passages[passages["is_usable"] & passages["retrieval_text"].str.strip().ne("")]
    usable_ids = usable["canonical_passage_id"].astype(str).to_numpy()
    dev_gold = sorted({g for ids in dev["gold_ids"] for g in ids} & set(usable_ids))
    n_distractors = max(0, args.corpus_size - len(dev_gold))
    distractors = rng.choice(np.setdiff1d(usable_ids, dev_gold), size=n_distractors, replace=False)
    corpus = pd.DataFrame({
        "canonical_passage_id": np.concatenate([dev_gold, np.sort(distractors)]),
        "is_dev_gold": [True] * len(dev_gold) + [False] * n_distractors,
    })

    dev.to_parquet(paths.DEV_QUESTIONS, index=False)
    locked.to_parquet(paths.LOCKED_QUESTIONS, index=False)
    corpus.to_parquet(paths.DEV_CORPUS_IDS, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "eligible_questions": int(len(pool)),
        "dev_300": int(len(dev)),
        "locked_100": int(len(locked)),
        "dev_by_type": dev["question_type_inferred"].value_counts().to_dict(),
        "locked_by_type": locked["question_type_inferred"].value_counts().to_dict(),
        "dev_corpus_passages": int(len(corpus)),
        "dev_corpus_gold": len(dev_gold),
        "dev_corpus_distractors": int(n_distractors),
        "note": "question types are NB01 heuristics, not BioASQ labels",
    }
    (out / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
