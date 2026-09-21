"""How far can the per-question judge be trusted? Measured on dev questions with gold.

For each question the Ask comparison is run (default flow as "yours"). Among the
combinations eligible to win (not self-judged), we compare the judge's pick with the
oracle - the combination with the best true nDCG@10 against the gold passages:

  pick_is_oracle   share of questions where the judge picked a truly best combination
  regret           mean (oracle nDCG@10 - picked nDCG@10)
  picked_ndcg      mean true nDCG@10 of the judge's pick
  default_ndcg     mean true nDCG@10 of the default flow (no per-question choice)
  oracle_ndcg      mean true nDCG@10 if the best combination were always known
  rank_corr        mean Spearman correlation, per question, of evidence score vs true nDCG

Written to data/processed/judge_calibration.json and shown on the Ask page.
Retrieval + local judge only: no LLM calls.

Usage (from adaptive-biomed-rag/):  python scripts/calibrate_judge.py --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.contract import RetrievalConfig  # noqa: E402
from chunking import registry  # noqa: E402
from common import paths  # noqa: E402
from ingestion.loader import load_question_set  # noqa: E402
from retrieval.compare import compare  # noqa: E402

OUT = paths.PROCESSED_DIR / "judge_calibration.json"


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100, help="dev questions to use")
    args = ap.parse_args()

    dev = load_question_set("dev_300")                  # ordered by type: sample evenly
    qs = dev.iloc[np.linspace(0, len(dev) - 1, min(args.n, len(dev))).astype(int)]
    t0 = time.perf_counter()
    picked, default, oracle, hits, corrs = [], [], [], [], []
    for i, q in enumerate(qs["question"], start=1):
        out = compare(q, RetrievalConfig())
        rows = [c for c in out["combos"] if not c["error"] and c["true_metrics"]]
        eligible = [c for c in rows if not c["self_judged"]]
        if not eligible:
            continue
        nd = {c["id"]: c["true_metrics"]["ndcg@10"] for c in rows}
        best = next(c for c in out["combos"] if c["is_best"])
        top = max(nd[c["id"]] for c in eligible)
        picked.append(nd[best["id"]])
        default.append(nd[out["user_id"]])
        oracle.append(top)
        hits.append(nd[best["id"]] >= top - 1e-9)
        r = spearman([c["evidence_score"] for c in eligible], [nd[c["id"]] for c in eligible])
        if r is not None:
            corrs.append(r)
        if i % 10 == 0:
            print(f"{i}/{len(qs)} questions | pick=oracle {np.mean(hits):.0%} "
                  f"| {time.perf_counter() - t0:.0f}s", flush=True)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": len(picked),
        "strategies": [s for s in registry.STRATEGIES if registry.is_ready(s) and s != "agentic"],
        "pick_is_oracle": round(float(np.mean(hits)), 3),
        "regret": round(float(np.mean(np.array(oracle) - np.array(picked))), 4),
        "picked_ndcg": round(float(np.mean(picked)), 4),
        "default_ndcg": round(float(np.mean(default)), 4),
        "oracle_ndcg": round(float(np.mean(oracle)), 4),
        "rank_corr": round(float(np.mean(corrs)), 3) if corrs else None,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
