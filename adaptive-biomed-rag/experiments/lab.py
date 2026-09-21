"""Retrieval lab: run a RetrievalConfig on a frozen question set, compare two runs.

A run retrieves for every question, scores the ranked pool against the gold
passages, and stores per-question rows, means with 95% bootstrap CIs, and slices
by question type / lexical overlap / evidence size. A comparison pairs two runs
question-by-question, so a delta is only called significant when its paired
bootstrap CI excludes zero. Runs execute one at a time in a background thread;
retrieval-only runs make no LLM calls.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from api.contract import RetrievalConfig
from evaluation.retrieval_metrics import per_query_metrics
from experiments import registry
from experiments.config import changes, to_recipe
from indexes.dense import encode_medcpt_query, get_dense
from ingestion.loader import QUESTION_SETS, load_question_set
from retrieval.hybrid import retrieve

# key, label, unit, higher_is_better
METRICS = [
    ("hit@1", "Hit@1", "", True),
    ("hit@5", "Hit@5", "", True),
    ("hit@10", "Hit@10", "", True),
    ("recall@5", "Recall@5", "", True),
    ("recall@10", "Recall@10", "", True),
    ("recall@20", "Recall@20", "", True),
    ("mrr@10", "MRR@10", "", True),
    ("ndcg@10", "nDCG@10", "", True),
    ("evidence_recall", "Evidence recall", "", True),
    ("evidence_precision", "Evidence precision", "", True),
    ("latency_ms", "Latency (mean)", "ms", False),
    ("ann_overlap@10", "ANN agreement@10", "", True),
]
HEADLINE = ["hit@10", "recall@10", "mrr@10", "ndcg@10", "evidence_recall", "latency_p50"]
SLICE_DIMS = {
    "question_type": "question_type_inferred",
    "lexical_overlap": "lexical_overlap_stratum",
    "evidence_size": "difficulty_stratum",
}
SLICE_METRICS = ["hit@10", "recall@10", "ndcg@10"]
DECISION_METRIC = "ndcg@10"
N_BOOT = 1000

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab-run")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _unique(ids) -> list[str]:
    return list(dict.fromkeys(str(i) for i in ids))


def _evaluate_question(row, recipe: dict, cfg: RetrievalConfig, corpus: str) -> dict:
    res = retrieve(row.question, recipe)
    ranked = _unique(p["parent_passage_id"] for p in res["reranked"])
    evidence = set(_unique(p["parent_passage_id"] for p in res["passages"]))
    gold = set(row.gold_ids)

    m = per_query_metrics(ranked, gold, ks=(1, 5, 10, 20))
    m["evidence_recall"] = len(evidence & gold) / len(gold)
    m["evidence_precision"] = len(evidence & gold) / len(evidence) if evidence else 0.0
    m["latency_ms"] = res["trace"]["latency_ms"]
    m["gold_in_top20"] = len(set(ranked[:20]) & gold)

    if cfg.index != "flat" and "dense" in cfg.channels and cfg.embedding == "medcpt":
        # How many of the exact dense top-10 the ANN index also returns.
        exact = get_dense(cfg.chunking, "flat", "medcpt", corpus).search(row.question, 10)
        approx = get_dense(cfg.chunking, cfg.index, "medcpt", corpus).search(
            row.question, 10, ef_search=cfg.ef_search, nprobe=cfg.nprobe)
        m["ann_overlap@10"] = len({p["chunk_id"] for p in exact} &
                                  {p["chunk_id"] for p in approx}) / 10

    return {
        "question_id": row.canonical_question_id,
        "question": row.question,
        "question_type_inferred": row.question_type_inferred,
        "lexical_overlap_stratum": row.lexical_overlap_stratum,
        "difficulty_stratum": row.difficulty_stratum,
        "n_gold": int(row.n_gold),
        **m,
    }


def _bootstrap_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _aggregate(pq: pd.DataFrame) -> list[dict]:
    out = []
    for key, label, unit, _ in METRICS:
        if key not in pq:
            continue
        v = pq[key].to_numpy(dtype=float)
        lo, hi = _bootstrap_ci(v)
        out.append({"key": key, "label": label, "value": round(float(v.mean()), 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4), "unit": unit})
    lat = pq["latency_ms"].to_numpy(dtype=float)
    out.append({"key": "latency_p50", "label": "Latency p50", "unit": "ms",
                "value": round(float(np.percentile(lat, 50)), 1)})
    out.append({"key": "latency_p95", "label": "Latency p95", "unit": "ms",
                "value": round(float(np.percentile(lat, 95)), 1)})
    return out


def _slices(pq: pd.DataFrame) -> list[dict]:
    out = []
    for dim, col in SLICE_DIMS.items():
        for value, g in pq.groupby(col, sort=True):
            out.append({"dimension": dim, "value": str(value), "n": int(len(g)),
                        "metrics": {k: round(float(g[k].mean()), 4) for k in SLICE_METRICS}})
    return out


def _run(rid: str, cfg: RetrievalConfig, recipe: dict, qs: pd.DataFrame, corpus: str) -> None:
    t0 = time.perf_counter()
    try:
        registry.start(rid)
        encode_medcpt_query.cache_clear()   # every run pays its own query-encoding latency
        rows = []
        for i, row in enumerate(qs.itertuples(index=False), start=1):
            rows.append(_evaluate_question(row, recipe, cfg, corpus))
            if i % 10 == 0:
                registry.progress(rid, i)
        pq = pd.DataFrame(rows)
        registry.finish(rid, _aggregate(pq), _slices(pq), pq, time.perf_counter() - t0)
    except Exception as e:  # recorded on the run so the UI can show it
        registry.fail(rid, f"{type(e).__name__}: {e}")


def submit(cfg: RetrievalConfig, question_set: str = "dev_300", corpus: str = "dev",
           name: str = "", wait: bool = False) -> str:
    """Validate and queue a run; returns its id. ``wait`` runs it in this thread."""
    if question_set not in QUESTION_SETS:
        raise ValueError(f"Unknown question set: {question_set}")
    if question_set == "locked_100":
        corpus = "full"                     # the final check always uses the real corpus
    if cfg.reranker == "llm_reranker" or cfg.chunking == "agentic":
        raise ValueError("GPT-4o reranking and agentic chunking spend LLM calls per "
                         "question; they are not allowed in batch runs")
    recipe = to_recipe(cfg, corpus)
    qs = load_question_set(question_set)
    rid = registry.create(name.strip(), cfg.model_dump(), question_set, corpus, len(qs))
    if wait:
        _run(rid, cfg, recipe, qs, corpus)
    else:
        _executor.submit(_run, rid, cfg, recipe, qs, corpus)
    return rid


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def summary(run: dict) -> dict:
    by_key = {m["key"]: m["value"] for m in run["metrics"]}
    return {
        "id": run["id"], "name": run["name"], "created_at": run["created_at"],
        "status": run["status"], "question_set": run["question_set"], "corpus": run["corpus"],
        "n_questions": run["n_questions"], "done": run["done"], "config": run["config"],
        "headline": {k: by_key[k] for k in HEADLINE if k in by_key},
        "duration_s": run["duration_s"], "error": run["error"],
    }


def detail(run: dict) -> dict:
    missed = []
    if run["status"] == "done":
        pq = registry.per_query(run["id"])
        miss = pq[pq["hit@10"] == 0].sort_values("n_gold", ascending=False).head(25)
        missed = [{"question": r.question, "question_type": r.question_type_inferred,
                   "n_gold": int(r.n_gold), "gold_in_top20": int(r.gold_in_top20)}
                  for r in miss.itertuples(index=False)]
    return {"run": summary(run), "metrics": run["metrics"], "slices": run["slices"],
            "missed": missed}


def compare(a: dict, b: dict) -> dict:
    """Paired comparison of run B against run A on the questions both answered."""
    reasons = []
    if a["question_set"] != b["question_set"]:
        reasons.append("different question sets")
    if a["corpus"] != b["corpus"]:
        reasons.append("different corpora")
    pa, pb = registry.per_query(a["id"]), registry.per_query(b["id"])
    j = pa.merge(pb, on="question_id", suffixes=("_a", "_b"))

    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(j), size=(N_BOOT, len(j)))
    metrics = []
    for key, label, _unit, hib in METRICS:
        if f"{key}_a" not in j or f"{key}_b" not in j:
            continue
        va, vb = j[f"{key}_a"].to_numpy(float), j[f"{key}_b"].to_numpy(float)
        d = vb - va
        lo, hi = np.percentile(d[boot].mean(axis=1), [2.5, 97.5])
        metrics.append({"key": key, "label": label, "a": round(float(va.mean()), 4),
                        "b": round(float(vb.mean()), 4), "delta": round(float(d.mean()), 4),
                        "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                        "significant": bool(lo > 0 or hi < 0), "higher_is_better": hib})

    dm = j[f"{DECISION_METRIC}_b"] - j[f"{DECISION_METRIC}_a"]
    slices = []
    for dim, col in SLICE_DIMS.items():
        for value, g in j.groupby(f"{col}_a", sort=True):
            slices.append({"dimension": dim, "value": str(value), "n": int(len(g)),
                           "delta": {k: round(float((g[f"{k}_b"] - g[f"{k}_a"]).mean()), 4)
                                     for k in SLICE_METRICS}})

    def top(order: pd.Series) -> list[dict]:
        return [{"question": j.at[i, "question_a"],
                 "a": round(float(j.at[i, f"{DECISION_METRIC}_a"]), 4),
                 "b": round(float(j.at[i, f"{DECISION_METRIC}_b"]), 4)} for i in order.index]

    ca, cb = RetrievalConfig(**a["config"]), RetrievalConfig(**b["config"])
    return {
        "a": summary(a), "b": summary(b),
        "comparable": not reasons, "reason": "; ".join(reasons),
        "changes": [{"field": f, "a": x, "b": y} for f, x, y in changes(ca, cb)],
        "metrics": metrics,
        "decision_metric": DECISION_METRIC,
        "wins": int((dm > 1e-9).sum()), "losses": int((dm < -1e-9).sum()),
        "ties": int((dm.abs() <= 1e-9).sum()),
        "slices": slices,
        "top_gains": top(dm[dm > 1e-9].sort_values(ascending=False).head(5)),
        "top_losses": top(dm[dm < -1e-9].sort_values().head(5)),
    }
