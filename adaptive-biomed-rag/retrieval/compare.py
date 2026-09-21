"""Per-question comparison of retrieval combinations (the Chat trust report).

Candidates (duplicates removed):
  yours      the flow chosen in the top bar
  chunking   every ready chunking strategy with the standard search
             (BM25 + dense, RRF k=60, no rerank, MMR 5)
  retrieval  on your chunking: BM25 only, dense only, hybrid, hybrid + MedCPT cross-encoder,
             graph traversal only, PageIndex only, hybrid + graph, hybrid + PageIndex
             (the last four when their index is built)
  query      on your chunking: synonym expansion, agentic router

Each runs on the full corpus and is scored on the evidence it would send to the LLM:
  evidence_score  mean judge relevance of that evidence (0..1) - decides the winner
  best_evidence   judge relevance of its single best passage
  coverage        share of the question's content words found in the evidence
  agreement       share of BM25's top-10 passages that dense search also ranks top-10
  true_metrics    Hit@5 / Recall@10 / nDCG@10 / evidence recall against the gold
                  passages, when the question is a dataset question
Agentic chunking (an LLM call) only runs when it is your own choice.

The judge is the MedCPT cross-encoder, which is also one of the rerankers. A
combination reranked by it is judged by its own ranker, so its evidence score is
inflated by construction: such rows are flagged (``self_judged``) and are not
eligible to be "best".
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache

from api.contract import RetrievalConfig
from chunking import registry
from common import paths
from evaluation.retrieval_metrics import per_query_metrics
from experiments.config import canonical, embedding_for, to_recipe
from ingestion.loader import load_questions
from retrieval.hybrid import retrieve
from retrieval.structured import is_ready as channel_ready
from retrieval.judge import JUDGE_MODEL, JUDGE_RERANKER, relevance

COMPARE_RERANK_CANDIDATES = 20    # cross-encoder depth in the comparison (~1.3 s on M1)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = set("the a an of and or to in for with on by is are was were be as at from that this these "
            "those what which who how why when where does do did can may it its their there any "
            "into than then also not no such via per about".split())


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if w.lower() not in _STOP and len(w) > 2}


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


@lru_cache(maxsize=1)
def _gold_index() -> dict[str, list[str]]:
    q = load_questions()
    q = q[q["has_usable_gold_evidence"] == True]  # noqa: E712
    return {_norm(t): [str(x) for x in g] for t, g in zip(q["question"], q["usable_gold_canonical_ids"])}


def _calibration() -> dict | None:
    """Judge-vs-gold agreement measured by scripts/calibrate_judge.py, if run."""
    try:
        return json.loads((paths.PROCESSED_DIR / "judge_calibration.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def gold_for(question: str) -> list[str]:
    return _gold_index().get(_norm(question), [])


def candidates(user: RetrievalConfig) -> list[tuple[str, RetrievalConfig]]:
    """(group, config) pairs, user's first, without duplicates."""
    out: list[tuple[str, RetrievalConfig]] = [("yours", user)]
    for s in registry.STRATEGIES:
        if s == "agentic" or not registry.is_ready(s):
            continue
        out.append(("chunking", RetrievalConfig(chunking=s, embedding=embedding_for(s))))
    base = user.chunking if user.chunking != "agentic" else "passage"
    emb = embedding_for(base)
    variants = [(["lexical"], "none"), (["dense"], "none"), (["lexical", "dense"], "none"),
                (["lexical", "dense"], "medcpt_ce")]
    for ch in ("graph", "pageindex"):
        if channel_ready(ch):
            variants += [([ch], "none"), (["lexical", "dense", ch], "none")]
    for mode in ("synonyms", "router"):
        if mode == "router" or channel_ready("synonyms"):
            out.append(("query", RetrievalConfig(chunking=base, embedding=emb, query=mode)))
    for channels, reranker in variants:
        out.append(("retrieval", RetrievalConfig(chunking=base, embedding=emb, channels=channels,
                                                 reranker=reranker,
                                                 rerank_candidates=COMPARE_RERANK_CANDIDATES)))
    seen, unique = set(), []
    for group, cfg in out:
        key = canonical(cfg).model_dump_json()
        if key not in seen:
            seen.add(key)
            unique.append((group, cfg))
    return unique


def _structured_notes(tr: dict) -> list[str]:
    """One line each on what graph traversal / PageIndex did for this question."""
    notes = []
    if "graph" in tr:
        g = tr["graph"]
        nbrs = ", ".join(n["entity"] for n in g["neighbors"][:4]) or "no neighbours"
        notes.append(f"Graph: question entities {', '.join(g['seeds'])}; followed {nbrs}."
                     if g["seeds"] else f"Graph: {g.get('note', 'no question entities')}.")
    if tr.get("synonyms"):
        notes.append("Synonyms added: " + ", ".join(f"{a['added']} (for {a['term']})"
                                                   for a in tr["synonyms"]) + ".")
    if "router" in tr:
        r = tr["router"]
        steps = "; ".join(r["rules"])
        rounds = r["rounds"]
        retry = (f" Coverage {rounds[0]['coverage']:.2f} < {r['weak_coverage']} → round 2 "
                 f"({' + '.join(rounds[1]['channels'])}, coverage {rounds[1]['coverage']:.2f}); "
                 f"kept round {r['chosen_round']}." if len(rounds) > 1 else
                 f" Coverage {rounds[0]['coverage']:.2f}: no retry.")
        notes.append(f"Router: {steps}.{retry}")
    if "pageindex" in tr:
        p = tr["pageindex"]
        notes.append(f"PageIndex: read {len(p['leaves'])} sections "
                     f"({p.get('passages_read', 0)} passages), first '{p['leaves'][0]['label']}'."
                     if p["leaves"] else f"PageIndex: {p.get('note', 'nothing read')}.")
    return notes


def _evidence_items(res: dict, judge: list[float], gold: set[str]) -> list[dict]:
    items = []
    for p, j in zip(res["passages"], judge):
        pid = str(p["parent_passage_id"])
        scores = {k: float(p[k]) for k in ("bm25_score", "dense_score", "graph_score",
                                           "pageindex_score", "rrf_score", "rerank_score")
                  if p.get(k) is not None}
        why = " | ".join(x for x in (p.get("graph_path") and f"graph: {p['graph_path']}",
                                     p.get("tree_path") and f"tree: {p['tree_path']}") if x)
        items.append({
            "passage_id": pid, "chunk_id": str(p.get("chunk_id", pid)), "text": p["text"],
            "child_text": p.get("child_text"), "judge": round(j, 4), "scores": scores,
            "why": why or None,
            "rank": int(p.get("final_rank", 0)), "is_gold": (pid in gold) if gold else None,
        })
    return items


def compare(question: str, user: RetrievalConfig) -> dict:
    t0 = time.perf_counter()
    gold_list = gold_for(question)
    gold = set(gold_list)
    qwords = _content_words(question)

    runs = []
    for i, (group, cfg) in enumerate(candidates(user)):
        self_judged = cfg.reranker == JUDGE_RERANKER
        row = {"id": f"c{i}", "group": group, "config": cfg.model_dump(), "is_user": i == 0,
               "is_best": False, "self_judged": self_judged, "evidence_score": 0.0,
               "best_evidence": 0.0, "coverage": 0.0, "agreement": None, "latency_ms": 0.0,
               "true_metrics": None, "evidence": [], "error": "",
               "note": ("Reranked by the judge's own model, so its evidence score is inflated; "
                        "not eligible for best." if self_judged else "")}
        try:
            res = retrieve(question, to_recipe(cfg))
        except Exception as e:                    # one broken combination must not sink the report
            row["error"] = f"{type(e).__name__}: {e}"
            runs.append((row, None))
            continue
        tr = res["trace"]
        row["latency_ms"] = tr.get("latency_ms", 0.0)
        if tr.get("bm25_top") and tr.get("dense_top"):
            row["agreement"] = round(len(set(tr["bm25_top"]) & set(tr["dense_top"])) / 10, 2)
        if "agentic" in tr:
            m = tr["agentic"]
            agentic_note = (f"Agentic chunking: {m['llm_planned']} new LLM plans, {m['cached']} "
                            f"cached, {m['fallback']} fell back to recursive "
                            f"(LLM {m['llm_source']}).")
            row["note"] = f"{row['note']} {agentic_note}".strip()
        for note in _structured_notes(tr):
            row["note"] = f"{row['note']} {note}".strip()
        runs.append((row, res))

    # Judge every evidence text once, across all combinations.
    texts = [p["text"] for _, res in runs if res for p in res["passages"]]
    scores = dict(zip(texts, relevance(question, texts))) if texts else {}

    for row, res in runs:
        if not res:
            continue
        judge = [scores[p["text"]] for p in res["passages"]]
        row["evidence"] = _evidence_items(res, judge, gold)
        if judge:
            row["evidence_score"] = round(sum(judge) / len(judge), 4)
            row["best_evidence"] = round(max(judge), 4)
        ev_words = set().union(*(_content_words(p["text"]) for p in res["passages"])) \
            if res["passages"] else set()
        row["coverage"] = round(len(qwords & ev_words) / len(qwords), 3) if qwords else 0.0
        if gold:
            ranked = list(dict.fromkeys(str(p["parent_passage_id"]) for p in res["reranked"]))
            m = per_query_metrics(ranked, gold, ks=(5, 10))
            ev = {e["passage_id"] for e in row["evidence"]}
            row["true_metrics"] = {"hit@5": m["hit@5"], "recall@10": m["recall@10"],
                                   "ndcg@10": m["ndcg@10"],
                                   "evidence_recall": round(len(ev & gold) / len(gold), 4)}

    rows = [r for r, _ in runs]
    ok = [r for r in rows if not r["error"] and r["evidence"] and not r["self_judged"]]
    best = max(ok, key=lambda r: (r["evidence_score"], -r["latency_ms"])) if ok else rows[0]
    best["is_best"] = True
    return {
        "question": question,
        "gold_available": bool(gold),
        "n_gold": len(gold),
        "combos": rows,
        "best_id": best["id"],
        "user_id": rows[0]["id"],
        "judge": {"model": JUDGE_MODEL, "calibration": _calibration(),
                  "how": "Mean relevance (0-1) of the evidence each combination would send to "
                         "the LLM, judged by the MedCPT cross-encoder. Estimated: typed questions "
                         "have no gold passages. Rows reranked by that same model are not "
                         "eligible for best."},
        "seconds": round(time.perf_counter() - t0, 2),
    }
