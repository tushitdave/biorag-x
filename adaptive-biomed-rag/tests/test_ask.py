"""Chunking strategies, agentic fallback, fast BM25, and the per-question comparison.

LLM is off: agentic chunking must fall back to recursive without caching the fallback.
The comparison test pretends only whole-passage chunking is built, so it does not
depend on the background index build.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ["LLM_ENABLED"] = "false"

from api.contract import RetrievalConfig
from chunking import agentic, registry
from chunking.base import tokens
from common import paths
from chunking.strategies import (
    STRATEGIES, adaptive_chunks, document_profile, route, semantic_chunks,
)
from ingestion.loader import load_question_set

TEXT = ("BRCA1 participates in DNA damage repair. Loss of BRCA1 function increases "
        "homologous recombination deficiency and PARP inhibition can create synthetic "
        "lethality. However, one cohort did not show a significant treatment response. "
        "Olaparib is a PARP inhibitor.")


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_strategy_covers_the_passage(name):
    chunks = STRATEGIES[name]("P1", TEXT)
    assert chunks and all(c["parent_passage_id"] == "P1" and c["text"] for c in chunks)
    got = " ".join(c["text"] for c in chunks)
    # No words lost (proposition chunking drops the conjunctions it splits on).
    split_words = {"and", "but", "while", "whereas", "although", "because", "therefore", "however"}
    assert set(tokens(TEXT)) - split_words <= set(tokens(got))


def test_semantic_breaks_only_at_dissimilar_sentences():
    sims = [0.9, 0.1, 0.9]                             # 4 sentences, one weak transition
    chunks = semantic_chunks("P1", TEXT, sims, threshold=0.5, min_tokens=0)
    assert len(chunks) == 2


def test_adaptive_routes_and_tags():
    choice, _ = route(document_profile(TEXT))
    chunks = adaptive_chunks("P1", TEXT, lambda pid, t: semantic_chunks(pid, t, None, 0.0))
    assert all(c["strategy"] == "adaptive" and c["routed_to"] == choice for c in chunks)


def test_agentic_plan_validation():
    assert agentic.validate_plan([1, 3], 5)[0]
    assert not agentic.validate_plan([0], 5)[0]         # boundary must be >= 1
    assert not agentic.validate_plan([3, 1], 5)[0]      # sorted
    assert not agentic.validate_plan("1,2", 5)[0]


def test_agentic_falls_back_without_llm_and_does_not_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(agentic, "CACHE_DIR", tmp_path)
    chunks, meta = agentic.agentic_chunks([("P9", TEXT)])
    assert chunks and meta["fallback"] == 1 and meta["llm_calls"] == 0
    assert all(c["agentic_source"] == "fallback" for c in chunks)
    assert not list(tmp_path.iterdir())                # fallback is retried next time


def test_sparse_bm25_matches_rank_bm25():
    from rank_bm25 import BM25Okapi
    from indexes.bm25 import BM25Index, _tokenize
    docs = [TEXT, "PARP inhibitors in ovarian cancer.", "Aspirin inhibits COX-2.", "BRCA1 BRCA1"]
    fast = BM25Index(["a", "b", "c", "d"], ["a", "b", "c", "d"], docs)
    ref = BM25Okapi([_tokenize(d) for d in docs])
    for q in ["BRCA1 PARP", "aspirin", "BRCA1 BRCA1 inhibitor", "nothing here"]:
        assert np.allclose(fast.scores(q), ref.get_scores(_tokenize(q)), atol=1e-5)


def test_compare_on_a_dataset_question(monkeypatch):
    from retrieval import compare as cmp
    monkeypatch.setattr(registry, "is_ready", lambda s: s == "passage")
    q = load_question_set("dev_300").iloc[0]["question"]
    out = cmp.compare(q, RetrievalConfig())
    assert out["gold_available"] and out["n_gold"] > 0
    groups = [c["group"] for c in out["combos"]]
    assert groups[0] == "yours" and "retrieval" in groups
    from experiments.config import canonical
    keys = [canonical(RetrievalConfig(**c["config"])).model_dump_json() for c in out["combos"]]
    assert len(keys) == len(set(keys))                 # no behaviourally identical duplicates
    ok = [c for c in out["combos"] if not c["error"]]
    eligible = [c for c in ok if not c["self_judged"]]
    best = next(c for c in out["combos"] if c["is_best"])
    assert not best["self_judged"]                     # judge's own reranker can't win
    assert any(c["self_judged"] for c in ok)
    assert best["evidence_score"] == max(c["evidence_score"] for c in eligible)
    for c in ok:
        assert 0 <= c["evidence_score"] <= 1 and c["true_metrics"] is not None
        assert all(0 <= e["judge"] <= 1 for e in c["evidence"])


def test_ask_answer_offline(monkeypatch):
    from fastapi.testclient import TestClient
    from api.app import app
    from common.llm import llm_usage
    with TestClient(app) as client:
        r = client.post("/ask/answer", json={"question": "What does olaparib inhibit?",
                                             "config": RetrievalConfig().model_dump()})
        assert r.status_code == 200 and r.json()["llm_used"] is False
        bad = client.post("/ask/compare", json={"question": "x",
                                                "config": {"chunking": "no_such_strategy"}})
        assert bad.status_code == 400
    assert llm_usage()["live_calls"] == 0


@pytest.mark.skipif(not (paths.GRAPH_DIR / "manifest.json").exists()
                    or not (paths.PAGEINDEX_DIR / "manifest.json").exists(),
                    reason="graph / PageIndex not built")
@pytest.mark.parametrize("channel, path_key", [("graph", "graph_path"), ("pageindex", "tree_path")])
def test_graph_and_pageindex_channels(channel, path_key):
    from experiments.config import to_recipe
    from retrieval.hybrid import retrieve
    q = "Is olaparib effective in BRCA1 mutated ovarian cancer?"
    res = retrieve(q, to_recipe(RetrievalConfig(chunking="parent_child", channels=["lexical", channel])))
    assert res["trace"][f"{channel}_hits"] > 0
    assert any(p.get(path_key) for p in res["reranked"])       # every hit says how it was found


def test_schwartz_hearst_long_forms():
    from retrieval.synonyms import best_long_form, definitions
    assert best_long_form("ECMO", "patients treated with extracorporeal membrane oxygenation") == \
        "extracorporeal membrane oxygenation"
    assert best_long_form("XYZ", "no matching letters here") is None
    assert ("SRY", "sex determining region y") in set(
        definitions("the sex determining region Y (SRY) gene"))


@pytest.mark.skipif(not (paths.SYNONYM_DIR / "manifest.json").exists(), reason="synonyms not built")
def test_router_rules_and_synonyms(monkeypatch):
    from retrieval import router
    from experiments.config import to_recipe
    from retrieval.hybrid import retrieve
    f = router.features("Can RG7112 inhibit MDM2?")
    channels, mode, fired = router.plan(f)
    assert "graph" in channels and mode == "synonyms" and len(fired) == 2
    assert router.plan(router.features("Describe the role of autophagy in neurodegeneration."))[0][-1] \
        == "pageindex"
    res = retrieve("What is ECMO?", to_recipe(RetrievalConfig(query="synonyms")))
    assert res["trace"]["synonyms"][0]["added"] == "extracorporeal membrane oxygenation"
    # weak first round -> exactly one retry with every ready channel
    monkeypatch.setattr(router, "WEAK_COVERAGE", 1.01)
    res = retrieve("What is ECMO?", to_recipe(RetrievalConfig(query="router")))
    rounds = res["trace"]["router"]["rounds"]
    assert len(rounds) == 2 and {"graph", "pageindex"} <= set(rounds[1]["channels"])
