"""Backend tests for BioRAG-X.

Covers:
  * retrieval metric math (recall/hit/MRR/nDCG) on hand-checked cases,
  * citation validation labelling + precision/recall,
  * /chat with the LLM DISABLED returns a valid extractive ChatResponse and makes
    ZERO live LLM calls (llm_usage live_calls == 0).

Runs fully offline. The /chat test hits the real retrieval stack (BM25 + dense),
so it exercises the extractive fallback end-to-end without any model call.
"""
from __future__ import annotations

import os

import pytest

# Force the LLM master switch OFF before importing anything that reads settings.
os.environ["LLM_ENABLED"] = "false"

from evaluation.retrieval_metrics import (
    recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k, per_query_metrics, aggregate_metrics,
)
from evaluation.bioasq_metrics import token_f1
from generation.citation_validator import validate_claims
from api.contract import Claim
from retrieval.reranker import rerank


# --------------------------------------------------------------------------- #
# Retrieval metric math
# --------------------------------------------------------------------------- #
def test_recall_and_hit():
    retrieved = ["p1", "p2", "p3", "p4", "p5"]
    gold = {"p3", "p9"}
    assert recall_at_k(retrieved, gold, 5) == pytest.approx(0.5)   # 1 of 2 gold
    assert hit_at_k(retrieved, gold, 5) == 1.0
    assert hit_at_k(retrieved, gold, 2) == 0.0                     # p3 not in top2


def test_mrr():
    # First gold at rank 3 -> 1/3.
    assert mrr_at_k(["a", "b", "g", "d"], {"g"}, 10) == pytest.approx(1.0 / 3.0)
    # Gold at rank 1 -> 1.0.
    assert mrr_at_k(["g", "b"], {"g"}, 10) == 1.0
    # No gold in list -> 0.
    assert mrr_at_k(["a", "b"], {"g"}, 10) == 0.0


def test_ndcg():
    import math
    # Single gold at rank 1 -> perfect nDCG.
    assert ndcg_at_k(["g", "x", "y"], {"g"}, 10) == pytest.approx(1.0)
    # Single gold at rank 2 -> dcg=1/log2(3), idcg=1/log2(2)=1.
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(["x", "g", "y"], {"g"}, 10) == pytest.approx(expected, rel=1e-6)


def test_per_query_and_aggregate():
    rows = [
        per_query_metrics(["g", "b", "c"], {"g"}),
        per_query_metrics(["b", "c", "g"], {"g"}),
    ]
    agg = aggregate_metrics(rows)
    assert 0.0 <= agg["ndcg@10"] <= 1.0
    assert agg["hit@5"] == 1.0
    assert "mrr@10" in agg


def test_token_f1():
    assert token_f1("the BRCA1 gene", "BRCA1 gene") == pytest.approx(1.0)
    assert token_f1("", "something") == 0.0


@pytest.mark.parametrize("method", ["none", "lexical_overlap", "cross_encoder", "colbert"])
def test_local_reranker_options(method):
    candidates = [
        {"text": "Metformin improves insulin sensitivity in type 2 diabetes."},
        {"text": "Aspirin inhibits platelet aggregation."},
    ]
    ranked = rerank("metformin type 2 diabetes", candidates, method=method, top_k=2)
    assert len(ranked) == 2
    assert "Metformin" in ranked[0]["text"]


# --------------------------------------------------------------------------- #
# Citation validation
# --------------------------------------------------------------------------- #
def test_citation_validation_supported():
    evidence = [
        {"parent_passage_id": "P1", "text": "Aspirin inhibits platelet aggregation "
                                            "by blocking cyclooxygenase enzymes."},
        {"parent_passage_id": "P2", "text": "Statins lower cholesterol synthesis."},
    ]
    claims = [
        Claim(text="Aspirin inhibits platelet aggregation via cyclooxygenase.",
              citations=["P1"]),
        Claim(text="Aspirin cures cancer completely overnight.", citations=["P99"]),
    ]
    claims, metrics = validate_claims(claims, evidence)
    assert claims[0].support == "SUPPORTED"        # strong token overlap with P1
    assert claims[1].support == "UNSUPPORTED"       # phantom citation P99 dropped
    assert claims[1].citations == []
    assert 0.0 <= metrics["citation_precision"] <= 1.0
    assert metrics["evidence_total"] == 2
    assert metrics["support_histogram"]["SUPPORTED"] >= 1


def test_citation_precision_recall_bounds():
    evidence = [{"parent_passage_id": "P1", "text": "gene expression regulation pathway"}]
    claims = [Claim(text="gene expression regulation pathway", citations=["P1"])]
    claims, metrics = validate_claims(claims, evidence)
    assert metrics["citation_precision"] == pytest.approx(1.0)
    assert metrics["citation_recall"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# /chat offline: valid ChatResponse + zero live LLM calls
# --------------------------------------------------------------------------- #
def test_chat_offline_zero_live_calls():
    from fastapi.testclient import TestClient
    from api.app import app
    from common.llm import llm_usage
    from common.config import SETTINGS

    assert SETTINGS.llm_enabled is False          # master switch off for the test

    with TestClient(app) as client:
        resp = client.post("/chat", json={
            "question": "What is the role of BRCA1 in DNA repair?",
            "recipe": "hybrid_rerank", "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        # Contract shape sanity.
        assert data["question"]
        assert isinstance(data["answer"], str) and data["answer"]
        assert data["evidence_status"] in ("sufficient", "weak", "insufficient")
        assert isinstance(data["claims"], list)
        assert isinstance(data["citations"], list)
        assert isinstance(data["grounding"], list) and len(data["grounding"]) >= 1
        # Offline: extractive fallback, no live model call.
        assert data["llm_used"] is False
        assert data["llm_source"] in ("disabled", "budget")

    # The critical budget guarantee.
    assert llm_usage()["live_calls"] == 0


def test_health_and_overview_offline():
    from fastapi.testclient import TestClient
    from api.app import app
    from common.llm import llm_usage

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "ok"

        o = client.get("/overview")
        assert o.status_code == 200
        body = o.json()
        assert body["corpus_passages"] > 0
        assert body["questions"] > 0

    assert llm_usage()["live_calls"] == 0
