"""Experiment-lab tests: frozen benchmark, config validation, run + paired compare.

Runs use a temporary run store and 5 dev questions, retrieval-only (no LLM).
"""
from __future__ import annotations

import os

import pytest

os.environ["LLM_ENABLED"] = "false"

from api.contract import RetrievalConfig
from common import paths
from experiments import lab, registry
from experiments.config import changes, to_recipe
from ingestion.loader import load_dev_corpus_ids, load_question_set


def test_benchmark_is_disjoint_and_dev_corpus_covers_dev_gold():
    dev, locked = load_question_set("dev_300"), load_question_set("locked_100")
    assert (len(dev), len(locked)) == (300, 100)
    assert not set(dev["canonical_question_id"]) & set(locked["canonical_question_id"])
    assert not set(dev["question_sha256"]) & set(locked["question_sha256"])
    dev_gold = {g for ids in dev["gold_ids"] for g in ids}
    assert dev_gold <= load_dev_corpus_ids()


@pytest.mark.parametrize("cfg, corpus", [
    (RetrievalConfig(chunking="semantic_nb04"), "full"),              # no MedCPT vectors for NB04
    (RetrievalConfig(chunking="semantic_nb04", embedding="ada002_azure", index="hnsw"), "full"),
    (RetrievalConfig(chunking="semantic_nb04", embedding="ada002_azure"), "dev"),
    (RetrievalConfig(embedding="ada002_azure"), "full"),               # no ada vectors for passages
    (RetrievalConfig(chunking="no_such_strategy"), "full"),
    (RetrievalConfig(channels=[]), "full"),
    (RetrievalConfig(channels=["keywords"]), "full"),
])
def test_invalid_flows_are_rejected(cfg, corpus):
    with pytest.raises(ValueError):
        to_recipe(cfg, corpus)


def test_changes_ignore_settings_that_do_not_matter():
    a = RetrievalConfig(index="flat", ef_search=64)
    b = RetrievalConfig(index="flat", ef_search=128)                    # efSearch unused by flat
    assert changes(a, b) == []
    c = RetrievalConfig(index="hnsw", ef_search=128)
    assert {f for f, _, _ in changes(a, c)} == {"index", "ef_search"}


def test_llm_reranker_not_allowed_in_batch_runs():
    with pytest.raises(ValueError):
        lab.submit(RetrievalConfig(reranker="llm_reranker"), wait=True)


def test_run_and_paired_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DB", tmp_path / "runs.db")
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    five = load_question_set("dev_300").head(5)
    monkeypatch.setattr(lab, "load_question_set", lambda name: five)

    a = lab.submit(RetrievalConfig(channels=["lexical"]), "dev_300", "dev", "bm25", wait=True)
    b = lab.submit(RetrievalConfig(), "dev_300", "dev", "hybrid", wait=True)
    ra, rb = registry.get(a), registry.get(b)
    assert ra["status"] == rb["status"] == "done", (ra["error"], rb["error"])

    keys = {m["key"] for m in ra["metrics"]}
    assert {"hit@10", "recall@10", "ndcg@10", "latency_p50"} <= keys
    for m in ra["metrics"]:
        if m.get("ci_low") is not None:
            assert m["ci_low"] <= m["value"] <= m["ci_high"]

    cmp = lab.compare(ra, rb)
    assert cmp["comparable"]
    assert [c["field"] for c in cmp["changes"]] == ["channels"]
    assert cmp["wins"] + cmp["losses"] + cmp["ties"] == 5
    assert all(m["ci_low"] <= m["delta"] <= m["ci_high"] for m in cmp["metrics"])
