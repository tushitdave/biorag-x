"""BioRAG-X FastAPI application.

Serves the exact api.contract response types to the React frontend. On startup it
warms the BM25 + dense indexes so the first user request is fast (the dense index
build + first query embedding are otherwise ~30s). CORS is enabled for the Vite
dev server (http://localhost:5173).

Every LLM-touching path goes through generation.generator -> common.llm, so the
master switch + budget + disk cache are always enforced. With LLM disabled (the
default) all endpoints work fully offline and make zero live LLM calls.
"""
from __future__ import annotations

import platform
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from common import paths
from common.config import SETTINGS, load_recipe, list_recipes
from common.llm import llm_usage, budget_remaining

from ingestion.loader import (
    corpus_counts, gold_ids_for_question, load_questions, load_chunks,
    load_dev_corpus_ids, load_question_set,
)

from indexes.bm25 import get_bm25
from indexes.dense import get_dense, encode_medcpt_query
from retrieval.hybrid import retrieve

from generation.generator import generate_answer
from experiments.run_experiment import run_recipe, run_leaderboard
from experiments import lab, registry
from experiments.config import EMBED_MODELS, embedding_for, to_recipe
from chunking import registry as chunk_registry
from retrieval.reranker import _medcpt_cross_encoder
from evaluation.retrieval_metrics import per_query_metrics

from api.capabilities import build_capabilities
from api import architecture
from api.mermaid import build_workflow_mermaid, build_system_mermaid
from api.contract import (
    HealthInfo, Capabilities, HeadlineMetrics, ExperimentDNA, MetricValue,
    PipelineView, PipelineStage, WorkflowDiagram,
    RetrievalRequest, RetrievalResponse, RetrievedPassage,
    ChatRequest, ChatResponse,
    EvaluationReport, LeaderboardRow, ExplainableChart, ChartSeries, OracleComparison,
    DiagnosticsReport, FailureBucket, AblationRow,
    SystemDesign, VersionInfo,
    RetrievalConfig, LabInfo, QuestionSetInfo, CorpusInfo, RunCreate, RunSummary,
    RunDetail, CompareResult, AskRequest, AskCompareResponse, ArchitectureView,
)
from retrieval.compare import compare as compare_combinations


# --------------------------------------------------------------------------- #
# Lifespan: warm the default flow's indexes at startup.
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.fail_interrupted()          # runs cannot survive a server restart
    try:
        recipe = to_recipe(RetrievalConfig())
        strategy = recipe["chunking"]["method"]
        model = recipe["embedding"]["model"]
        get_bm25(strategy, recipe["corpus"])                           # build + cache BM25
        get_dense(strategy, recipe["ann"]["index"], model, recipe["corpus"])
        if model == "medcpt":
            encode_medcpt_query("warmup")   # load the query encoder now, not on first chat
    except Exception as e:             # pragma: no cover - warmup is best-effort
        print(f"[startup] index warmup skipped: {e}")
    threading.Thread(target=_warm_chunk_strategies, daemon=True).start()
    yield


def _warm_chunk_strategies() -> None:
    """Load the judge and each strategy's indexes in the background - including
    strategies that finish building while the server runs - so a per-question
    comparison does not pay for loading them. Best-effort."""
    loaded: set[str] = set()
    try:
        _medcpt_cross_encoder()
        from retrieval import structured
        if structured.is_ready("graph"):
            structured.get_graph()
        if structured.is_ready("pageindex"):
            structured.get_tree()
        if structured.is_ready("synonyms"):
            structured.get_synonyms()
        while True:
            for s in chunk_registry.BUILT + ["semantic_nb04"]:
                if s not in loaded and chunk_registry.is_ready(s):
                    get_bm25(s, "full")
                    get_dense(s, "flat", EMBED_MODELS[embedding_for(s)], "full")
                    loaded.add(s)
            if all(s in loaded for s in chunk_registry.BUILT):
                return
            time.sleep(60)
    except Exception as e:             # pragma: no cover
        print(f"[startup] strategy warmup stopped: {e}")


app = FastAPI(title="BioRAG-X API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _experiment_dna(recipe: dict) -> ExperimentDNA:
    ch = recipe.get("chunking", {})
    emb = recipe.get("embedding", {})
    ann = recipe.get("ann", {})
    retr = recipe.get("retrieval", {})
    fus = recipe.get("fusion", {})
    rr = recipe.get("reranker", {})
    ag = recipe.get("agent", {})
    ev = recipe.get("evidence", {})
    gen = recipe.get("generation", {})
    cit = recipe.get("citation", {})
    return ExperimentDNA(
        experiment_id=str(recipe.get("experiment_id", "EXP")),
        chunking=str(ch.get("method", "semantic")),
        embedding=str(emb.get("model", "none")),
        ann=str(ann.get("index", "none")),
        retrieval="+".join(retr.get("channels", [])) or "none",
        fusion=str(fus.get("method", "none")),
        reranker=(rr.get("method", "none") if rr.get("enabled", False) else "none"),
        agent=str(ag.get("mode", "manual")),
        evidence=str(ev.get("selector", "topk")),
        generator=str(gen.get("llm", "none")),
        citation=str(cit.get("validation", "strict")),
    )


def _gold_for_question(question: str) -> set[str]:
    """Look up gold parent ids for a question by exact text match (best-effort)."""
    try:
        q = load_questions()
        for col in ("question", "question_text", "body", "text"):
            if col in q.columns:
                hit = q[q[col].astype(str) == question]
                if len(hit):
                    return set(gold_ids_for_question(hit.iloc[0]))
    except Exception:
        pass
    return set()


def _request_recipe(name: str, config: RetrievalConfig | None) -> dict:
    """The flow from the frontend when given, else the named recipe."""
    if config is not None:
        try:
            return to_recipe(config)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        return load_recipe(name)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Unknown recipe: {name}")


def _read_csv(path: Path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthInfo)
def health() -> HealthInfo:
    return HealthInfo(
        status="ok",
        artifacts=paths.exists_report(),
        settings=SETTINGS.to_public_dict(),
        llm_usage=llm_usage(),
    )


@app.get("/capabilities", response_model=Capabilities)
def capabilities() -> Capabilities:
    return build_capabilities()


@app.get("/architecture", response_model=ArchitectureView)
def architecture_view() -> ArchitectureView:
    """Process-flow and system-design diagrams, generated from the live system."""
    return ArchitectureView(process=architecture.process_diagrams(),
                            system=architecture.system_diagrams(),
                            legend=architecture.LEGEND)


@app.get("/overview", response_model=HeadlineMetrics)
def overview() -> HeadlineMetrics:
    counts = corpus_counts()
    recipe = load_recipe("hybrid_rerank")
    dna = _experiment_dna(recipe)

    metrics = [
        MetricValue(key="corpus_passages", label="Corpus passages",
                    value=float(counts["passages"]), higher_is_better=True,
                    definition="Canonical biomedical passages available for retrieval."),
        MetricValue(key="questions", label="Benchmark questions",
                    value=float(counts["questions"]),
                    definition="Questions with gold evidence for evaluation."),
        MetricValue(key="gold_relationships", label="Gold relationships",
                    value=float(counts["gold_relationships"]),
                    definition="Question-to-passage relevance judgements."),
        MetricValue(key="llm_budget_remaining", label="LLM budget remaining",
                    value=float(budget_remaining()), unit="calls",
                    definition="Live LLM calls still allowed this process (cap 12).",
                    source="live"),
    ]
    return HeadlineMetrics(
        experiment=dna,
        metrics=metrics,
        corpus_passages=counts["passages"],
        questions=counts["questions"],
        gold_relationships=counts["gold_relationships"],
    )


@app.get("/pipeline", response_model=PipelineView)
def pipeline() -> PipelineView:
    recipe = load_recipe("hybrid_rerank")
    retr = recipe.get("retrieval", {})
    rr = recipe.get("reranker", {})
    ev = recipe.get("evidence", {})
    strategy = recipe.get("chunking", {}).get("method", "semantic")
    model = recipe.get("embedding", {}).get("model", "text-embedding-ada-002")

    stages = [
        PipelineStage(
            id="ingest", name="Ingestion & Chunking", layer="data",
            summary=f"Retrieval units from the '{strategy}' strategy.",
            metrics=[MetricValue(key="strategy_chunks", label="Retrieval units",
                                 value=float(len(load_chunks(strategy))),
                                 definition="Units (passages or chunks) indexed for retrieval.")],
            config={"method": strategy},
        ),
        PipelineStage(
            id="represent", name="Representation", layer="representation",
            summary=f"BM25 lexical + {model} dense vectors (precomputed).",
            config={"embedding": model},
        ),
        PipelineStage(
            id="retrieve", name="Hybrid Retrieval", layer="retrieval",
            summary="Runs configured channels and pulls top candidates each.",
            metrics=[MetricValue(key="channel_top_k", label="Per-channel top-k",
                                 value=float(retr.get("top_k", 40)))],
            config={"channels": retr.get("channels", [])},
        ),
        PipelineStage(
            id="fuse", name="RRF Fusion", layer="retrieval",
            summary="Rank-based reciprocal rank fusion across channels.",
            config={"k": recipe.get("fusion", {}).get("k", 60)},
        ),
        PipelineStage(
            id="rerank", name="Reranker", layer="retrieval",
            summary=(f"Reranks the fused pool with '{rr.get('method')}'." if rr.get("enabled")
                     else "Off: the fused RRF order is kept."),
            config={"method": rr.get("method", "none"),
                    "enabled": rr.get("enabled", False)},
        ),
        PipelineStage(
            id="evidence", name="Evidence Selection", layer="retrieval",
            summary="MMR selection for coverage + diversity.",
            metrics=[MetricValue(key="evidence_top_k", label="Evidence set size",
                                 value=float(ev.get("top_k", 5)))],
            config={"selector": ev.get("selector", "topk")},
        ),
        PipelineStage(
            id="generate", name="Grounded Generation", layer="answering",
            summary="LLM (if enabled) or deterministic extractive fallback.",
            config={"llm_enabled": SETTINGS.llm_enabled},
        ),
        PipelineStage(
            id="validate", name="Citation Validation", layer="answering",
            summary="Checks each citation against the evidence set.",
            config={"validation": recipe.get("citation", {}).get("validation", "strict")},
        ),
    ]
    order = [s.id for s in stages]
    workflow = WorkflowDiagram(mermaid=build_workflow_mermaid(order),
                               stage_ids_in_order=order)
    return PipelineView(stages=stages, workflow=workflow)


@app.post("/retrieval", response_model=RetrievalResponse)
def retrieval(req: RetrievalRequest) -> RetrievalResponse:
    recipe = _request_recipe(req.recipe, req.config)
    result = retrieve(req.question, recipe, top_k=req.top_k,
                      reranker_method=None if req.config else req.reranker)
    gold = _gold_for_question(req.question)

    passages = []
    for p in result.get("passages", []):
        pid = str(p.get("parent_passage_id") or p.get("passage_id"))
        passages.append(RetrievedPassage(
            passage_id=pid,
            text=p.get("text", ""),
            bm25_score=p.get("bm25_score"),
            dense_score=p.get("dense_score"),
            rrf_score=p.get("rrf_score"),
            rerank_score=p.get("rerank_score"),
            final_rank=int(p.get("final_rank", 0)),
            is_gold=(pid in gold) if gold else None,
        ))

    metrics = []
    if gold:
        ranked = [str(p.get("parent_passage_id")) for p in result.get("reranked", [])]
        pq = per_query_metrics(ranked, gold)
        for k, v in pq.items():
            metrics.append(MetricValue(key=k, label=k, value=v, source="live"))

    return RetrievalResponse(
        question=req.question, recipe=req.recipe,
        passages=passages, metrics=metrics, trace=result.get("trace", {}),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    recipe = _request_recipe(req.recipe, req.config)
    result = retrieve(req.question, recipe, top_k=req.top_k,
                      reranker_method=None if req.config else req.reranker)
    top_k = req.config.evidence_k if req.config else req.top_k
    return generate_answer(req.question, recipe=req.recipe, top_k=top_k,
                           retrieval_result=result)


# --------------------------------------------------------------------------- #
# Ask: the per-question trust report. Compare combinations first (retrieval + local
# judge, no LLM), then answer once from the chosen combination (1 LLM call).
# --------------------------------------------------------------------------- #
@app.post("/ask/compare", response_model=AskCompareResponse)
def ask_compare(req: AskRequest) -> AskCompareResponse:
    try:
        to_recipe(req.config)                       # reject an impossible flow up front
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AskCompareResponse(**compare_combinations(req.question, req.config))


@app.post("/ask/answer", response_model=ChatResponse)
def ask_answer(req: AskRequest) -> ChatResponse:
    try:
        recipe = to_recipe(req.config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = retrieve(req.question, recipe)
    return generate_answer(req.question, top_k=req.config.evidence_k, retrieval_result=result)


# --------------------------------------------------------------------------- #
# Experiment lab: frozen question sets, runs, paired comparisons.
# --------------------------------------------------------------------------- #
@app.get("/lab/info", response_model=LabInfo)
def lab_info() -> LabInfo:
    dev = len(load_question_set("dev_300"))
    locked = len(load_question_set("locked_100"))
    return LabInfo(
        default_config=RetrievalConfig(),
        question_sets=[
            QuestionSetInfo(id="dev_300", label=f"Dev ({dev} questions)", n=dev,
                            locked=False, corpus="dev"),
            QuestionSetInfo(id="locked_100", label=f"Locked ({locked} questions)", n=locked,
                            locked=True, corpus="full"),
        ],
        corpora=[
            CorpusInfo(id="dev", label="Dev corpus", passages=len(load_dev_corpus_ids())),
            CorpusInfo(id="full", label="Full corpus", passages=len(load_chunks("passage"))),
        ],
    )


@app.post("/runs", response_model=RunSummary)
def create_run(req: RunCreate) -> RunSummary:
    try:
        rid = lab.submit(req.config, req.question_set, req.corpus, req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RunSummary(**lab.summary(registry.get(rid)))


@app.get("/runs", response_model=list[RunSummary])
def list_runs() -> list[RunSummary]:
    return [RunSummary(**lab.summary(r)) for r in registry.all_runs()]


@app.get("/runs/compare", response_model=CompareResult)
def compare_runs(a: str, b: str) -> CompareResult:
    ra, rb = registry.get(a), registry.get(b)
    if not ra or not rb:
        raise HTTPException(status_code=404, detail="Unknown run id")
    if ra["status"] != "done" or rb["status"] != "done":
        raise HTTPException(status_code=409, detail="Both runs must be finished")
    return CompareResult(**lab.compare(ra, rb))


@app.get("/runs/{rid}", response_model=RunDetail)
def get_run(rid: str) -> RunDetail:
    run = registry.get(rid)
    if not run:
        raise HTTPException(status_code=404, detail=f"Unknown run: {rid}")
    return RunDetail(**lab.detail(run))


@app.get("/evaluation", response_model=EvaluationReport)
def evaluation() -> EvaluationReport:
    # Retrieval-only leaderboard (0 LLM calls) on the reproducible sample.
    results = run_leaderboard(["hybrid_rerank", "baseline_bm25"],
                              n=SETTINGS.benchmark_size, top_k=10)
    leaderboard = []
    for rank, r in enumerate(results, start=1):
        leaderboard.append(LeaderboardRow(
            rank=rank,
            configuration=r["experiment_id"],
            experiment_dna=_experiment_dna(load_recipe(r["recipe"])),
            metrics={k: float(v) for k, v in r["metrics"].items()},
        ))

    top = results[0]["metrics"] if results else {}
    metrics = [
        MetricValue(key="recall@5", label="Recall@5",
                    value=float(top.get("recall@5", 0.0)), unit="",
                    definition="Fraction of gold passages retrieved in top 5.", source="live"),
        MetricValue(key="mrr@10", label="MRR@10",
                    value=float(top.get("mrr@10", 0.0)),
                    definition="Mean reciprocal rank of the first gold hit.", source="live"),
        MetricValue(key="ndcg@10", label="nDCG@10",
                    value=float(top.get("ndcg@10", 0.0)),
                    definition="Rank-weighted retrieval quality.", source="live"),
    ]

    charts = _evaluation_charts(results)

    # Oracle framing (retrieval nDCG as the achievable ceiling proxy on the sample).
    oracle = None
    if results:
        actual = float(results[0]["metrics"].get("ndcg@10", 0.0)) * 100
        base = float(results[-1]["metrics"].get("ndcg@10", 0.0)) * 100
        oracle = OracleComparison(
            no_retrieval=round(base, 1), actual=round(actual, 1), oracle=100.0,
            generation_gain=round(actual - base, 1),
            retrieval_gap=round(100.0 - actual, 1), unit="%",
        )

    return EvaluationReport(
        benchmark_size=SETTINGS.benchmark_size,
        leaderboard=leaderboard,
        metrics=metrics,
        charts=charts,
        oracle=oracle,
        notes="Retrieval-only leaderboard (0 live LLM calls). "
              "Chart data sourced from notebook artifacts when present.",
    )


def _evaluation_charts(results: list[dict]) -> list[ExplainableChart]:
    charts: list[ExplainableChart] = []

    # 1) Retrieval leaderboard from live sample runs.
    if results:
        names = [r["experiment_id"] for r in results]
        charts.append(ExplainableChart(
            id="retrieval_leaderboard", title="Retrieval quality by configuration",
            kind="grouped_bar",
            series=[
                ChartSeries(name="Recall@5", x=names,
                            y=[float(r["metrics"].get("recall@5", 0.0)) for r in results]),
                ChartSeries(name="MRR@10", x=names,
                            y=[float(r["metrics"].get("mrr@10", 0.0)) for r in results]),
                ChartSeries(name="nDCG@10", x=names,
                            y=[float(r["metrics"].get("ndcg@10", 0.0)) for r in results]),
            ],
            x_label="Configuration", y_label="Score (0-1)",
            explanation="Live retrieval metrics on the benchmark sample for each recipe.",
            takeaway="Hybrid + rerank vs BM25 baseline on the same questions.",
        ))

    # 2) Chunk quality from notebook artifact (real CSV).
    cq = _read_csv(paths.NB_ARTIFACTS_DIR / "05_agentic_chunking" / "chunk_quality_scores.csv")
    if cq is not None and "strategy" in cq.columns and "intrinsic_score" in cq.columns:
        charts.append(ExplainableChart(
            id="chunk_quality", title="Intrinsic chunk quality by strategy",
            kind="bar",
            series=[ChartSeries(name="Intrinsic score",
                                x=[str(s) for s in cq["strategy"].tolist()],
                                y=[float(v) for v in cq["intrinsic_score"].tolist()])],
            x_label="Chunking strategy", y_label="Intrinsic score",
            explanation="Composite chunk-quality score (boundary, integrity, redundancy) "
                        "from NB05.",
            takeaway="Semantic/recursive chunks score highest on intrinsic quality.",
        ))

    # 3) Index economics from notebook artifact (real CSV).
    ie = _read_csv(paths.NB_ARTIFACTS_DIR / "04_advanced_chunking" / "index_economics.csv")
    if ie is not None and "strategy" in ie.columns:
        ycol = "approx_index_bytes_f32_1536d" if "approx_index_bytes_f32_1536d" in ie.columns else ie.columns[-1]
        charts.append(ExplainableChart(
            id="index_economics", title="Index memory cost by strategy",
            kind="bar",
            series=[ChartSeries(name="Index bytes (f32, 1536d)",
                                x=[str(s) for s in ie["strategy"].tolist()],
                                y=[float(v) / 1e6 for v in ie[ycol].tolist()])],
            x_label="Chunking strategy", y_label="Index size (MB)",
            explanation="Approximate in-RAM index size per chunking strategy from NB04.",
            takeaway="Proposition chunking explodes index size ~16x vs semantic.",
        ))

    return charts


@app.get("/diagnostics", response_model=DiagnosticsReport)
def diagnostics() -> DiagnosticsReport:
    # Ablation: hybrid_rerank vs baseline_bm25 on the sample (retrieval-only, 0 LLM).
    results = run_leaderboard(["hybrid_rerank", "baseline_bm25"],
                              n=SETTINGS.benchmark_size, top_k=10)
    by_id = {r["experiment_id"]: r["metrics"] for r in results}
    hybrid = next((r["metrics"] for r in results if "hybrid" in r["recipe"]), {})
    baseline = next((r["metrics"] for r in results if "baseline" in r["recipe"]), {})

    ablations = []
    for mk in ("recall@5", "mrr@10", "ndcg@10"):
        w = float(hybrid.get(mk, 0.0))
        wo = float(baseline.get(mk, 0.0))
        ablations.append(AblationRow(
            disabled_component="dense+fusion+rerank", metric_key=mk,
            with_value=round(w, 4), without_value=round(wo, 4),
            delta=round(w - wo, 4),
        ))

    # Failure taxonomy derived from per-query misses on the sample.
    hybrid_full = next((r for r in results if "hybrid" in r["recipe"]), None)
    total = len(hybrid_full["per_query"]) if hybrid_full else 0
    misses = sum(1 for pq in (hybrid_full["per_query"] if hybrid_full else [])
                 if pq.get("hit@5", 0.0) == 0.0)
    hits = total - misses
    failures = [
        FailureBucket(code="F01", label="Retrieval miss (no gold in top 5)",
                      count=misses, pct=round((misses / total * 100) if total else 0.0, 1)),
        FailureBucket(code="F00", label="Retrieval hit", count=hits,
                      pct=round((hits / total * 100) if total else 0.0, 1)),
    ]

    charts = [ExplainableChart(
        id="ablation_delta", title="Hybrid+rerank uplift vs BM25 baseline",
        kind="bar",
        series=[ChartSeries(name="Delta",
                            x=[a.metric_key for a in ablations],
                            y=[a.delta for a in ablations])],
        x_label="Metric", y_label="Improvement (0-1)",
        explanation="Difference between the full hybrid pipeline and the BM25-only baseline.",
        takeaway="Positive deltas show where dense+fusion+rerank help on this sample.",
    )]

    return DiagnosticsReport(
        failures=failures, ablations=ablations, charts=charts,
        notes="Ablation + failure buckets computed retrieval-only (0 live LLM calls).",
    )


@app.get("/system", response_model=SystemDesign)
def system() -> SystemDesign:
    import fastapi
    import pydantic
    import numpy
    versions = [
        VersionInfo(component="python", version=platform.python_version()),
        VersionInfo(component="fastapi", version=fastapi.__version__),
        VersionInfo(component="pydantic", version=pydantic.VERSION),
        VersionInfo(component="pandas", version=pd.__version__),
        VersionInfo(component="numpy", version=numpy.__version__),
    ]
    try:
        import rank_bm25  # noqa
        versions.append(VersionInfo(component="rank-bm25", version=getattr(rank_bm25, "__version__", "0.2.2")))
    except Exception:
        pass

    observability = [
        MetricValue(key="llm_live_calls", label="LLM live calls",
                    value=float(llm_usage()["live_calls"]), unit="calls",
                    higher_is_better=False,
                    definition="Live model calls made this process (budget-capped at 12).",
                    source="live"),
        MetricValue(key="llm_cache_hits", label="LLM cache hits",
                    value=float(llm_usage()["cache_hits"]), unit="calls", source="live",
                    definition="Answered from disk cache without a live call."),
        MetricValue(key="llm_budget_remaining", label="LLM budget remaining",
                    value=float(budget_remaining()), unit="calls", source="live",
                    definition="Remaining live-call budget."),
        MetricValue(key="recipes", label="Available recipes",
                    value=float(len(list_recipes())), source="live",
                    definition="Experiment recipes discoverable in configs/."),
    ]

    recipe = load_recipe("hybrid_rerank")
    details = {
        "guard": {"llm_enabled": SETTINGS.llm_enabled,
                  "budget": SETTINGS.llm_call_budget,
                  "cache": "disk (sha256-keyed)"},
        "retr": {"channels": recipe.get("retrieval", {}).get("channels", []),
                 "embedding": recipe.get("embedding", {}).get("model"),
                 "index": recipe.get("ann", {}).get("index")},
        "artifacts": paths.exists_report(),
    }

    return SystemDesign(
        mermaid=build_system_mermaid(),
        versions=versions,
        observability=observability,
        component_details=details,
    )
