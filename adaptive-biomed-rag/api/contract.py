"""Shared API data contract for BioRAG-X.

These Pydantic models are the single source of truth for the request/response
shapes exchanged between the FastAPI backend and the React frontend. Both tracks
build against these types. Keep them stable; extend rather than rename.

Page -> primary payloads:
  Overview      -> HealthInfo, HeadlineMetrics, ExperimentDNA
  Pipeline      -> PipelineView (stages) + WorkflowDiagram (mermaid)
  Chat          -> ChatResponse (answer + claims + grounding + provenance)
  Evaluation    -> EvaluationReport (metrics + leaderboard + oracle) with explanations
  System Design -> SystemDesign (mermaid) + Versions + Observability
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Capabilities — the selectable options the frontend exposes per layer.
# Each option carries an evidence-based note + cost/quality hints so the UI can
# render an explainable dropdown. `available` reflects whether the backend can
# actually run it in this environment (e.g., Ollama installed, Azure configured).
# --------------------------------------------------------------------------- #
class Option(BaseModel):
    id: str
    label: str
    note: str = ""                       # plain-language / evidence note
    cost: str = "low"                    # low | medium | high (latency/$)
    quality: str = "medium"              # low | medium | high (expected)
    default: bool = False
    available: bool = True
    uses_llm: bool = False
    planned: bool = False                # in the research design, not built yet (never selectable)
    status: str = ""                     # index build state: ready | building | queued | failed
    progress: Optional[float] = None     # 0..1 while building


class LayerOptions(BaseModel):
    layer: str                           # chunking|embedding|index|retrieval|reranker|generation
    label: str
    multi_select: bool = False           # e.g. retrieval channels can be multi
    execution_modes: list[str] = Field(default_factory=list)  # Manual/Adaptive/Agentic where relevant
    options: list[Option]


class Capabilities(BaseModel):
    """Full set of user-selectable choices, grounded in the research."""
    layers: list[LayerOptions]


# --------------------------------------------------------------------------- #
# Common
# --------------------------------------------------------------------------- #
class MetricValue(BaseModel):
    """A single metric with an explanation, for click-to-explain UI drawers."""
    key: str
    label: str
    value: float
    unit: str = ""                       # "", "%", "ms", "USD", "/5"
    higher_is_better: bool = True
    definition: str = ""                 # plain-language meaning
    baseline_delta: Optional[float] = None   # vs baseline config
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    source: str = "artifact"             # artifact | live | mock


class HealthInfo(BaseModel):
    status: str = "ok"
    artifacts: dict[str, bool]
    settings: dict
    llm_usage: dict


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
class ExperimentDNA(BaseModel):
    experiment_id: str
    chunking: str
    embedding: str
    ann: str
    retrieval: str
    fusion: str
    reranker: str
    agent: str
    evidence: str
    generator: str
    citation: str


class HeadlineMetrics(BaseModel):
    experiment: ExperimentDNA
    metrics: list[MetricValue]
    corpus_passages: int
    questions: int
    gold_relationships: int


# --------------------------------------------------------------------------- #
# Pipeline (Workflow)
# --------------------------------------------------------------------------- #
class PipelineStage(BaseModel):
    id: str
    name: str
    layer: str                           # data|representation|retrieval|answering|research
    status: str = "ready"                # ready|active|off
    summary: str
    metrics: list[MetricValue] = Field(default_factory=list)
    config: dict = Field(default_factory=dict)


class WorkflowDiagram(BaseModel):
    mermaid: str                         # tree/block diagram source
    stage_ids_in_order: list[str]


class PipelineView(BaseModel):
    stages: list[PipelineStage]
    workflow: WorkflowDiagram


# --------------------------------------------------------------------------- #
# Chat / Answering
# --------------------------------------------------------------------------- #
class ProvenanceStep(BaseModel):
    level: str                           # claim|citation|passage|chunk|section|document|source
    label: str
    ref_id: Optional[str] = None


class Citation(BaseModel):
    passage_id: str
    text: str
    support: str = "SUPPORTED"           # SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED
    scores: dict = Field(default_factory=dict)   # bm25/dense/rrf/rerank/final_rank
    provenance: list[ProvenanceStep] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)   # passage_ids
    support: str = "SUPPORTED"


class RetrievalConfig(BaseModel):
    """One retrieval flow: the frontend's pipeline choices. Used by Chat and runs."""
    chunking: str = "passage"            # any id in chunking/registry.py STRATEGIES
    embedding: str = "medcpt"            # medcpt | ada002_azure
    index: str = "flat"                  # flat (exact) | hnsw | ivf
    ef_search: int = Field(64, ge=8, le=1024)       # HNSW search breadth
    nprobe: int = Field(8, ge=1, le=512)            # IVF clusters searched
    query: str = "as_is"                 # as_is | synonyms | router (router picks channels)
    channels: list[str] = Field(default_factory=lambda: ["lexical", "dense"])   # + graph | pageindex
    depth: int = Field(40, ge=5, le=200)            # candidates per search channel
    rrf_k: int = Field(60, ge=1, le=200)
    reranker: str = "none"               # none | medcpt_ce | lexical_overlap | cross_encoder | colbert | llm_reranker
    rerank_candidates: int = Field(50, ge=5, le=200)
    evidence: str = "mmr"                # mmr | topk
    evidence_k: int = Field(5, ge=1, le=20)


class ChatRequest(BaseModel):
    question: str
    recipe: str = "hybrid_rerank"
    top_k: int = 5
    reranker: Optional[str] = None
    config: Optional[RetrievalConfig] = None   # the flow; overrides the recipe when set


class ChatResponse(BaseModel):
    question: str
    answer: str
    evidence_status: str                 # sufficient|weak|insufficient
    claims: list[Claim]
    citations: list[Citation]
    grounding: list[MetricValue]         # coverage, citation support, faithfulness, ...
    llm_used: bool
    llm_source: str                      # cache|live|disabled|budget|unusable
    retrieval_trace: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Retrieval (used inside Pipeline + Chat inspection)
# --------------------------------------------------------------------------- #
class RetrievedPassage(BaseModel):
    passage_id: str
    text: str
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_rank: int
    is_gold: Optional[bool] = None


class RetrievalRequest(BaseModel):
    question: str
    recipe: str = "hybrid_rerank"
    top_k: int = 10
    reranker: Optional[str] = None
    config: Optional[RetrievalConfig] = None


class RetrievalResponse(BaseModel):
    question: str
    recipe: str
    passages: list[RetrievedPassage]
    metrics: list[MetricValue]           # per-query recall/mrr if gold known
    trace: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class LeaderboardRow(BaseModel):
    rank: int
    configuration: str
    experiment_dna: Optional[ExperimentDNA] = None
    metrics: dict[str, float]            # metric_key -> value


class OracleComparison(BaseModel):
    no_retrieval: float
    actual: float
    oracle: float
    generation_gain: float               # actual - no_retrieval
    retrieval_gap: float                 # oracle - actual
    unit: str = "%"


class ChartSeries(BaseModel):
    name: str
    x: list[str]
    y: list[float]


class ExplainableChart(BaseModel):
    id: str
    title: str
    kind: str                            # bar|grouped_bar|radar|scatter|line
    series: list[ChartSeries]
    x_label: str = ""
    y_label: str = ""
    explanation: str                     # what/why this chart shows
    takeaway: str = ""                   # the one-line finding


class EvaluationReport(BaseModel):
    benchmark_size: int
    leaderboard: list[LeaderboardRow]
    metrics: list[MetricValue]
    charts: list[ExplainableChart]
    oracle: Optional[OracleComparison] = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
class FailureBucket(BaseModel):
    code: str                            # F01..F20
    label: str
    count: int
    pct: float


class AblationRow(BaseModel):
    disabled_component: str
    metric_key: str
    with_value: float
    without_value: float
    delta: float


class DiagnosticsReport(BaseModel):
    failures: list[FailureBucket]
    ablations: list[AblationRow]
    charts: list[ExplainableChart]
    notes: str = ""


# --------------------------------------------------------------------------- #
# System Design
# --------------------------------------------------------------------------- #
class VersionInfo(BaseModel):
    component: str
    version: str


class SystemDesign(BaseModel):
    mermaid: str                         # component/block tree
    versions: list[VersionInfo]
    observability: list[MetricValue]
    component_details: dict = Field(default_factory=dict)  # node_id -> details


# --------------------------------------------------------------------------- #
# Experiment lab: runs of a RetrievalConfig on a frozen question set, and
# paired comparisons between two runs.
# --------------------------------------------------------------------------- #
class QuestionSetInfo(BaseModel):
    id: str                              # dev_300 | locked_100
    label: str
    n: int
    locked: bool                         # final check only; never tune on it
    corpus: str                          # corpus its runs use: dev | full


class CorpusInfo(BaseModel):
    id: str                              # dev | full
    label: str
    passages: int


class LabInfo(BaseModel):
    default_config: RetrievalConfig
    question_sets: list[QuestionSetInfo]
    corpora: list[CorpusInfo]


class RunCreate(BaseModel):
    config: RetrievalConfig
    question_set: str = "dev_300"
    corpus: str = "dev"
    name: str = ""


class RunMetric(BaseModel):
    key: str
    label: str
    value: float
    ci_low: Optional[float] = None       # 95% bootstrap CI over questions
    ci_high: Optional[float] = None
    unit: str = ""


class RunSummary(BaseModel):
    id: str
    name: str
    created_at: str
    status: str                          # queued | running | done | failed
    question_set: str
    corpus: str
    n_questions: int
    done: int
    config: RetrievalConfig
    headline: dict[str, float] = Field(default_factory=dict)   # key metrics for tables
    duration_s: Optional[float] = None
    error: str = ""


class SliceRow(BaseModel):
    dimension: str                       # question_type | lexical_overlap | evidence_size
    value: str
    n: int
    metrics: dict[str, float]


class MissedQuestion(BaseModel):
    question: str
    question_type: str
    n_gold: int
    gold_in_top20: int


class RunDetail(BaseModel):
    run: RunSummary
    metrics: list[RunMetric]
    slices: list[SliceRow]
    missed: list[MissedQuestion]


class ConfigChange(BaseModel):
    field: str
    a: str
    b: str


class CompareMetric(BaseModel):
    key: str
    label: str
    a: float
    b: float
    delta: float                         # b - a
    ci_low: float                        # 95% paired bootstrap CI of the delta
    ci_high: float
    significant: bool                    # CI excludes 0
    higher_is_better: bool = True


class SliceDelta(BaseModel):
    dimension: str
    value: str
    n: int
    delta: dict[str, float]


class QuestionDelta(BaseModel):
    question: str
    a: float
    b: float


class CompareResult(BaseModel):
    a: RunSummary
    b: RunSummary
    comparable: bool
    reason: str = ""
    changes: list[ConfigChange]
    metrics: list[CompareMetric]
    decision_metric: str                 # metric used for wins/losses
    wins: int                            # questions where B beats A
    losses: int
    ties: int
    slices: list[SliceDelta]
    top_gains: list[QuestionDelta]
    top_losses: list[QuestionDelta]


# --------------------------------------------------------------------------- #
# Ask: the per-question trust report (compare combinations, then answer once).
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str
    config: RetrievalConfig              # the user's flow (always one of the candidates)


class EvidenceItem(BaseModel):
    passage_id: str
    chunk_id: str
    text: str                            # what the LLM would receive
    child_text: Optional[str] = None     # parent-child: the child that matched
    judge: float                         # 0..1 relevance to the question (MedCPT cross-encoder)
    scores: dict[str, float] = Field(default_factory=dict)   # bm25 / dense / graph / pageindex / rrf / rerank
    why: Optional[str] = None            # graph path or PageIndex tree path that found it
    rank: int
    is_gold: Optional[bool] = None       # only for dataset questions


class ComboResult(BaseModel):
    id: str
    group: str                           # yours | chunking | retrieval
    config: RetrievalConfig
    is_user: bool
    is_best: bool
    self_judged: bool = False            # reranked by the judge's own model: not eligible for best
    evidence_score: float                # mean judge relevance of the evidence (0..1)
    best_evidence: float
    coverage: float                      # share of question content words in the evidence
    agreement: Optional[float] = None    # BM25 vs dense top-10 overlap (hybrid only)
    latency_ms: float
    true_metrics: Optional[dict[str, float]] = None   # vs gold, dataset questions only
    evidence: list[EvidenceItem]
    note: str = ""
    error: str = ""


class AskCompareResponse(BaseModel):
    question: str
    gold_available: bool
    n_gold: int
    combos: list[ComboResult]
    best_id: str
    user_id: str
    judge: dict
    seconds: float


# --------------------------------------------------------------------------- #
# Architecture diagrams (generated from live constants and build status).
# --------------------------------------------------------------------------- #
class ArchitectureNode(BaseModel):
    title: str
    what: str
    rules: list[str] = Field(default_factory=list)   # thresholds / decision rules in force
    source: str = ""                                 # where it lives in the code
    status: str = "live"                             # live | building | planned


class ArchitectureDiagram(BaseModel):
    id: str
    title: str
    description: str
    mermaid: str
    nodes: dict[str, ArchitectureNode]


class LegendItem(BaseModel):
    label: str
    cls: str


class ArchitectureView(BaseModel):
    process: list[ArchitectureDiagram]
    system: list[ArchitectureDiagram]
    legend: list[LegendItem]
