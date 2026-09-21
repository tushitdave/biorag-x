// Mock data fallback so every page renders even when the backend is down.
// Shapes mirror api/contract.py exactly. All metrics carry definitions and all
// charts carry explanation + takeaway (explainability is a hard requirement).
import type {
  Capabilities,
  ChatResponse,
  EvaluationReport,
  HealthInfo,
  HeadlineMetrics,
  PipelineView,
  SystemDesign,
} from "./types";

const m = (
  key: string,
  label: string,
  value: number,
  unit: string,
  definition: string,
  extra: Partial<import("./types").MetricValue> = {}
): import("./types").MetricValue => ({
  key,
  label,
  value,
  unit,
  higher_is_better: true,
  definition,
  baseline_delta: null,
  ci_low: null,
  ci_high: null,
  source: "mock",
  ...extra,
});

export const mockHealth: HealthInfo = {
  status: "ok (mock)",
  artifacts: { bm25: true, dense: true, passages: true, gold: true },
  settings: { llm_enabled: false, budget: 12, recipe: "hybrid_rerank" },
  llm_usage: { calls: 0, budget: 12, remaining: 12 },
};

export const mockCapabilities: Capabilities = {
  layers: [
    {
      layer: "embedding",
      label: "Embedding model",
      multi_select: false,
      execution_modes: ["Manual"],
      options: [
        { id: "ada002_azure", label: "Azure ada-002", note: "Precomputed for the corpus; reused, so free at index time.", cost: "medium", quality: "medium", default: true, available: true, uses_llm: false },
        { id: "ollama_nomic", label: "Ollama nomic-embed-text", note: "Local, fast, free; good for on-demand query embedding.", cost: "low", quality: "medium", default: false, available: false, uses_llm: false },
        { id: "ollama_bge_m3", label: "Ollama BGE-M3", note: "Local multilingual dense; strong retrieval quality.", cost: "low", quality: "high", default: false, available: false, uses_llm: false },
        { id: "medcpt", label: "MedCPT (biomedical)", note: "Trained on 255M PubMed query-article pairs; domain-specific.", cost: "medium", quality: "high", default: false, available: false, uses_llm: false },
      ],
    },
    {
      layer: "generation",
      label: "Answer generator",
      multi_select: false,
      execution_modes: ["Manual"],
      options: [
        { id: "ollama_llama32", label: "Ollama Llama 3.2", note: "Local, fast, free; avoids the API budget entirely.", cost: "low", quality: "medium", default: true, available: false, uses_llm: true },
        { id: "ollama_qwen25", label: "Ollama Qwen2.5", note: "Local, fast; strong instruction following.", cost: "low", quality: "medium", default: false, available: false, uses_llm: true },
        { id: "gpt4o_azure", label: "Azure GPT-4o", note: "Highest quality; counts against the strict LLM budget.", cost: "high", quality: "high", default: false, available: false, uses_llm: true },
      ],
    },
    {
      layer: "reranker",
      label: "Reranker",
      multi_select: false,
      execution_modes: ["Manual", "Adaptive"],
      options: [
        { id: "none", label: "None (fusion order)", note: "Use RRF order directly; fastest.", cost: "low", quality: "low", default: false, available: true, uses_llm: false },
        { id: "lexical_overlap", label: "Lexical overlap (local)", note: "Zero-dependency local reranker; POC default.", cost: "low", quality: "medium", default: true, available: true, uses_llm: false },
        { id: "cross_encoder", label: "Cross-encoder interaction (local)", note: "Dependency-free unigram/bigram interaction proxy for local testing.", cost: "medium", quality: "medium", default: false, available: true, uses_llm: false },
        { id: "colbert", label: "ColBERT-style MaxSim (local)", note: "Dependency-free token-level MaxSim proxy for local testing.", cost: "low", quality: "medium", default: false, available: true, uses_llm: false },
        { id: "llm_reranker", label: "LLM listwise reranker", note: "Highest quality; consumes the LLM budget.", cost: "high", quality: "high", default: false, available: false, uses_llm: true },
      ],
    },
    {
      layer: "index",
      label: "Vector index",
      multi_select: false,
      execution_modes: ["Manual"],
      options: [
        { id: "flat", label: "Flat (exact)", note: "Exact cosine; correct + cheap at 40k passages. POC default.", cost: "low", quality: "high", default: true, available: true, uses_llm: false },
        { id: "hnsw", label: "HNSW", note: "Recommended RAG default: 95-99% recall, ~100x speedup, in-RAM.", cost: "medium", quality: "high", default: false, available: true, uses_llm: false },
        { id: "ivf", label: "IVF", note: "Partitioned; for larger corpora.", cost: "medium", quality: "medium", default: false, available: true, uses_llm: false },
        { id: "ivf_pq", label: "IVF-PQ", note: "Compressed; memory-constrained / huge corpora.", cost: "low", quality: "medium", default: false, available: true, uses_llm: false },
      ],
    },
    {
      layer: "retrieval",
      label: "Retrieval channels",
      multi_select: true,
      execution_modes: ["Manual", "Adaptive", "Agentic"],
      options: [
        { id: "lexical", label: "BM25 lexical", note: "Robust for exact biomedical terms/abbreviations.", cost: "low", quality: "medium", default: true, available: true, uses_llm: false },
        { id: "dense", label: "Dense (vector)", note: "Semantic matches; reuses ada-002 vectors.", cost: "low", quality: "high", default: true, available: true, uses_llm: false },
        { id: "graph", label: "Graph traversal", note: "Adds structural associations for multi-hop relational questions.", cost: "medium", quality: "medium", default: false, available: false, uses_llm: false },
        { id: "pageindex", label: "PageIndex (vectorless)", note: "Reasoning-based hierarchical retrieval.", cost: "medium", quality: "medium", default: false, available: false, uses_llm: false },
      ],
    },
    {
      layer: "graph",
      label: "Graph mode",
      multi_select: false,
      execution_modes: ["Manual", "Adaptive"],
      options: [
        { id: "off", label: "Off", note: "Vector-only retrieval.", cost: "low", quality: "low", default: true, available: true, uses_llm: false },
        { id: "1hop", label: "1-hop neighborhood", note: "Immediate related entities.", cost: "medium", quality: "medium", default: false, available: false, uses_llm: false },
        { id: "2hop", label: "2-hop constrained", note: "Bridges entities for multi-hop questions.", cost: "high", quality: "medium", default: false, available: false, uses_llm: false },
        { id: "provenance", label: "Provenance-backed", note: "Only keep graph paths supported by real passages; safest.", cost: "high", quality: "high", default: false, available: false, uses_llm: false },
      ],
    },
    {
      layer: "retrieval_combination",
      label: "Combination strategy",
      multi_select: false,
      execution_modes: ["Manual", "Adaptive"],
      options: [
        { id: "vector_only", label: "Vector only", note: "Best for broad, messy document search.", cost: "low", quality: "medium", default: true, available: true, uses_llm: false },
        { id: "graph_only", label: "Graph only", note: "Best when the answer depends on connected facts.", cost: "medium", quality: "medium", default: false, available: false, uses_llm: false },
        { id: "hybrid_both", label: "Use both (Hybrid)", note: "HybridRAG (vector + KG) outperforms either alone on biomedical multi-hop.", cost: "high", quality: "high", default: false, available: false, uses_llm: false },
      ],
    },
  ],
};

const mockDNA = {
  experiment_id: "hybrid_rerank",
  chunking: "sentence-window",
  embedding: "ada-002",
  ann: "flat",
  retrieval: "bm25+dense",
  fusion: "rrf",
  reranker: "lexical_overlap",
  agent: "off",
  evidence: "top-5",
  generator: "extractive (LLM disabled)",
  citation: "claim-level",
};

export const mockOverview: HeadlineMetrics = {
  experiment: mockDNA,
  corpus_passages: 41200,
  questions: 500,
  gold_relationships: 1834,
  metrics: [
    m("recall_at_5", "Recall@5", 0.71, "", "Fraction of questions whose gold passage appears in the top-5 retrieved results.", { baseline_delta: 0.12, ci_low: 0.67, ci_high: 0.75 }),
    m("mrr", "MRR", 0.58, "", "Mean Reciprocal Rank: how high up the first correct passage lands, averaged over questions.", { baseline_delta: 0.09 }),
    m("citation_support", "Citation support", 0.83, "", "Share of answer claims backed by at least one retrieved passage marked SUPPORTED.", { baseline_delta: 0.05 }),
    m("faithfulness", "Faithfulness", 0.79, "", "How well the answer stays grounded in cited evidence rather than inventing facts.", { ci_low: 0.74, ci_high: 0.84 }),
  ],
};

export const mockPipeline: PipelineView = {
  workflow: {
    stage_ids_in_order: ["ingest", "chunk", "embed", "retrieve", "fuse", "rerank", "select", "generate", "cite"],
    mermaid: `flowchart TD
  ingest["Ingest\\nPubMed passages"] --> chunk["Chunk\\nsentence-window"]
  chunk --> embed["Embed\\nada-002 (cached)"]
  embed --> retrieve["Retrieve\\nBM25 + Dense"]
  retrieve --> fuse["Fuse\\nRRF"]
  fuse --> rerank["Rerank\\nlexical overlap"]
  rerank --> select["Select evidence\\ntop-5"]
  select --> generate["Generate\\nextractive fallback"]
  generate --> cite["Cite + validate\\nclaim-level"]`,
  },
  stages: [
    { id: "ingest", name: "Ingest", layer: "data", status: "ready", summary: "Load biomedical passages, questions and gold relationships from cached artifacts.", config: { source: "PubMedQA-derived", passages: 41200 }, metrics: [m("passages", "Passages", 41200, "", "Total passages in the searchable corpus.")] },
    { id: "chunk", name: "Chunk", layer: "data", status: "ready", summary: "Sentence-window chunking preserves local context around each sentence.", config: { strategy: "sentence-window", window: 3 }, metrics: [] },
    { id: "embed", name: "Embed", layer: "representation", status: "ready", summary: "Reuse precomputed ada-002 vectors; nothing recomputed at query time except the query itself.", config: { model: "ada-002", dims: 1536, cached: true }, metrics: [] },
    { id: "retrieve", name: "Retrieve", layer: "retrieval", status: "active", summary: "Run BM25 lexical and dense vector search in parallel.", config: { channels: ["bm25", "dense"], top_k: 50 }, metrics: [m("recall_at_50", "Recall@50", 0.92, "", "Fraction of questions whose gold passage is anywhere in the top-50 candidate pool.")] },
    { id: "fuse", name: "Fuse", layer: "retrieval", status: "active", summary: "Reciprocal Rank Fusion blends the two ranked lists into one.", config: { method: "rrf", k: 60 }, metrics: [] },
    { id: "rerank", name: "Rerank", layer: "retrieval", status: "active", summary: "Lexical-overlap reranker reorders the fused pool toward the query terms.", config: { method: "lexical_overlap" }, metrics: [m("mrr", "MRR", 0.58, "", "Mean Reciprocal Rank after reranking.")] },
    { id: "select", name: "Select evidence", layer: "retrieval", status: "active", summary: "Keep the top-5 passages as the evidence set handed to generation.", config: { top_k: 5 }, metrics: [] },
    { id: "generate", name: "Generate", layer: "answering", status: "active", summary: "LLM disabled by default -> extractive fallback stitches the strongest sentences.", config: { llm_enabled: false, mode: "extractive" }, metrics: [] },
    { id: "cite", name: "Cite + validate", layer: "answering", status: "active", summary: "Attach claim-level citations and validate each against its cited passage.", config: { granularity: "claim" }, metrics: [m("citation_support", "Citation support", 0.83, "", "Share of claims backed by a SUPPORTED passage.")] },
  ],
};

export const mockChat: ChatResponse = {
  question: "Does metformin reduce cardiovascular risk in type 2 diabetes?",
  answer:
    "Metformin is associated with reduced cardiovascular events in patients with type 2 diabetes [P1023]. Observational cohorts report lower all-cause mortality versus sulfonylureas [P0847], though randomized evidence remains limited [P0312].",
  evidence_status: "sufficient",
  llm_used: false,
  llm_source: "disabled",
  claims: [
    { text: "Metformin is associated with reduced cardiovascular events in type 2 diabetes.", citations: ["P1023"], support: "SUPPORTED" },
    { text: "Observational cohorts report lower all-cause mortality versus sulfonylureas.", citations: ["P0847"], support: "PARTIALLY_SUPPORTED" },
    { text: "Randomized evidence remains limited.", citations: ["P0312"], support: "SUPPORTED" },
  ],
  citations: [
    {
      passage_id: "P1023",
      text: "In a meta-analysis of type 2 diabetes cohorts, metformin use was associated with a significant reduction in major adverse cardiovascular events.",
      support: "SUPPORTED",
      scores: { bm25: 12.4, dense: 0.81, rrf: 0.032, rerank: 0.91, final_rank: 1 },
      provenance: [
        { level: "claim", label: "Metformin reduces CV events", ref_id: "C1" },
        { level: "citation", label: "P1023", ref_id: "P1023" },
        { level: "passage", label: "Meta-analysis of T2D cohorts", ref_id: "P1023" },
        { level: "chunk", label: "sentence-window #4", ref_id: "P1023#4" },
        { level: "source", label: "PubMed 28931234", ref_id: "PMID:28931234" },
      ],
    },
    {
      passage_id: "P0847",
      text: "Retrospective cohort data suggest lower all-cause mortality among metformin users compared with sulfonylurea monotherapy.",
      support: "PARTIALLY_SUPPORTED",
      scores: { bm25: 9.1, dense: 0.77, rrf: 0.026, rerank: 0.74, final_rank: 2 },
      provenance: [
        { level: "claim", label: "Lower mortality vs sulfonylureas", ref_id: "C2" },
        { level: "citation", label: "P0847", ref_id: "P0847" },
        { level: "passage", label: "Retrospective cohort", ref_id: "P0847" },
        { level: "chunk", label: "sentence-window #2", ref_id: "P0847#2" },
        { level: "source", label: "PubMed 26551272", ref_id: "PMID:26551272" },
      ],
    },
    {
      passage_id: "P0312",
      text: "The authors note that randomized controlled trial evidence for cardiovascular benefit of metformin remains limited.",
      support: "SUPPORTED",
      scores: { bm25: 7.8, dense: 0.70, rrf: 0.021, rerank: 0.66, final_rank: 3 },
      provenance: [
        { level: "claim", label: "RCT evidence limited", ref_id: "C3" },
        { level: "citation", label: "P0312", ref_id: "P0312" },
        { level: "passage", label: "Review discussion", ref_id: "P0312" },
        { level: "chunk", label: "sentence-window #7", ref_id: "P0312#7" },
        { level: "source", label: "PubMed 30153394", ref_id: "PMID:30153394" },
      ],
    },
  ],
  grounding: [
    m("coverage", "Claim coverage", 1.0, "", "Fraction of answer claims that carry at least one citation.", { source: "mock" }),
    m("citation_support", "Citation support", 0.83, "", "Fraction of claims whose citation is judged SUPPORTED.", { source: "mock" }),
    m("faithfulness", "Faithfulness", 0.79, "", "Degree to which the answer stays within the cited evidence.", { source: "mock" }),
  ],
  retrieval_trace: { recipe: "hybrid_rerank", top_k: 5, channels: ["bm25", "dense"], fused: 50 },
};

export const mockEvaluation: EvaluationReport = {
  benchmark_size: 500,
  notes: "Mock evaluation report — figures illustrate the intended layout, not real results.",
  oracle: {
    no_retrieval: 41.0,
    actual: 68.0,
    oracle: 82.0,
    generation_gain: 27.0,
    retrieval_gap: 14.0,
    unit: "%",
  },
  metrics: [
    m("recall_at_5", "Recall@5", 0.71, "", "Top-5 retrieval recall across the benchmark.", { baseline_delta: 0.12 }),
    m("mrr", "MRR", 0.58, "", "Mean reciprocal rank of the first gold passage.", { baseline_delta: 0.09 }),
    m("faithfulness", "Faithfulness", 0.79, "", "Answer grounding in cited evidence.", {}),
    m("latency", "Latency", 320, "ms", "Median end-to-end query latency after warm indexes.", { higher_is_better: false }),
  ],
  leaderboard: [
    { rank: 1, configuration: "hybrid_rerank", experiment_dna: mockDNA, metrics: { recall_at_5: 0.71, mrr: 0.58, faithfulness: 0.79, latency_ms: 320 } },
    { rank: 2, configuration: "hybrid_no_rerank", experiment_dna: null, metrics: { recall_at_5: 0.66, mrr: 0.52, faithfulness: 0.77, latency_ms: 210 } },
    { rank: 3, configuration: "dense_only", experiment_dna: null, metrics: { recall_at_5: 0.61, mrr: 0.49, faithfulness: 0.75, latency_ms: 180 } },
    { rank: 4, configuration: "baseline_bm25", experiment_dna: null, metrics: { recall_at_5: 0.59, mrr: 0.44, faithfulness: 0.72, latency_ms: 120 } },
  ],
  charts: [
    {
      id: "recall_by_config",
      title: "Recall@5 and MRR by configuration",
      kind: "grouped_bar",
      x_label: "Configuration",
      y_label: "Score",
      explanation: "Compares retrieval quality across pipeline configurations on the same benchmark.",
      takeaway: "Hybrid + rerank leads on both recall and MRR; BM25 alone trails.",
      series: [
        { name: "Recall@5", x: ["hybrid_rerank", "hybrid_no_rerank", "dense_only", "baseline_bm25"], y: [0.71, 0.66, 0.61, 0.59] },
        { name: "MRR", x: ["hybrid_rerank", "hybrid_no_rerank", "dense_only", "baseline_bm25"], y: [0.58, 0.52, 0.49, 0.44] },
      ],
    },
    {
      id: "quality_profile",
      title: "Quality profile (radar)",
      kind: "radar",
      x_label: "",
      y_label: "",
      explanation: "Shows the winning configuration across multiple quality axes at once.",
      takeaway: "Balanced profile — no single axis is sacrificed for another.",
      series: [
        { name: "hybrid_rerank", x: ["Recall", "MRR", "Faithfulness", "Coverage", "Support"], y: [0.71, 0.58, 0.79, 1.0, 0.83] },
      ],
    },
    {
      id: "quality_vs_cost",
      title: "Quality vs cost (Pareto)",
      kind: "scatter",
      x_label: "Latency (ms)",
      y_label: "Recall@5",
      explanation: "Each point is a configuration; up-and-left is better (more quality, less cost).",
      takeaway: "hybrid_rerank sits on the Pareto frontier — extra latency buys real recall.",
      series: [
        { name: "configs", x: ["120", "180", "210", "320"], y: [0.59, 0.61, 0.66, 0.71] },
      ],
    },
  ],
};

export const mockSystem: SystemDesign = {
  mermaid: `flowchart TD
  UI["React + TS UI"] --> API["FastAPI service"]
  API --> CAP["Capabilities"]
  API --> RET["Retrieval core"]
  RET --> BM25["BM25 index"]
  RET --> DENSE["Dense index (flat)"]
  RET --> FUSE["RRF fusion"]
  FUSE --> RR["Reranker"]
  API --> GEN["Generation (LLM gated)"]
  GEN --> LLM["common/llm.py\\nbudget=12, cache"]
  API --> EVAL["Evaluation artifacts"]
  API --> STORE["Cached artifacts\\npaths.py"]`,
  versions: [
    { component: "FastAPI", version: "0.115.x" },
    { component: "React", version: "18.3" },
    { component: "Vite", version: "5.4" },
    { component: "TanStack Query", version: "5.59" },
    { component: "Plotly.js", version: "2.35" },
    { component: "Mermaid", version: "11.4" },
    { component: "Embeddings", version: "ada-002 (cached)" },
  ],
  observability: [
    m("llm_calls", "LLM calls used", 0, "", "Live LLM calls consumed this session (hard cap 12).", { higher_is_better: false }),
    m("llm_remaining", "LLM budget remaining", 12, "", "Remaining live LLM calls before the budget blocks further calls.", {}),
    m("cache_hit", "Cache hit rate", 0.94, "", "Fraction of embedding/LLM lookups served from disk cache.", {}),
    m("warm_latency", "Warm query latency", 320, "ms", "Median latency once BM25 + dense indexes are warmed at startup.", { higher_is_better: false }),
  ],
  component_details: {
    UI: { role: "5-page explainable frontend", tech: "React + TS + Vite" },
    API: { role: "Endpoint layer + startup index warming", tech: "FastAPI" },
    RET: { role: "Hybrid retrieval orchestration", tech: "hybrid.retrieve()" },
    BM25: { role: "Lexical index", tech: "rank_bm25" },
    DENSE: { role: "Exact cosine over cached vectors", tech: "flat index" },
    FUSE: { role: "Reciprocal rank fusion", tech: "rrf k=60" },
    RR: { role: "Reorder fused pool", tech: "lexical overlap" },
    GEN: { role: "Answer synthesis (extractive fallback when LLM off)", tech: "generation module" },
    LLM: { role: "Single gated LLM entrypoint", tech: "common/llm.py" },
    EVAL: { role: "Benchmark metrics + charts", tech: "artifacts" },
    STORE: { role: "Cached embeddings/indexes", tech: "common/paths.py" },
  },
};
