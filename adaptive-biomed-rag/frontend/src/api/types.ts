// TypeScript mirror of api/contract.py (Pydantic models).
// Keep field names identical to the backend contract. Extend, don't rename.

// --------------------------------------------------------------------------- //
// Capabilities
// --------------------------------------------------------------------------- //
export interface Option {
  id: string;
  label: string;
  note: string;
  cost: string; // low | medium | high
  quality: string; // low | medium | high
  default: boolean;
  available: boolean;
  uses_llm: boolean;
  planned?: boolean; // in the research design, not built yet (never selectable)
  status?: string; // index build state: ready | building | queued | failed
  progress?: number | null; // 0..1 while building
}

export interface LayerOptions {
  layer: string; // chunking|embedding|index|retrieval|reranker|generation
  label: string;
  multi_select: boolean;
  execution_modes: string[]; // Manual|Adaptive|Agentic
  options: Option[];
}

export interface Capabilities {
  layers: LayerOptions[];
}

// --------------------------------------------------------------------------- //
// Common
// --------------------------------------------------------------------------- //
export interface MetricValue {
  key: string;
  label: string;
  value: number;
  unit: string; // "" | "%" | "ms" | "USD" | "/5"
  higher_is_better: boolean;
  definition: string;
  baseline_delta: number | null;
  ci_low: number | null;
  ci_high: number | null;
  source: string; // artifact | live | mock
}

export interface HealthInfo {
  status: string;
  artifacts: Record<string, boolean>;
  settings: Record<string, unknown>;
  llm_usage: Record<string, unknown>;
}

// --------------------------------------------------------------------------- //
// Overview
// --------------------------------------------------------------------------- //
export interface ExperimentDNA {
  experiment_id: string;
  chunking: string;
  embedding: string;
  ann: string;
  retrieval: string;
  fusion: string;
  reranker: string;
  agent: string;
  evidence: string;
  generator: string;
  citation: string;
}

export interface HeadlineMetrics {
  experiment: ExperimentDNA;
  metrics: MetricValue[];
  corpus_passages: number;
  questions: number;
  gold_relationships: number;
}

// --------------------------------------------------------------------------- //
// Pipeline (Workflow)
// --------------------------------------------------------------------------- //
export interface PipelineStage {
  id: string;
  name: string;
  layer: string; // data|representation|retrieval|answering|research
  status: string; // ready|active|off
  summary: string;
  metrics: MetricValue[];
  config: Record<string, unknown>;
}

export interface WorkflowDiagram {
  mermaid: string;
  stage_ids_in_order: string[];
}

export interface PipelineView {
  stages: PipelineStage[];
  workflow: WorkflowDiagram;
}

// --------------------------------------------------------------------------- //
// Chat / Answering
// --------------------------------------------------------------------------- //
export interface ProvenanceStep {
  level: string; // claim|citation|passage|chunk|section|document|source
  label: string;
  ref_id: string | null;
}

export type SupportLevel = "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED";

export interface Citation {
  passage_id: string;
  text: string;
  support: string; // SupportLevel
  scores: Record<string, number>;
  provenance: ProvenanceStep[];
}

export interface Claim {
  text: string;
  citations: string[]; // passage_ids
  support: string; // SupportLevel
}

export interface ChatRequest {
  question: string;
  recipe: string;
  top_k: number;
  reranker?: string;
  config?: RetrievalConfig; // the pipeline flow; overrides the recipe when set
}

export interface ChatResponse {
  question: string;
  answer: string;
  evidence_status: string; // sufficient|weak|insufficient
  claims: Claim[];
  citations: Citation[];
  grounding: MetricValue[];
  llm_used: boolean;
  llm_source: string; // cache|live|disabled|budget
  retrieval_trace: Record<string, unknown>;
}

// --------------------------------------------------------------------------- //
// Retrieval
// --------------------------------------------------------------------------- //
export interface RetrievedPassage {
  passage_id: string;
  text: string;
  bm25_score: number | null;
  dense_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  final_rank: number;
  is_gold: boolean | null;
}

export interface RetrievalResponse {
  question: string;
  recipe: string;
  passages: RetrievedPassage[];
  metrics: MetricValue[];
  trace: Record<string, unknown>;
}

// --------------------------------------------------------------------------- //
// Evaluation
// --------------------------------------------------------------------------- //
export interface LeaderboardRow {
  rank: number;
  configuration: string;
  experiment_dna: ExperimentDNA | null;
  metrics: Record<string, number>;
}

export interface OracleComparison {
  no_retrieval: number;
  actual: number;
  oracle: number;
  generation_gain: number;
  retrieval_gap: number;
  unit: string;
}

export interface ChartSeries {
  name: string;
  x: string[];
  y: number[];
}

export type ChartKind = "bar" | "grouped_bar" | "radar" | "scatter" | "line";

export interface ExplainableChart {
  id: string;
  title: string;
  kind: string; // ChartKind
  series: ChartSeries[];
  x_label: string;
  y_label: string;
  explanation: string;
  takeaway: string;
}

export interface EvaluationReport {
  benchmark_size: number;
  leaderboard: LeaderboardRow[];
  metrics: MetricValue[];
  charts: ExplainableChart[];
  oracle: OracleComparison | null;
  notes: string;
}

// --------------------------------------------------------------------------- //
// System Design
// --------------------------------------------------------------------------- //
export interface VersionInfo {
  component: string;
  version: string;
}

export interface SystemDesign {
  mermaid: string;
  versions: VersionInfo[];
  observability: MetricValue[];
  component_details: Record<string, unknown>;
}

// --------------------------------------------------------------------------- //
// Experiment lab (mirrors RetrievalConfig / Run* / Compare* in api/contract.py)
// --------------------------------------------------------------------------- //
export interface RetrievalConfig {
  chunking: string; // passage | semantic
  embedding: string; // medcpt | ada002_azure
  index: string; // flat | hnsw | ivf
  query?: string; // as_is | synonyms | router (router picks the channels)
  ef_search: number;
  nprobe: number;
  channels: string[]; // subset of lexical, dense, graph, pageindex
  depth: number; // candidates per search channel
  rrf_k: number;
  reranker: string;
  rerank_candidates: number;
  evidence: string; // mmr | topk
  evidence_k: number;
}

export interface QuestionSetInfo {
  id: string;
  label: string;
  n: number;
  locked: boolean;
  corpus: string;
}

export interface CorpusInfo {
  id: string;
  label: string;
  passages: number;
}

export interface LabInfo {
  default_config: RetrievalConfig;
  question_sets: QuestionSetInfo[];
  corpora: CorpusInfo[];
}

export interface RunCreate {
  config: RetrievalConfig;
  question_set: string;
  corpus: string;
  name: string;
}

export interface RunMetric {
  key: string;
  label: string;
  value: number;
  ci_low: number | null;
  ci_high: number | null;
  unit: string;
}

export interface RunSummary {
  id: string;
  name: string;
  created_at: string;
  status: string; // running | done | failed
  question_set: string;
  corpus: string;
  n_questions: number;
  done: number;
  config: RetrievalConfig;
  headline: Record<string, number>;
  duration_s: number | null;
  error: string;
}

export interface SliceRow {
  dimension: string;
  value: string;
  n: number;
  metrics: Record<string, number>;
}

export interface MissedQuestion {
  question: string;
  question_type: string;
  n_gold: number;
  gold_in_top20: number;
}

export interface RunDetail {
  run: RunSummary;
  metrics: RunMetric[];
  slices: SliceRow[];
  missed: MissedQuestion[];
}

export interface ConfigChange {
  field: string;
  a: string;
  b: string;
}

export interface CompareMetric {
  key: string;
  label: string;
  a: number;
  b: number;
  delta: number;
  ci_low: number;
  ci_high: number;
  significant: boolean;
  higher_is_better: boolean;
}

export interface SliceDelta {
  dimension: string;
  value: string;
  n: number;
  delta: Record<string, number>;
}

export interface QuestionDelta {
  question: string;
  a: number;
  b: number;
}

export interface CompareResult {
  a: RunSummary;
  b: RunSummary;
  comparable: boolean;
  reason: string;
  changes: ConfigChange[];
  metrics: CompareMetric[];
  decision_metric: string;
  wins: number;
  losses: number;
  ties: number;
  slices: SliceDelta[];
  top_gains: QuestionDelta[];
  top_losses: QuestionDelta[];
}

// --------------------------------------------------------------------------- //
// Ask: per-question trust report (mirrors Ask* / ComboResult in api/contract.py)
// --------------------------------------------------------------------------- //
export interface AskRequest {
  question: string;
  config: RetrievalConfig;
}

export interface EvidenceItem {
  passage_id: string;
  chunk_id: string;
  text: string; // what the LLM would receive
  child_text: string | null; // parent-child: the child that matched
  judge: number; // 0..1 relevance to the question
  scores: Record<string, number>; // bm25 / dense / graph / pageindex / rrf / rerank scores
  why?: string | null; // graph path or PageIndex tree path that found it
  rank: number;
  is_gold: boolean | null; // dataset questions only
}

export interface ComboResult {
  id: string;
  group: string; // yours | chunking | retrieval
  config: RetrievalConfig;
  is_user: boolean;
  is_best: boolean;
  self_judged: boolean; // reranked by the judge's own model: not eligible for best
  evidence_score: number;
  best_evidence: number;
  coverage: number;
  agreement: number | null;
  latency_ms: number;
  true_metrics: Record<string, number> | null;
  evidence: EvidenceItem[];
  note: string;
  error: string;
}

export interface AskCompareResponse {
  question: string;
  gold_available: boolean;
  n_gold: number;
  combos: ComboResult[];
  best_id: string;
  user_id: string;
  judge: { model: string; how: string; calibration: JudgeCalibration | null };
  seconds: number;
}

export interface JudgeCalibration {
  questions: number;
  strategies: string[];
  pick_is_oracle: number; // share of questions where the judge picked a truly best combination
  regret: number; // mean nDCG@10 lost vs the truly best combination
  picked_ndcg: number;
  default_ndcg: number;
  oracle_ndcg: number;
  rank_corr: number | null;
}

// --------------------------------------------------------------------------- //
// Architecture diagrams (mirrors Architecture* in api/contract.py)
// --------------------------------------------------------------------------- //
export interface ArchitectureNode {
  title: string;
  what: string;
  rules: string[]; // thresholds / decision rules in force
  source: string; // where it lives in the code
  status: string; // live | building | planned
}

export interface ArchitectureDiagram {
  id: string;
  title: string;
  description: string;
  mermaid: string;
  nodes: Record<string, ArchitectureNode>;
}

export interface ArchitectureView {
  process: ArchitectureDiagram[];
  system: ArchitectureDiagram[];
  legend: { label: string; cls: string }[];
}
