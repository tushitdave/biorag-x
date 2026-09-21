// Plain-language names for flow settings, question slices and metrics, shared by
// the Retrieval Lab and Runs & Compare pages.
import type { RetrievalConfig } from "../api/types";

/** Run statuses that are still in progress (poll while any run has one). */
export const ACTIVE = new Set(["queued", "running"]);

const NAMES: Record<string, string> = {
  passage: "Whole passage",
  fixed: "Fixed-size",
  recursive: "Recursive",
  semantic: "Semantic",
  biomedical: "Biomedical-aware",
  proposition: "Proposition",
  parent_child: "Parent-child",
  late: "Late chunking",
  adaptive: "Adaptive (router)",
  agentic: "Agentic (LLM)",
  semantic_nb04: "Semantic (NB04 ada-002)",
  medcpt: "MedCPT",
  ada002_azure: "ada-002",
  flat: "Exact",
  hnsw: "HNSW",
  ivf: "IVF",
  lexical: "BM25",
  dense: "Dense",
  graph: "Graph",
  as_is: "As typed",
  synonyms: "Synonyms",
  router: "Agentic router",
  pageindex: "PageIndex",
  none: "No rerank",
  medcpt_ce: "MedCPT cross-encoder",
  lexical_overlap: "Word overlap",
  cross_encoder: "Word-pair proxy",
  colbert: "Fuzzy-token proxy",
  llm_reranker: "GPT-4o ranking",
  mmr: "MMR",
  topk: "Top-k",
};

export const nameOf = (id: string) =>
  id
    .split(" + ")
    .map((part) => NAMES[part] ?? part)
    .join(" + ");

export const FIELD_LABELS: Record<string, string> = {
  chunking: "Chunking",
  embedding: "Embedding",
  index: "Vector index",
  ef_search: "efSearch",
  nprobe: "nprobe",
  channels: "Search",
  depth: "Candidates per search",
  rrf_k: "RRF k",
  reranker: "Reranker",
  rerank_candidates: "Rerank candidates",
  evidence: "Evidence",
  evidence_k: "Evidence k",
};

/** The pipeline step each setting belongs to (a step can own several settings). */
export const STEP_OF: Record<string, string> = {
  chunking: "Chunking",
  embedding: "Embedding",
  index: "Vector index",
  ef_search: "Vector index",
  nprobe: "Vector index",
  channels: "Search",
  depth: "Search",
  rrf_k: "Fusion",
  reranker: "Reranking",
  rerank_candidates: "Reranking",
  evidence: "Evidence",
  evidence_k: "Evidence",
};

/** Short chips describing only the settings that affect this flow. */
export function describeConfig(c: RetrievalConfig): string[] {
  const parts = [nameOf(c.chunking)];
  const routed = c.query === "router";
  if (c.query && c.query !== "as_is") parts.push(nameOf(c.query));
  if (routed || c.channels.includes("dense")) {
    parts.push(nameOf(c.embedding));
    parts.push(
      c.index === "hnsw"
        ? `HNSW ef=${c.ef_search}`
        : c.index === "ivf"
          ? `IVF nprobe=${c.nprobe}`
          : "Exact index"
    );
  }
  parts.push(`${routed ? "Channels per question" : c.channels.map(nameOf).join(" + ")} top ${c.depth}`);
  if (routed || c.channels.length > 1) parts.push(`RRF k=${c.rrf_k}`);
  parts.push(c.reranker === "none" ? "No rerank" : `${nameOf(c.reranker)} (${c.rerank_candidates})`);
  parts.push(`${nameOf(c.evidence)} ${c.evidence_k}`);
  return parts;
}

/** Rough run time on this machine, measured: ~0.05 s/question retrieval,
 *  ~0.066 s per candidate pair for the MedCPT cross-encoder on the M1 GPU. */
export function estimateSeconds(c: RetrievalConfig, questions: number): number {
  const perQuestion = 0.05 + (c.reranker === "medcpt_ce" ? 0.066 * c.rerank_candidates : 0);
  return questions * perQuestion;
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  return `${Math.round(seconds / 60)} min`;
}

// Slice values in a meaningful order with readable names.
export const SLICE_DIMENSIONS: { id: string; label: string; values: [string, string][] }[] = [
  {
    id: "question_type",
    label: "Question type (inferred)",
    values: [
      ["yes_no_candidate", "Yes/No"],
      ["factoid_candidate", "Factoid"],
      ["list_or_factoid_candidate", "List"],
      ["summary_or_mechanism_candidate", "Summary"],
      ["other", "Other"],
    ],
  },
  {
    id: "lexical_overlap",
    label: "Word overlap with gold evidence",
    values: [
      ["low_overlap", "Low"],
      ["medium_low", "Medium-low"],
      ["medium_high", "Medium-high"],
      ["high_overlap", "High"],
    ],
  },
  {
    id: "evidence_size",
    label: "Gold passages per question",
    values: [
      ["single", "1"],
      ["two", "2"],
      ["multi_3_5", "3-5"],
      ["multi_6_plus", "6+"],
    ],
  },
];

export const METRIC_HELP: Record<string, string> = {
  "hit@1": "Share of questions whose top result is a gold passage.",
  "hit@5": "Share of questions with at least one gold passage in the top 5.",
  "hit@10": "Share of questions with at least one gold passage in the top 10.",
  "recall@5": "Average share of each question's gold passages found in the top 5.",
  "recall@10": "Average share of each question's gold passages found in the top 10.",
  "recall@20": "Average share of each question's gold passages found in the top 20.",
  "mrr@10": "1 / rank of the first gold passage (0 if none in the top 10), averaged.",
  "ndcg@10": "Ranking quality of the top 10: gold passages count more the higher they rank.",
  evidence_recall: "Share of gold passages that made it into the evidence sent to the LLM.",
  evidence_precision: "Share of the evidence sent to the LLM that is gold.",
  latency_ms: "Average retrieval time per question.",
  latency_p50: "Median retrieval time per question.",
  latency_p95: "95% of questions retrieve faster than this.",
  "ann_overlap@10": "How many of exact search's top 10 the approximate index also returns.",
};

export const fmtMetric = (key: string, v: number) =>
  key.startsWith("latency") ? `${Math.round(v)} ms` : v.toFixed(3);
