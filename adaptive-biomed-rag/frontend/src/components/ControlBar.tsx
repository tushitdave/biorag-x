import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import type { LayerOptions, Option } from "../api/types";
import { useExperiment, type FlowParams } from "./ExperimentState";

// MedCPT vectors exist for every chunking strategy except NB04's ada-002 semantic
// chunks, so choosing a chunking method also fixes the embedding model.
const embeddingFor = (unit: string) => (unit === "semantic_nb04" ? "ada002_azure" : "medcpt");
const NEEDS: Record<string, string> = {
  medcpt: "a MedCPT-built chunking",
  ada002_azure: "NB04 semantic chunks",
};

interface Step {
  title: string;
  learn: string; // what happens at this step, for a learner
  layer?: string; // capability layer chosen at this step
  fixed?: string; // shown when the step has no choice
}

const BUILD: Step[] = [
  {
    title: "Ingestion",
    fixed: "Cleaned corpus",
    learn:
      "Load the 40,221 BioASQ passages, clean the text and drop the 12,220 empty ones. Done once, offline (Notebook 02).",
  },
  {
    title: "Chunking",
    layer: "chunking",
    learn: "Choose the unit we search over: a whole passage or smaller pieces of it.",
  },
  {
    title: "Embedding",
    layer: "embedding",
    learn: "Turn each unit into a vector, so texts with similar meaning sit close together.",
  },
  {
    title: "Vector index",
    layer: "index",
    learn: "Store the vectors so the closest ones are found fast. Exact checks all; ANN trades a little recall for speed.",
  },
];

const ANSWER: Step[] = [
  {
    title: "Query",
    layer: "query",
    learn: "Decide what to search with: the question as typed, with synonyms added, or let the router choose the search per question.",
  },
  {
    title: "Search",
    layer: "retrieval",
    learn: "BM25 finds exact words; dense finds meaning; graph follows entity links; PageIndex browses a topic tree. Combine any.",
  },
  {
    title: "Fusion",
    fixed: "RRF",
    learn: "Merge the ranked lists: passages ranked high by either search rise to the top.",
  },
  {
    title: "Reranking",
    layer: "reranker",
    learn: "Optionally re-score the top candidates with a slower, more careful model.",
  },
  {
    title: "Evidence",
    layer: "evidence",
    learn: "Pick the passages sent to the LLM, ideally relevant and not repeating each other.",
  },
  {
    title: "Generation",
    layer: "generation",
    learn: "The LLM answers using only the evidence and cites a passage for every claim.",
  },
  {
    title: "Citation check",
    fixed: "Strict",
    learn: "Each citation must point to a retrieved passage whose words back the claim.",
  },
];

const ROWS: { label: string; steps: Step[] }[] = [
  { label: "Build the index (offline, once)", steps: BUILD },
  { label: "Answer a question (online, every time)", steps: ANSWER },
];
const ALL_STEPS = [...BUILD, ...ANSWER];
const COLLAPSE_KEY = "biorag.flow.collapsed";

function optionLabel(o: Option, blockedBy?: string): string {
  if (o.planned) return `${o.label} (planned)`;
  if (o.status === "building")
    return `${o.label} (building ${Math.round((o.progress ?? 0) * 100)}%)`;
  if (o.status === "paused")
    return `${o.label} (build paused at ${Math.round((o.progress ?? 0) * 100)}%)`;
  if (o.status === "queued") return `${o.label} (queued to build)`;
  if (o.status === "failed") return `${o.label} (build failed)`;
  if (blockedBy) return `${o.label} (needs ${blockedBy})`;
  if (!o.available) return `${o.label} (unavailable)`;
  return o.label;
}

function OptionPicker({ layer }: { layer: LayerOptions }) {
  const { state, setLayer } = useExperiment();
  const current = state[layer.layer]?.selected[0] ?? "";
  const unit = state.chunking?.selected[0];

  // An embedding model is only usable with the unit type its vectors were built for.
  const blockedBy = (o: Option) =>
    layer.layer === "embedding" && unit && !o.planned && embeddingFor(unit) !== o.id
      ? NEEDS[o.id] ?? "another chunking"
      : undefined;

  const pick = (id: string) => {
    setLayer(layer.layer, { selected: [id] });
    if (layer.layer === "chunking") setLayer("embedding", { selected: [embeddingFor(id)] });
  };

  return (
    <select value={current} onChange={(e) => pick(e.target.value)} aria-label={layer.label}>
      {layer.options.map((o) => {
        const blocked = blockedBy(o);
        return (
          <option key={o.id} value={o.id} disabled={!o.available || !!blocked}>
            {optionLabel(o, blocked)}
          </option>
        );
      })}
    </select>
  );
}

function ChannelPicker({ layer }: { layer: LayerOptions }) {
  const { state, setLayer } = useExperiment();
  const sel = state[layer.layer]?.selected ?? [];

  const toggle = (id: string) => {
    const next = sel.includes(id) ? sel.filter((x) => x !== id) : [...sel, id];
    if (next.length) setLayer(layer.layer, { selected: next }); // keep at least one search
  };

  return (
    <div className="flow-channels">
      {layer.options.map((o) => (
        <label key={o.id} className={o.available ? "" : "disabled"} title={o.note}>
          <input
            type="checkbox"
            checked={sel.includes(o.id)}
            disabled={!o.available}
            onChange={() => toggle(o.id)}
          />
          {o.label}
          {o.planned && <span className="flow-planned">planned</span>}
        </label>
      ))}
    </div>
  );
}

function NumberField({ label, name, min, max }: {
  label: string;
  name: keyof FlowParams;
  min: number;
  max: number;
}) {
  const { params, setParam } = useExperiment();
  return (
    <label className="flow-param">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        value={params[name]}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (Number.isFinite(v)) setParam(name, Math.min(max, Math.max(min, Math.round(v))));
        }}
      />
    </label>
  );
}

function chosenOption(layer: LayerOptions | undefined, selected: string[] | undefined) {
  return layer?.options.find((o) => o.id === selected?.[0]);
}

// Persistent pipeline flow fed by GET /capabilities: one numbered card per
// step, in pipeline order, with a plain-language explanation of what happens
// there and the learner's choice. Chat and Retrieval Lab runs use exactly it.
export function ControlBar() {
  const { data } = useQuery({
    queryKey: ["capabilities"],
    queryFn: api.capabilities,
    // While chunking indexes build in the background, refresh their progress.
    refetchInterval: (q) =>
      q.state.data?.data.layers.some((l) =>
        l.options.some((o) => o.status === "building" || o.status === "queued")
      )
        ? 15000
        : false,
  });
  const { state, params, initFromCapabilities } = useExperiment();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === "1";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    if (data?.data) initFromCapabilities(data.data);
  }, [data, initFromCapabilities]);

  const toggleCollapsed = () => {
    const next = !collapsed;
    setCollapsed(next);
    try {
      localStorage.setItem(COLLAPSE_KEY, next ? "1" : "0");
    } catch {
      /* per-viewer convenience only */
    }
  };

  if (!data) return <div className="flow-bar">Loading pipeline options…</div>;

  const layers = Object.fromEntries(data.data.layers.map((l) => [l.layer, l]));
  const channels = state.retrieval?.selected ?? [];
  const routed = state.query?.selected[0] === "router";
  const denseOff = !routed && !channels.includes("dense");
  const index = state.index?.selected[0];
  const reranker = state.reranker?.selected[0];

  // What each step shows as its current value (used by cards and the summary).
  const valueOf = (step: Step): string => {
    if (step.layer === "retrieval" && routed) return "Chosen per question by the router";
    if (step.title === "Fusion")
      return !routed && channels.length < 2 ? "Skipped (one search)" : `RRF k=${params.rrf_k}`;
    if (step.fixed) return step.fixed;
    const layer = layers[step.layer!];
    if (!layer) return "—";
    if (layer.multi_select) {
      return layer.options.filter((o) => channels.includes(o.id)).map((o) => o.label).join(" + ");
    }
    return chosenOption(layer, state[layer.layer]?.selected)?.label ?? "—";
  };

  // Why a step does nothing in the current flow (null = it is used).
  const unusedReason = (step: Step): string | null => {
    if (denseOff && (step.layer === "embedding" || step.layer === "index"))
      return "Not used: dense search is off.";
    if (routed && step.layer === "retrieval")
      return "The agentic router picks the channels for each question.";
    return null;
  };
  const inactive = (step: Step) => unusedReason(step) !== null;

  // Extra numeric settings shown under a step's main control.
  const paramsFor = (step: Step) => {
    if (step.layer === "index" && index === "hnsw")
      return <NumberField label="efSearch" name="ef_search" min={8} max={1024} />;
    if (step.layer === "index" && index === "ivf")
      return <NumberField label="nprobe" name="nprobe" min={1} max={512} />;
    if (step.layer === "retrieval")
      return <NumberField label="Candidates per search" name="depth" min={5} max={200} />;
    if (step.title === "Fusion" && (routed || channels.length > 1))
      return <NumberField label="RRF k" name="rrf_k" min={1} max={200} />;
    if (step.layer === "reranker" && reranker && reranker !== "none")
      return <NumberField label="Candidates to rerank" name="rerank_candidates" min={5} max={200} />;
    if (step.layer === "evidence")
      return <NumberField label="Passages (k)" name="evidence_k" min={1} max={20} />;
    return null;
  };

  let n = 0;
  return (
    <div className="flow-bar">
      <div className="flow-head">
        <strong>Pipeline flow</strong>
        <span className="flow-sub">
          Pick an option at each step. Chat compares your flow with the alternatives and
          shows which works best for each question.
        </span>
        {data.isMock && (
          <span className="badge mock" title={data.error}>
            backend offline
          </span>
        )}
        <button className="flow-toggle" onClick={toggleCollapsed}>
          {collapsed ? "Show steps" : "Hide steps"}
        </button>
      </div>

      {collapsed ? (
        <div className="flow-summary">
          {ALL_STEPS.map((s, i) => (
            <span key={s.title} className={inactive(s) ? "inactive" : ""}>
              {i > 0 && <span className="flow-arrow">→</span>}
              {valueOf(s)}
            </span>
          ))}
        </div>
      ) : (
        ROWS.map((row, r) => (
          <div key={row.label}>
            <div className="flow-row-label">{row.label}</div>
            <div className="flow-row">
              {r > 0 && <span className="flow-arrow flow-wrap">↳</span>}
              {row.steps.map((step, i) => {
                n += 1;
                const layer = step.layer ? layers[step.layer] : undefined;
                const note = layer && !layer.multi_select
                  ? chosenOption(layer, state[layer.layer]?.selected)?.note
                  : undefined;
                return (
                  <div className="flow-cell" key={step.title}>
                    {i > 0 && <span className="flow-arrow">→</span>}
                    <div className={`flow-step ${inactive(step) ? "inactive" : ""}`}>
                      <div className="flow-step-head">
                        <span className="flow-num">{n}</span>
                        <span className="flow-title">{step.title}</span>
                        <span className={`flow-tag ${layer ? "choice" : ""}`}>
                          {layer ? "choose" : "fixed"}
                        </span>
                      </div>
                      {layer ? (
                        layer.multi_select ? (
                          <ChannelPicker layer={layer} />
                        ) : (
                          <OptionPicker layer={layer} />
                        )
                      ) : (
                        <div className="flow-fixed">{valueOf(step)}</div>
                      )}
                      {(!inactive(step) || (routed && step.layer === "retrieval")) && paramsFor(step)}
                      <p className="flow-learn">
                        {unusedReason(step) ?? step.learn}
                      </p>
                      {note && !inactive(step) && <p className="flow-note">{note}</p>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
