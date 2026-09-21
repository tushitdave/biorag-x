import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Capabilities, RetrievalConfig } from "../api/types";

// Per-layer selection: for single-select layers we store one option id; for
// multi-select (search channels) we store an array. Each layer also tracks an
// execution mode where the backend offers them.
export interface LayerSelection {
  selected: string[]; // one entry for single-select, many for multi
  mode: string; // execution mode
}

export type ExperimentState = Record<string, LayerSelection>;

// Numeric settings of the flow (the rest of RetrievalConfig are layer choices).
export type FlowParams = Pick<
  RetrievalConfig,
  "ef_search" | "nprobe" | "depth" | "rrf_k" | "rerank_candidates" | "evidence_k"
>;

// Mirrors RetrievalConfig() defaults in api/contract.py.
export const DEFAULT_PARAMS: FlowParams = {
  ef_search: 64,
  nprobe: 8,
  depth: 40,
  rrf_k: 60,
  rerank_candidates: 50,
  evidence_k: 5,
};

interface ExperimentCtx {
  state: ExperimentState;
  params: FlowParams;
  setLayer: (layer: string, sel: Partial<LayerSelection>) => void;
  setParam: (key: keyof FlowParams, value: number) => void;
  initFromCapabilities: (caps: Capabilities) => void;
  toConfig: () => RetrievalConfig;
  loadConfig: (cfg: RetrievalConfig) => void;
}

const Ctx = createContext<ExperimentCtx | null>(null);

export function useExperiment() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useExperiment must be used within ExperimentProvider");
  return ctx;
}

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ExperimentState>({});
  const [params, setParams] = useState<FlowParams>(DEFAULT_PARAMS);

  const setLayer = useCallback(
    (layer: string, sel: Partial<LayerSelection>) =>
      setState((prev) => ({
        ...prev,
        [layer]: { ...prev[layer], ...sel } as LayerSelection,
      })),
    []
  );

  const setParam = useCallback(
    (key: keyof FlowParams, value: number) => setParams((prev) => ({ ...prev, [key]: value })),
    []
  );

  const initFromCapabilities = useCallback(
    (caps: Capabilities) =>
      setState((prev) => {
        // Only seed layers not already chosen by the user. Return `prev` when
        // nothing was seeded so React skips the re-render.
        const next = { ...prev };
        let changed = false;
        for (const layer of caps.layers) {
          if (next[layer.layer]) continue;
          changed = true;
          const defaults = layer.options
            .filter((o) => o.default && o.available)
            .map((o) => o.id);
          const firstAvailable = layer.options.find((o) => o.available)?.id;
          const selected = layer.multi_select
            ? defaults.length
              ? defaults
              : firstAvailable
                ? [firstAvailable]
                : []
            : defaults.length
              ? [defaults[0]]
              : firstAvailable
                ? [firstAvailable]
                : [];
          next[layer.layer] = {
            selected,
            mode: layer.execution_modes[0] ?? "Manual",
          };
        }
        return changed ? next : prev;
      }),
    []
  );

  const value = useMemo(() => {
    const first = (layer: string, fallback: string) => state[layer]?.selected[0] ?? fallback;

    const toConfig = (): RetrievalConfig => ({
      chunking: first("chunking", "passage"),
      embedding: first("embedding", "medcpt"),
      index: first("index", "flat"),
      query: first("query", "as_is"),
      channels: state.retrieval?.selected ?? ["lexical", "dense"],
      reranker: first("reranker", "none"),
      evidence: first("evidence", "mmr"),
      ...params,
    });

    const loadConfig = (cfg: RetrievalConfig) => {
      setState((prev) => {
        const pick = (layer: string, selected: string[]) => ({
          selected,
          mode: prev[layer]?.mode ?? "Manual",
        });
        return {
          ...prev,
          chunking: pick("chunking", [cfg.chunking]),
          embedding: pick("embedding", [cfg.embedding]),
          index: pick("index", [cfg.index]),
          query: pick("query", [cfg.query ?? "as_is"]),
          retrieval: pick("retrieval", cfg.channels),
          reranker: pick("reranker", [cfg.reranker]),
          evidence: pick("evidence", [cfg.evidence]),
        };
      });
      setParams({
        ef_search: cfg.ef_search,
        nprobe: cfg.nprobe,
        depth: cfg.depth,
        rrf_k: cfg.rrf_k,
        rerank_candidates: cfg.rerank_candidates,
        evidence_k: cfg.evidence_k,
      });
    };

    return { state, params, setLayer, setParam, initFromCapabilities, toConfig, loadConfig };
  }, [state, params, setLayer, setParam, initFromCapabilities]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
