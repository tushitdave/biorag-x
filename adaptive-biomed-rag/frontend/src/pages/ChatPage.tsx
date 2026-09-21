import { Fragment, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, ask } from "../api/client";
import type {
  AskCompareResponse,
  ChatResponse,
  ComboResult,
  EvidenceItem,
  RetrievalConfig,
} from "../api/types";
import { MetricCard } from "../components/MetricCard";
import { useExperiment } from "../components/ExperimentState";
import { InlineBar } from "../components/LabCharts";
import { describeConfig, nameOf } from "../components/labLabels";

// Render answer text, turning [P1023]-style refs into markers that scroll to the evidence.
function AnswerText({ text }: { text: string }) {
  const parts = text.split(/(\[[A-Za-z0-9_-]+\])/g);
  return (
    <p className="answer">
      {parts.map((part, i) => {
        const match = part.match(/^\[([A-Za-z0-9_-]+)\]$/);
        if (!match) return <span key={i}>{part}</span>;
        return (
          <span key={i} className="cite-ref" title={`Jump to evidence ${match[1]}`}
            onClick={() => document.getElementById(`ev-${match[1]}`)
              ?.scrollIntoView({ behavior: "smooth", block: "center" })}>
            {part}
          </span>
        );
      })}
    </p>
  );
}

const GROUP: Record<string, string> = {
  yours: "Your flow",
  chunking: "Chunking",
  retrieval: "Retrieval",
  query: "Query",
};

const search = (c: RetrievalConfig) =>
  c.query === "router" ? "Agentic router (channels per question)" :
  (c.query === "synonyms" ? "Synonyms · " : "") +
  c.channels.map(nameOf).join(" + ") +
  (c.channels.includes("dense") && c.index !== "flat" ? ` (${nameOf(c.index)})` : "") +
  (c.channels.length > 1 ? " · RRF" : "");

const SCORE_LABEL: Record<string, string> = {
  bm25_score: "BM25",
  dense_score: "Dense",
  graph_score: "Graph",
  pageindex_score: "PageIndex",
  rrf_score: "RRF",
  rerank_score: "Rerank",
};

function EvidenceList({ items, support }: {
  items: EvidenceItem[];
  support?: Record<string, string>;
}) {
  return (
    <div className="evidence-list">
      {items.map((e) => (
        <div className="evi" key={e.chunk_id} id={`ev-${e.passage_id}`}>
          <div className="evi-head">
            <span className="evi-rank">#{e.rank}</span>
            <strong>{e.passage_id}</strong>
            <span className="evi-judge" title="How relevant the MedCPT judge finds this text">
              relevance <InlineBar value={e.judge} /> {e.judge.toFixed(2)}
            </span>
            {e.is_gold === true && <span className="badge SUPPORTED">gold passage</span>}
            {e.is_gold === false && <span className="badge muted-badge">not gold</span>}
            {support?.[e.passage_id] && (
              <span className={`badge ${support[e.passage_id]}`}>cited: {support[e.passage_id]}</span>
            )}
          </div>
          <div className="chips">
            {Object.entries(e.scores).map(([k, v]) => (
              <span className="chip" key={k}>
                <span className="k">{SCORE_LABEL[k] ?? k}:</span> {v.toFixed(3)}
              </span>
            ))}
            {e.chunk_id !== e.passage_id && <span className="chip">chunk {e.chunk_id.slice(0, 12)}</span>}
          </div>
          {e.why && <p className="evi-child"><strong>Found via</strong> {e.why}</p>}
          {e.child_text && (
            <p className="evi-child"><strong>Matched child:</strong> {e.child_text}</p>
          )}
          <p className="evi-text">{e.text}</p>
        </div>
      ))}
    </div>
  );
}

function ComboTable({ data, answeredId, onAnswerWith, busy }: {
  data: AskCompareResponse;
  answeredId?: string;
  onAnswerWith: (c: ComboResult) => void;
  busy: boolean;
}) {
  const [open, setOpen] = useState<string | null>(null);
  const rows = [...data.combos].sort(
    (a, b) => Number(!!a.error) - Number(!!b.error) || b.evidence_score - a.evidence_score
  );
  const gold = data.gold_available;
  return (
    <table className="tbl combo-tbl">
      <thead>
        <tr>
          <th />
          <th>Chunking</th>
          <th>Search</th>
          <th>Rerank</th>
          <th title="Mean relevance of the evidence this combination would send to the LLM">
            Evidence score
          </th>
          <th title="Relevance of its single best passage">Best passage</th>
          <th title="Share of the question's key words found in the evidence">Word coverage</th>
          <th title="Share of BM25's top 10 that dense search also ranks top 10">Agreement</th>
          {gold && <th title="Against this question's gold passages">True nDCG@10</th>}
          {gold && <th>True Hit@5</th>}
          <th>Time</th>
          <th />
        </tr>
      </thead>
      <tbody>
        {rows.map((c) => (
          <Fragment key={c.id}>
            <tr className={`${c.is_best ? "best" : ""} ${c.id === answeredId ? "answered" : ""}`}
              onClick={() => setOpen(open === c.id ? null : c.id)}>
              <td className="tags">
                {c.is_best && <span className="tag best">✓ Best</span>}
                {c.is_user && <span className="tag yours">Yours</span>}
                {!c.is_best && !c.is_user && <span className="tag">{GROUP[c.group]}</span>}
                {c.self_judged && (
                  <span className="tag warn" title={c.note}>self-judged</span>
                )}
              </td>
              <td>{nameOf(c.config.chunking)}</td>
              <td>{search(c.config)}</td>
              <td>{c.config.reranker === "none" ? "—" : nameOf(c.config.reranker)}</td>
              {c.error ? (
                <td colSpan={gold ? 7 : 5} className="muted">Could not run: {c.error}</td>
              ) : (
                <>
                  <td><InlineBar value={c.evidence_score} /> <strong>{c.evidence_score.toFixed(2)}</strong></td>
                  <td>{c.best_evidence.toFixed(2)}</td>
                  <td>{Math.round(c.coverage * 100)}%</td>
                  <td>{c.agreement === null ? "—" : `${Math.round(c.agreement * 100)}%`}</td>
                  {gold && <td>{c.true_metrics?.["ndcg@10"].toFixed(2) ?? "—"}</td>}
                  {gold && <td>{c.true_metrics ? (c.true_metrics["hit@5"] ? "yes" : "no") : "—"}</td>}
                  <td>{Math.round(c.latency_ms)} ms</td>
                </>
              )}
              <td className="expand">{open === c.id ? "▾" : "▸"}</td>
            </tr>
            {open === c.id && !c.error && (
              <tr className="combo-detail">
                <td colSpan={gold ? 12 : 10}>
                  <div className="chips">
                    {describeConfig(c.config).map((p) => <span className="chip" key={p}>{p}</span>)}
                  </div>
                  {c.note && <p className="hint">{c.note}</p>}
                  <EvidenceList items={c.evidence} />
                  {c.id !== answeredId && (
                    <button className="primary small" disabled={busy}
                      onClick={(ev) => { ev.stopPropagation(); onAnswerWith(c); }}>
                      Answer with this combination (1 LLM call)
                    </button>
                  )}
                </td>
              </tr>
            )}
          </Fragment>
        ))}
      </tbody>
    </table>
  );
}

function StrategiesPanel() {
  const { data } = useQuery({ queryKey: ["capabilities"], queryFn: api.capabilities });
  if (!data) return null;
  const show = ["chunking", "retrieval", "index", "reranker", "evidence", "query"];
  const label = (o: { status?: string; planned?: boolean; available: boolean; progress?: number | null }) =>
    o.planned ? "planned" :
    o.status === "building" ? `building ${Math.round((o.progress ?? 0) * 100)}%` :
    o.status === "paused" ? `paused ${Math.round((o.progress ?? 0) * 100)}%` :
    o.status === "queued" ? "queued" :
    o.available ? "ready" : "unavailable";
  return (
    <div className="grid cols-3">
      {data.data.layers.filter((l) => show.includes(l.layer)).map((l) => (
        <div className="card" key={l.layer}>
          <h3>{l.label}</h3>
          <ul className="strategy-list">
            {l.options.map((o) => (
              <li key={o.id} title={o.note}>
                <span className={`state ${label(o).split(" ")[0]}`}>{label(o)}</span> {o.label}
                <div className="hint">{o.note}</div>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export function ChatPage() {
  const { toConfig, state } = useExperiment();
  const [q, setQ] = useState("");
  const [answerWith, setAnswerWith] = useState<"best" | "mine">("best");
  const [answeredId, setAnsweredId] = useState<string | undefined>();

  const answer = useMutation({
    mutationFn: ({ question, combo }: { question: string; combo: ComboResult }) =>
      ask.answer({ question, config: combo.config }),
    onMutate: ({ combo }) => setAnsweredId(combo.id),
  });
  const compare = useMutation({
    mutationFn: (question: string) => ask.compare({ question, config: toConfig() }),
    onSuccess: (res) => {
      const id = answerWith === "best" ? res.best_id : res.user_id;
      const combo = res.combos.find((c) => c.id === id)!;
      answer.mutate({ question: res.question, combo });
    },
  });

  const run = (question = q) => {
    if (!question.trim()) return;
    answer.reset();
    setAnsweredId(undefined);
    compare.mutate(question.trim());
  };

  // A link like /chat?q=... asks that question once the flow options are loaded.
  const [params] = useSearchParams();
  const asked = useRef(false);
  const flowReady = !!state.chunking;
  useEffect(() => {
    const linked = params.get("q");
    if (linked && flowReady && !asked.current) {
      asked.current = true;
      setQ(linked);
      run(linked);
    }
  });

  const cmp = compare.data;
  const r: ChatResponse | undefined = answer.data;
  const answered = cmp?.combos.find((c) => c.id === answeredId);
  const support = Object.fromEntries((r?.citations ?? []).map((c) => [c.passage_id, c.support]));
  const busy = compare.isPending || answer.isPending;

  return (
    <div>
      <div className="page-head">
        <h1>Ask</h1>
      </div>
      <p className="page-sub">
        Every question is run through several chunking and retrieval combinations. You see
        which one found the most relevant evidence for <em>your</em> question, the answer it
        produced, what was extracted, and how every citation was checked.
      </p>

      <div className="chat-input">
        <input value={q} placeholder="e.g. Which miRNA is targeted by SRY/Sox9?"
          onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} />
        <button onClick={() => run()} disabled={busy || !q.trim()}>
          {compare.isPending ? "Comparing…" : answer.isPending ? "Answering…" : "Ask"}
        </button>
      </div>
      <fieldset className="seg answer-with" aria-label="Answer with">
        {([["best", "Answer with the best combination"], ["mine", "Answer with my flow"]] as const)
          .map(([v, text]) => (
            <label key={v} className={answerWith === v ? "active" : ""}>
              <input type="radio" name="answer-with" checked={answerWith === v}
                onChange={() => setAnswerWith(v)} />
              {text}
            </label>
          ))}
      </fieldset>

      {compare.isPending && (
        <p className="progress-note">
          Comparing combinations for this question: retrieval + MedCPT judge, no LLM calls…
        </p>
      )}
      {compare.isError && <div className="error-box">{(compare.error as Error).message}</div>}
      {answer.isPending && (
        <p className="progress-note">Writing the answer from the chosen combination (1 LLM call)…</p>
      )}
      {answer.isError && <div className="error-box">{(answer.error as Error).message}</div>}

      {r && answered && (
        <div className="card answer-card">
          <h3>
            Answer{" "}
            <span className={`badge ${r.evidence_status === "sufficient" ? "live" : "mock"}`}>
              evidence: {r.evidence_status}
            </span>{" "}
            <span className={`badge ${r.llm_used ? "live" : "mock"}`}>
              {r.llm_used ? "LLM used" : "LLM off"} · {r.llm_source}
            </span>
          </h3>
          <AnswerText text={r.answer} />
          <div className="built-with">
            <strong>Built with</strong>{" "}
            {answered.is_best ? "the best combination for this question" : "your chosen combination"}:
            <div className="chips">
              {describeConfig(answered.config).map((p) => <span className="chip" key={p}>{p}</span>)}
            </div>
            <span className="hint">
              Evidence score {answered.evidence_score.toFixed(2)} · retrieval{" "}
              {Math.round(answered.latency_ms)} ms · comparison {cmp!.seconds.toFixed(1)} s
            </span>
          </div>
        </div>
      )}

      {cmp && (
        <div className="card">
          <h3>Which combination worked best for this question ({cmp.combos.length} compared)</h3>
          <p className="hint">
            <strong>Evidence score</strong>: {cmp.judge.how}{" "}
            {cmp.gold_available
              ? `This question is in the dataset, so the True columns check each combination against its ${cmp.n_gold} gold passages.`
              : "True scores appear only for questions from the dataset."}{" "}
            Click a row to see its evidence.
          </p>
          {cmp.judge.calibration ? (
            <p className="calibration">
              <strong>How far to trust the judge:</strong> on {cmp.judge.calibration.questions} dev
              questions with gold passages, its pick was a truly best combination{" "}
              {Math.round(cmp.judge.calibration.pick_is_oracle * 100)}% of the time. Picking per
              question gave true nDCG@10 {cmp.judge.calibration.picked_ndcg.toFixed(2)} vs{" "}
              {cmp.judge.calibration.default_ndcg.toFixed(2)} for the default flow (best possible{" "}
              {cmp.judge.calibration.oracle_ndcg.toFixed(2)}).
            </p>
          ) : (
            <p className="calibration">The judge has not been calibrated against gold passages yet.</p>
          )}
          <ComboTable data={cmp} answeredId={answeredId} busy={busy}
            onAnswerWith={(c) => answer.mutate({ question: cmp.question, combo: c })} />
        </div>
      )}

      {answered && (
        <div className="card">
          <h3>What we extracted ({answered.evidence.length} passages sent to the LLM)</h3>
          <EvidenceList items={answered.evidence} support={support} />
        </div>
      )}

      {r && (
        <div className="card">
          <h3>Claims and citation checks</h3>
          <div className="grid cols-3">
            {r.grounding.map((mt) => <MetricCard key={mt.key} metric={mt} />)}
          </div>
          {r.claims.length === 0 && <p className="hint">No claims: the model found nothing citable.</p>}
          {r.claims.map((cl, i) => (
            <div className="claim-row" key={i}>
              <span className={`badge ${cl.support}`}>{cl.support}</span>
              <div style={{ flex: 1 }}>
                <div>{cl.text}</div>
                <div className="hint">cites: {cl.citations.join(", ") || "—"}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      <h2 className="section-title">Strategies available</h2>
      <StrategiesPanel />
    </div>
  );
}
