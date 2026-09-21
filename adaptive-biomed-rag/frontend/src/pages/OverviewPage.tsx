import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { MetricCard } from "../components/MetricCard";
import { Loading, MockBadge } from "../components/Common";
import type { ExperimentDNA } from "../api/types";

function DnaChips({ dna }: { dna: ExperimentDNA }) {
  const entries: [string, string][] = [
    ["chunking", dna.chunking],
    ["embedding", dna.embedding],
    ["ann", dna.ann],
    ["retrieval", dna.retrieval],
    ["fusion", dna.fusion],
    ["reranker", dna.reranker],
    ["agent", dna.agent],
    ["evidence", dna.evidence],
    ["generator", dna.generator],
    ["citation", dna.citation],
  ];
  return (
    <div className="chips">
      {entries.map(([k, v]) => (
        <span className="chip" key={k}>
          <span className="k">{k}:</span>
          {v}
        </span>
      ))}
    </div>
  );
}

export function OverviewPage() {
  const { data } = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  if (!data) return <Loading />;
  const ov = data.data;

  return (
    <div>
      <div className="page-head">
        <h1>Overview</h1>
        <MockBadge src={data} />
      </div>
      <p className="page-sub">
        Headline retrieval and grounding metrics for experiment{" "}
        <strong>{ov.experiment.experiment_id}</strong>. Click any card to see what it means.
      </p>

      <div className="grid cols-4">
        {ov.metrics.map((mt) => (
          <MetricCard key={mt.key} metric={mt} />
        ))}
      </div>

      <h2 className="section-title">Experiment DNA</h2>
      <div className="card">
        <DnaChips dna={ov.experiment} />
      </div>

      <div className="grid cols-3" style={{ marginTop: 14 }}>
        <div className="card">
          <div className="metric-card" style={{ cursor: "default", boxShadow: "none", border: "none", padding: 0 }}>
            <span className="value">{ov.corpus_passages.toLocaleString()}</span>
            <div className="label">Corpus passages</div>
          </div>
        </div>
        <div className="card">
          <div className="metric-card" style={{ cursor: "default", boxShadow: "none", border: "none", padding: 0 }}>
            <span className="value">{ov.questions.toLocaleString()}</span>
            <div className="label">Benchmark questions</div>
          </div>
        </div>
        <div className="card">
          <div className="metric-card" style={{ cursor: "default", boxShadow: "none", border: "none", padding: 0 }}>
            <span className="value">{ov.gold_relationships.toLocaleString()}</span>
            <div className="label">Gold relationships</div>
          </div>
        </div>
      </div>

      <h2 className="section-title">Explore</h2>
      <div className="grid cols-4">
        <Link className="card" to="/pipeline"><h3>Pipeline →</h3><p className="explanation">Walk the retrieval workflow stage by stage.</p></Link>
        <Link className="card" to="/chat"><h3>Chat →</h3><p className="explanation">Ask a question and inspect claim-level citations.</p></Link>
        <Link className="card" to="/evaluation"><h3>Evaluation →</h3><p className="explanation">Compare configurations with explainable charts.</p></Link>
        <Link className="card" to="/system"><h3>System Design →</h3><p className="explanation">See the component architecture and observability.</p></Link>
      </div>
    </div>
  );
}
