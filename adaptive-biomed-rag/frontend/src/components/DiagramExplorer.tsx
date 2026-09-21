import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import type { ArchitectureDiagram } from "../api/types";
import { MermaidDiagram } from "./MermaidDiagram";

const STATUS_LABEL: Record<string, string> = {
  live: "live",
  building: "index building",
  planned: "planned",
};

// Tabs of architecture diagrams with a legend and a details panel: click any box
// to see what it does, the rules / thresholds in force, and where it lives in code.
export function DiagramExplorer({
  diagrams,
  legend,
}: {
  diagrams: ArchitectureDiagram[];
  legend: { label: string; cls: string }[];
}) {
  const [params, setParams] = useSearchParams();       // ?tab=<id> makes tabs linkable
  const tab = params.get("tab") ?? diagrams[0]?.id;
  const setTab = (id: string) => setParams({ tab: id }, { replace: true });
  const [nodeId, setNodeId] = useState<string | null>(null);
  const [fit, setFit] = useState(true);               // fit to width vs actual size
  const d = diagrams.find((x) => x.id === tab) ?? diagrams[0];
  if (!d) return null;
  const node = nodeId ? d.nodes[nodeId] : undefined;

  return (
    <div>
      <div className="tabs" role="tablist">
        {diagrams.map((x) => (
          <button key={x.id} role="tab" aria-selected={x.id === d.id}
            className={x.id === d.id ? "active" : ""}
            onClick={() => { setTab(x.id); setNodeId(null); }}>
            {x.title}
          </button>
        ))}
      </div>
      <p className="diagram-desc">{d.description}</p>
      <div className="diagram-toolbar">
        <fieldset className="seg" aria-label="Diagram size">
          {([[true, "Fit to width"], [false, "Actual size"]] as const).map(([v, label]) => (
            <label key={label} className={fit === v ? "active" : ""}>
              <input type="radio" name="diagram-size" checked={fit === v} onChange={() => setFit(v)} />
              {label}
            </label>
          ))}
        </fieldset>
        <span className="hint">Click a box for its rules, thresholds and code location.</span>
      </div>
      <div className="legend">
        {legend.map((l) => (
          <span key={l.cls} className="legend-item">
            <span className={`legend-swatch lg-${l.cls}`} /> {l.label}
          </span>
        ))}
      </div>

      <div className="diagram-layout">
        <div className={`diagram-wrap ${fit ? "fit" : "actual"}`}>
          <MermaidDiagram key={d.id} source={d.mermaid} onNodeClick={setNodeId} />
        </div>
        <aside className="card node-detail">
          {node ? (
            <>
              <h3>
                {node.title}{" "}
                <span className={`state ${node.status === "live" ? "ready" : node.status}`}>
                  {STATUS_LABEL[node.status] ?? node.status}
                </span>
              </h3>
              <p>{node.what}</p>
              {node.rules.length > 0 && (
                <>
                  <div className="detail-label">Rules and thresholds</div>
                  <ul className="rules">
                    {node.rules.map((r) => <li key={r}>{r}</li>)}
                  </ul>
                </>
              )}
              {node.source && (
                <>
                  <div className="detail-label">Where in the code</div>
                  <code className="source">{node.source}</code>
                </>
              )}
            </>
          ) : nodeId ? (
            <p className="hint">No extra details for this box.</p>
          ) : (
            <p className="hint">
              Click any box to see what it does, the rules and thresholds it uses, and where it
              lives in the code. Values are read from the running system.
            </p>
          )}
        </aside>
      </div>
    </div>
  );
}
