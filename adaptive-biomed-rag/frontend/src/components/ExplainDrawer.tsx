import { createContext, useContext, useState, type ReactNode } from "react";
import type { MetricValue } from "../api/types";

// Global click-to-explain drawer. Any metric card / table cell / chip can call
// explain(metric) to open a side drawer with the plain-language definition,
// CI and baseline delta from the MetricValue contract.
interface ExplainCtx {
  explain: (m: MetricValue) => void;
}

const Ctx = createContext<ExplainCtx>({ explain: () => {} });

export function useExplain() {
  return useContext(Ctx);
}

function fmt(v: number, unit: string): string {
  if (unit === "%") return `${v.toFixed(1)}%`;
  if (unit === "ms") return `${v.toFixed(0)} ms`;
  if (unit === "USD") return `$${v.toFixed(4)}`;
  if (unit === "/5") return `${v.toFixed(2)} / 5`;
  // unitless ratios
  return Math.abs(v) <= 1 ? v.toFixed(3) : v.toLocaleString();
}

export function ExplainProvider({ children }: { children: ReactNode }) {
  const [metric, setMetric] = useState<MetricValue | null>(null);

  return (
    <Ctx.Provider value={{ explain: setMetric }}>
      {children}
      {metric && (
        <>
          <div className="drawer-backdrop" onClick={() => setMetric(null)} />
          <aside className="drawer" role="dialog" aria-label={`Explain ${metric.label}`}>
            <button className="close" onClick={() => setMetric(null)} aria-label="Close">
              ×
            </button>
            <h2>{metric.label}</h2>
            <div className="metric-card" style={{ boxShadow: "none", border: "none", padding: 0, cursor: "default" }}>
              <span className="value">{fmt(metric.value, metric.unit)}</span>
            </div>
            <dl>
              <dt>What it means</dt>
              <dd>{metric.definition || "No definition provided."}</dd>

              <dt>Direction</dt>
              <dd>{metric.higher_is_better ? "Higher is better." : "Lower is better."}</dd>

              {metric.baseline_delta != null && (
                <>
                  <dt>Δ vs baseline</dt>
                  <dd className={metric.baseline_delta >= 0 ? "delta up" : "delta down"}>
                    {metric.baseline_delta >= 0 ? "+" : ""}
                    {fmt(metric.baseline_delta, metric.unit)}
                  </dd>
                </>
              )}

              {metric.ci_low != null && metric.ci_high != null && (
                <>
                  <dt>95% confidence interval</dt>
                  <dd>
                    {fmt(metric.ci_low, metric.unit)} — {fmt(metric.ci_high, metric.unit)}
                  </dd>
                </>
              )}

              <dt>Source</dt>
              <dd>
                <span className={`badge ${metric.source === "mock" ? "mock" : "live"}`}>
                  {metric.source}
                </span>
              </dd>
            </dl>
          </aside>
        </>
      )}
    </Ctx.Provider>
  );
}

export { fmt as formatMetric };
