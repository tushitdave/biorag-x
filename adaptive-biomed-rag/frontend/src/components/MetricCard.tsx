import type { MetricValue } from "../api/types";
import { useExplain, formatMetric } from "./ExplainDrawer";

// A click-to-explain metric card. Clicking opens the global explain drawer.
export function MetricCard({ metric }: { metric: MetricValue }) {
  const { explain } = useExplain();
  return (
    <div
      className="card metric-card"
      onClick={() => explain(metric)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && explain(metric)}
      title="Click to explain"
    >
      <span className="value">{formatMetric(metric.value, metric.unit)}</span>
      <div className="label">{metric.label}</div>
      {metric.baseline_delta != null && (
        <div className={`delta ${metric.baseline_delta >= 0 ? "up" : "down"}`}>
          {metric.baseline_delta >= 0 ? "▲ +" : "▼ "}
          {formatMetric(metric.baseline_delta, metric.unit)} vs baseline
        </div>
      )}
      <div className="hint">Click to explain →</div>
    </div>
  );
}
