import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { ExplainableChart } from "../api/types";

const PALETTE = ["#2c6e8f", "#7fae4b", "#c98a2b", "#8a5fa8", "#3aa6a0"];

// Builds Plotly traces from an ExplainableChart contract object, covering the
// kinds the backend emits: bar | grouped_bar | radar | scatter | line.
function toTraces(chart: ExplainableChart): unknown[] {
  const { kind, series } = chart;

  if (kind === "radar") {
    return series.map((s, i) => ({
      type: "scatterpolar",
      name: s.name,
      r: [...s.y, s.y[0]],
      theta: [...s.x, s.x[0]],
      fill: "toself",
      line: { color: PALETTE[i % PALETTE.length] },
    }));
  }

  if (kind === "scatter") {
    return series.map((s, i) => ({
      type: "scatter",
      mode: "markers+text",
      name: s.name,
      x: s.x.map(Number),
      y: s.y,
      text: s.x,
      textposition: "top center",
      marker: { size: 12, color: PALETTE[i % PALETTE.length] },
    }));
  }

  if (kind === "line") {
    return series.map((s, i) => ({
      type: "scatter",
      mode: "lines+markers",
      name: s.name,
      x: s.x,
      y: s.y,
      line: { color: PALETTE[i % PALETTE.length] },
    }));
  }

  // bar | grouped_bar
  return series.map((s, i) => ({
    type: "bar",
    name: s.name,
    x: s.x,
    y: s.y,
    marker: { color: PALETTE[i % PALETTE.length] },
  }));
}

function layoutFor(chart: ExplainableChart): Record<string, unknown> {
  const base: Record<string, unknown> = {
    margin: { t: 10, r: 16, b: 46, l: 52 },
    height: 300,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, Segoe UI, sans-serif", size: 12, color: "#1c2733" },
    legend: { orientation: "h", y: -0.25 },
    barmode: chart.kind === "grouped_bar" ? "group" : "stack",
  };
  if (chart.kind === "radar") {
    return {
      ...base,
      polar: { radialaxis: { visible: true, range: [0, 1] } },
    };
  }
  return {
    ...base,
    xaxis: { title: chart.x_label || undefined, automargin: true },
    yaxis: { title: chart.y_label || undefined, automargin: true },
  };
}

// Renders one chart with its explanation (what/why) and one-line takeaway.
export function ChartCard({ chart }: { chart: ExplainableChart }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    Plotly.react(el, toTraces(chart), layoutFor(chart), {
      displayModeBar: false,
      responsive: true,
    });
    return () => {
      if (el) Plotly.purge(el);
    };
  }, [chart]);

  return (
    <div className="card">
      <h3>{chart.title}</h3>
      <p className="explanation">{chart.explanation}</p>
      <div ref={ref} />
      {chart.takeaway && <div className="takeaway">💡 {chart.takeaway}</div>}
    </div>
  );
}
