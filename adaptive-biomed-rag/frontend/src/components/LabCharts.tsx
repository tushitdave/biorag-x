import { useState } from "react";
import type { RunSummary } from "../api/types";
import { describeConfig } from "./labLabels";

// Reference-palette roles (dataviz skill): one series hue, recessive chrome,
// status colours only for better/worse and always paired with a text label.
const SERIES = "#2a78d6";
const GRID = "#e1e0d9";
const AXIS = "#c3c2b7";
const MUTED = "#898781";
const INK = "#0b0b0b";
const INK_2 = "#52514e";
const GOOD = "#0ca30c";
const CRITICAL = "#d03b3b";

// --------------------------------------------------------------------------- //
// Quality vs latency scatter with the Pareto front (best trade-offs).
// --------------------------------------------------------------------------- //
const W = 960;
const H = 320;
const M = { l: 56, r: 24, t: 16, b: 46 };
const LABEL_GAP = 36; // skip a label whose anchor is this close to one already placed

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let d = Math.floor(Math.log10(lo)); d <= Math.ceil(Math.log10(hi)); d++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** d;
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
}

/** Runs no other run beats on both quality and speed, fastest first. */
function paretoFront(points: { id: string; x: number; y: number }[]) {
  const sorted = [...points].sort((a, b) => a.x - b.x || b.y - a.y);
  const front: typeof points = [];
  let best = -Infinity;
  for (const p of sorted) {
    if (p.y > best) {
      front.push(p);
      best = p.y;
    }
  }
  return front;
}

export function ParetoChart({
  runs,
  a,
  b,
  onPick,
}: {
  runs: RunSummary[];
  a?: string;
  b?: string;
  onPick: (id: string) => void;
}) {
  const [hover, setHover] = useState<string | null>(null);
  const pts = runs
    .filter((r) => r.headline["ndcg@10"] !== undefined && r.headline.latency_p50 !== undefined)
    .map((r) => ({ id: r.id, x: Math.max(1, r.headline.latency_p50), y: r.headline["ndcg@10"], run: r }));
  if (pts.length === 0) return <p className="hint">Finished runs will appear here.</p>;

  const xLo = 10 ** Math.floor(Math.log10(Math.min(...pts.map((p) => p.x))));
  const xHi = 10 ** Math.ceil(Math.log10(Math.max(...pts.map((p) => p.x)) * 1.01));
  // Round y ticks: the smallest step that gives at most 6 gridlines.
  const yMin = Math.min(...pts.map((p) => p.y)) - 0.02;
  const yMax = Math.max(...pts.map((p) => p.y)) + 0.02;
  const yStep = [0.01, 0.02, 0.025, 0.05, 0.1, 0.2].find((s) => (yMax - yMin) / s <= 5) ?? 0.25;
  const yLo = Math.max(0, Math.floor(yMin / yStep) * yStep);
  const yHi = Math.min(1, Math.ceil(yMax / yStep) * yStep);
  const sx = (x: number) =>
    M.l + ((Math.log10(x) - Math.log10(xLo)) / (Math.log10(xHi) - Math.log10(xLo))) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - ((y - yLo) / (yHi - yLo || 1)) * (H - M.t - M.b);
  const yTicks = Array.from({ length: Math.round((yHi - yLo) / yStep) + 1 }, (_, i) => yLo + i * yStep);
  const front = paretoFront(pts);
  const hovered = pts.find((p) => p.id === hover);

  // Selective direct labels: A and B first, then Pareto-front runs, skipping any
  // that would collide with a label already placed (the tooltip and the runs
  // table still carry every value).
  const labelled = new Set<string>();
  const placed: [number, number][] = [];
  for (const p of [...pts.filter((q) => q.id === a || q.id === b), ...front]) {
    if (labelled.has(p.id)) continue;
    const [x, y] = [sx(p.x), sy(p.y)];
    if (placed.some(([px, py]) => Math.hypot(px - x, py - y) < LABEL_GAP)) continue;
    labelled.add(p.id);
    placed.push([x, y]);
  }

  return (
    <div className="viz">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="nDCG@10 versus median latency per run">
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={sy(t)} y2={sy(t)} stroke={GRID} strokeWidth={1} />
            <text x={M.l - 8} y={sy(t) + 4} textAnchor="end" fontSize={11} fill={MUTED}>
              {t.toFixed(yStep === 0.025 ? 3 : 2)}
            </text>
          </g>
        ))}
        {logTicks(xLo, xHi).map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={H - M.b} y2={H - M.b + 4} stroke={AXIS} />
            <text x={sx(t)} y={H - M.b + 17} textAnchor="middle" fontSize={11} fill={MUTED}>
              {t >= 1000 ? `${t / 1000}s` : `${t}`}
            </text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} stroke={AXIS} />
        <text x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle" fontSize={11} fill={INK_2}>
          Median retrieval time per question (ms, log scale) - left is faster
        </text>
        <text transform={`translate(13 ${(M.t + H - M.b) / 2}) rotate(-90)`} textAnchor="middle"
          fontSize={11} fill={INK_2}>
          nDCG@10 - up is better
        </text>

        {front.length > 1 && (
          <polyline
            points={front.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
            fill="none" stroke={MUTED} strokeWidth={2} strokeDasharray="5 4"
          />
        )}

        {pts.map((p) => {
          const tag = p.id === a ? "A" : p.id === b ? "B" : null;
          return (
            <g key={p.id}>
              <circle cx={sx(p.x)} cy={sy(p.y)} r={tag ? 6.5 : 4.5} fill={SERIES}
                stroke={tag ? INK : "#fcfcfb"} strokeWidth={2}
                opacity={hover && hover !== p.id ? 0.55 : 1} />
              {labelled.has(p.id) && (
                <text x={sx(p.x) + 10} y={sy(p.y) - 8} fontSize={11}
                  fill={tag ? INK : INK_2} fontWeight={tag ? 700 : 400}>
                  {tag ? `${tag} · ${p.id}` : p.id}
                </text>
              )}
              <circle cx={sx(p.x)} cy={sy(p.y)} r={12} fill="transparent" tabIndex={0}
                style={{ cursor: "pointer", outline: "none" }}
                onMouseEnter={() => setHover(p.id)} onMouseLeave={() => setHover(null)}
                onFocus={() => setHover(p.id)} onBlur={() => setHover(null)}
                onClick={() => onPick(p.id)}
                onKeyDown={(e) => e.key === "Enter" && onPick(p.id)}>
                <title>{`${p.id}: nDCG@10 ${p.y.toFixed(3)}, ${Math.round(p.x)} ms`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
      {hovered && (
        <div className="viz-tooltip"
          style={{ left: `${(sx(hovered.x) / W) * 100}%`, top: `${(sy(hovered.y) / H) * 100}%` }}>
          <strong>{hovered.y.toFixed(3)}</strong> nDCG@10 · <strong>{Math.round(hovered.x)} ms</strong>
          <div className="viz-tooltip-sub">
            {hovered.run.id} · {hovered.run.name}
          </div>
          <div className="viz-tooltip-sub">{describeConfig(hovered.run.config).join(" · ")}</div>
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- //
// One paired delta with its 95% CI, drawn on its own symmetric scale around 0.
// --------------------------------------------------------------------------- //
export type Verdict = "better" | "worse" | "unclear";

export function verdictOf(delta: number, significant: boolean, higherIsBetter: boolean): Verdict {
  if (!significant) return "unclear";
  return (delta > 0) === higherIsBetter ? "better" : "worse";
}

export function DeltaInterval({
  delta,
  lo,
  hi,
  verdict,
}: {
  delta: number;
  lo: number;
  hi: number;
  verdict: Verdict;
}) {
  const w = 150;
  const h = 18;
  const span = Math.max(Math.abs(lo), Math.abs(hi), Math.abs(delta), 1e-9) * 1.1;
  const x = (v: number) => w / 2 + (v / span) * (w / 2 - 6);
  const color = verdict === "better" ? GOOD : verdict === "worse" ? CRITICAL : MUTED;
  return (
    <svg width={w} height={h} role="img" aria-label={`difference ${delta}, 95% CI ${lo} to ${hi}`}>
      <line x1={w / 2} x2={w / 2} y1={1} y2={h - 1} stroke={AXIS} strokeWidth={1} />
      <line x1={x(lo)} x2={x(hi)} y1={h / 2} y2={h / 2} stroke={color} strokeWidth={2}
        strokeLinecap="round" />
      <circle cx={x(delta)} cy={h / 2} r={4.5} fill={color} stroke="#fcfcfb" strokeWidth={2} />
    </svg>
  );
}

// --------------------------------------------------------------------------- //
// Inline magnitude bar for 0..1 metrics in tables.
// --------------------------------------------------------------------------- //
export function InlineBar({ value }: { value: number }) {
  return (
    <span className="inline-bar" aria-hidden>
      <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: SERIES }} />
    </span>
  );
}
