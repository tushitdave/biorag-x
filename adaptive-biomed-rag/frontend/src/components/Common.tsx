import type { Sourced } from "../api/client";

export function MockBadge({ src }: { src: Sourced<unknown> }) {
  if (!src.isMock) return <span className="badge live">live</span>;
  return (
    <span className="badge mock" title={src.error}>
      mock data (backend offline)
    </span>
  );
}

export function Loading() {
  return <div className="loading">Loading…</div>;
}
