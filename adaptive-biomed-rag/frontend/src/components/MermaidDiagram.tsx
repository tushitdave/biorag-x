import { useEffect, useRef, useState } from "react";
import mermaid from "mermaid";

mermaid.initialize({
  startOnLoad: false,
  theme: "neutral",
  securityLevel: "loose",
  // Natural size (the container scrolls) so labels stay readable on large diagrams.
  flowchart: { curve: "basis", htmlLabels: true, useMaxWidth: false },
});

let idCounter = 0;

// Renders a Mermaid diagram from source. After render, wires click handlers on
// nodes: the node id (mermaid data-id / element id) is passed to onNodeClick so
// the parent can show that stage/component's details.
export function MermaidDiagram({
  source,
  onNodeClick,
}: {
  source: string;
  onNodeClick?: (nodeId: string) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [idBase] = useState(() => `mmd-${idCounter++}`);

  useEffect(() => {
    let cancelled = false;
    async function render() {
      try {
        const { svg } = await mermaid.render(`${idBase}-svg`, source);
        if (cancelled || !ref.current) return;
        ref.current.innerHTML = svg;
        setError(null);

        if (onNodeClick) {
          // Mermaid flowchart nodes render as <g class="node" id="flowchart-<id>-N">
          const nodes = ref.current.querySelectorAll<SVGGElement>("g.node");
          nodes.forEach((node) => {
            node.classList.add("node-clickable");
            node.addEventListener("click", () => {
              const raw = node.id || "";
              // id looks like "flowchart-ingest-3" -> extract the middle token(s)
              const match = raw.match(/^flowchart-(.+?)-\d+$/);
              const nodeId = match ? match[1] : raw;
              onNodeClick(nodeId);
            });
          });
        }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    }
    render();
    return () => {
      cancelled = true;
    };
  }, [source, idBase, onNodeClick]);

  if (error) {
    return <div className="error-box">Diagram render error: {error}</div>;
  }
  return <div className="diagram" ref={ref} />;
}
