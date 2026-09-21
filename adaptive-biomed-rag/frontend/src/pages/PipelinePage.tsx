import { useQuery } from "@tanstack/react-query";
import { architecture } from "../api/client";
import { DiagramExplorer } from "../components/DiagramExplorer";
import { Loading } from "../components/Common";

export function PipelinePage() {
  const { data, error } = useQuery({
    queryKey: ["architecture"],
    queryFn: architecture,
    refetchInterval: 30000, // build status and counts change while indexes build
  });

  return (
    <div>
      <div className="page-head">
        <h1>Process flow</h1>
      </div>
      <p className="page-sub">
        How BioRAG-X builds its indexes and answers a question, step by step, with the exact
        rules and thresholds it uses. Every number is read from the running system.
      </p>
      {error && <div className="error-box">{(error as Error).message}</div>}
      {!data ? <Loading /> : <DiagramExplorer diagrams={data.process} legend={data.legend} />}
    </div>
  );
}
