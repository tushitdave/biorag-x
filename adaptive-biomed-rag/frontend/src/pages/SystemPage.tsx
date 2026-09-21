import { useQuery } from "@tanstack/react-query";
import { api, architecture } from "../api/client";
import { DiagramExplorer } from "../components/DiagramExplorer";
import { MetricCard } from "../components/MetricCard";
import { Loading } from "../components/Common";

export function SystemPage() {
  const arch = useQuery({ queryKey: ["architecture"], queryFn: architecture, refetchInterval: 30000 });
  const { data } = useQuery({ queryKey: ["system"], queryFn: api.system });

  return (
    <div>
      <div className="page-head">
        <h1>System design</h1>
      </div>
      <p className="page-sub">
        The components of BioRAG-X, what calls what, where data lives, and how the system
        keeps answering when a part fails.
      </p>
      {arch.error && <div className="error-box">{(arch.error as Error).message}</div>}
      {!arch.data ? <Loading /> : (
        <DiagramExplorer diagrams={arch.data.system} legend={arch.data.legend} />
      )}

      {data && (
        <div className="grid cols-2" style={{ marginTop: 18 }}>
          <div className="card">
            <h3>Versions</h3>
            <table className="tbl">
              <tbody>
                {data.data.versions.map((v) => (
                  <tr key={v.component}>
                    <td>{v.component}</td>
                    <td>{v.version}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3 className="section-title">Observability</h3>
            <div className="grid cols-2">
              {data.data.observability.map((mt) => <MetricCard key={mt.key} metric={mt} />)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
