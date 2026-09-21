import { NavLink, Route, Routes, Navigate } from "react-router-dom";
import { ControlBar } from "./components/ControlBar";
import { OverviewPage } from "./pages/OverviewPage";
import { PipelinePage } from "./pages/PipelinePage";
import { ChatPage } from "./pages/ChatPage";
import { SystemPage } from "./pages/SystemPage";

const NAV = [
  { to: "/chat", label: "Ask" },
  { to: "/overview", label: "Overview" },
  { to: "/pipeline", label: "Process flow" },
  { to: "/system", label: "System design" },
];

export default function App() {
  return (
    <div className="shell">
      <nav className="nav">
        <div className="brand">BioRAG-X</div>
        <div className="tagline">Adaptive Biomedical RAG</div>
        {NAV.map((n) => (
          <NavLink
            key={n.to}
            to={n.to}
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            {n.label}
          </NavLink>
        ))}
        <div className="spacer" />
        <div className="foot">
          LLM budget-capped · offline-first.
          <br />
          Non-LLM paths run fully offline.
        </div>
      </nav>

      <div className="main">
        <ControlBar />
        <div className="content">
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/pipeline" element={<PipelinePage />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/system" element={<SystemPage />} />
            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}
