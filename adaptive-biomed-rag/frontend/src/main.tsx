import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { ExplainProvider } from "./components/ExplainDrawer";
import { ExperimentProvider } from "./components/ExperimentState";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ExperimentProvider>
        <ExplainProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ExplainProvider>
      </ExperimentProvider>
    </QueryClientProvider>
  </StrictMode>
);
