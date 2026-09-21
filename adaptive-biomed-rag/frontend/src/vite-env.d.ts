/// <reference types="vite/client" />

declare module "plotly.js-dist-min" {
  const Plotly: {
    newPlot: (
      root: HTMLElement,
      data: unknown[],
      layout?: unknown,
      config?: unknown
    ) => Promise<void>;
    react: (
      root: HTMLElement,
      data: unknown[],
      layout?: unknown,
      config?: unknown
    ) => Promise<void>;
    purge: (root: HTMLElement) => void;
    Plots: { resize: (root: HTMLElement) => void };
  };
  export default Plotly;
}
