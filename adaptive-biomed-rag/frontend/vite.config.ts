import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: forward /api/* to the FastAPI backend on :8000 so the browser
// talks same-origin (no CORS headaches). The client uses base "/api".
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
