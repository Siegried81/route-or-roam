// Dev server proxies /api to the FastAPI backend on 8001. Fixed ports so it runs beside
// dreamjob (5173/8000) and grounded-rag (5180/8002); strictPort fails loudly instead of
// silently moving to a port another app owns.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5181,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8001" },
  },
});
