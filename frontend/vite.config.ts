import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy: the browser calls `/api/*` and Vite forwards to FastAPI,
// so no API key or backend URL is baked into the frontend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
