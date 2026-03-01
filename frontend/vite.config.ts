import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/audio": "http://localhost:8000",
      "/processing": "http://localhost:8000",
      "/clustering": "http://localhost:8000",
      "/admin": "http://localhost:8000",
    },
  },
  build: {
    outDir: "../src/humpback/static/dist",
    emptyOutDir: true,
  },
});
