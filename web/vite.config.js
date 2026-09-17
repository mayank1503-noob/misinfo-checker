import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The Flask app (backend/api/flask_app.py) serves ../frontend as static
// files: index.html at "/" and everything else by name. So that is where
// the build goes, and assets are referenced relatively ("./assets/...")
// so the same bundle works opened off disk as it does over HTTP.
//
// In dev, Vite serves the page on :5173 and proxies /api to Flask on
// :5000, which keeps the page and the API on one origin there too — the
// reason the client never needs CORS or an API base URL.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../frontend',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.API_ORIGIN || 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})
