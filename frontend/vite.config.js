import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev proxy target used by the Vite dev server.
// - Local dev: defaults to http://127.0.0.1:8000
// - Docker compose: set VITE_PROXY_TARGET=http://backend:8000 (service name)
const proxyTarget = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API calls so the browser can call `/api/...` on the same origin.
    // This avoids CORS issues and removes the need to hardcode an API base.
    proxy: {
      '/api': {
        target: proxyTarget,
        changeOrigin: true
      }
    }
  }
})
