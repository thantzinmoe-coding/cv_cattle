import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/predict': {
        target: 'http://localhost:8000',
        ws: true,
        // A browser navigation is an SPA route, not a FastAPI request.
        // API fetches and WebSocket upgrades continue through the proxy.
        bypass(request) {
          if (
            request.method === 'GET'
            && request.headers.upgrade !== 'websocket'
            && request.headers.accept?.includes('text/html')
          ) {
            return '/index.html'
          }
        },
      },
      '/outputs':  'http://localhost:8000',
      '/health':   'http://localhost:8000',
    },
  },
})
