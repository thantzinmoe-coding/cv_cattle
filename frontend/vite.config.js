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
      '/train': 'http://localhost:8000',
      '/predict': 'http://localhost:8000',
      '/evaluate': 'http://localhost:8000',
      '/outputs': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
