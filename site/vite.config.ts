import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tanstackRouter({
      target: 'react',
      autoCodeSplitting: true,
    }),
    react(),
  ],
  server: {
    host: true,
  },
  // `public/` is reserved for the deployed/built output (wrangler.toml at
  // the repo root points there), so Vite's own source static-passthrough
  // dir lives at `static/` instead - keeps the two from colliding
  publicDir: 'static',
  build: {
    outDir: 'public',
  },
})
