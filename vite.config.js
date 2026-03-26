import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  base: './',
  plugins: [react(), wasm(), topLevelAwait()],
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs-backend-webgpu',
    ],
    exclude: [
      '@dimforge/rapier2d-compat',
    ]
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  worker: {
    format: 'es',
    plugins: () => [wasm(), topLevelAwait()]
  }
})
