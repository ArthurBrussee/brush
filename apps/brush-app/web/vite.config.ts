import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

// `BRUSH_BASE_PATH` lets us deploy under a sub-path (e.g. `/brush-demo` on
// GitHub Pages) without hard-coding it.
const base = process.env.BRUSH_BASE_PATH || '/';

export default defineConfig({
  base,
  plugins: [react(), wasm(), topLevelAwait()],
  build: { target: 'es2022' },
  server: {
    port: 5173,
    strictPort: true,
    fs: {
      // Allow Vite to read the generated `pkg/` next to this config.
      allow: ['..'],
    },
  },
});
