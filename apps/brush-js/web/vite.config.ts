import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  build: {
    target: 'es2022',
  },
  server: {
    port: 5174,
    strictPort: true,
    fs: {
      // Allow Vite to read the generated `pkg/` next to this config.
      allow: ['..'],
    },
  },
});
