# brush-js

JavaScript / WebAssembly bindings for [Brush](../../README.md) Gaussian-splat
training. Hands its GPU buffers (transforms, SH coefficients, opacities) back
to JS as `GPUBuffer`s — bind them straight into your own WebGPU pipelines, no
copies.

## Demo

A minimal Vite + WebGPU demo lives under `web/`. From the repo root:

```sh
npm install
npm run dev:lib
```

The demo expects a Chromium-based browser (it uses `showDirectoryPicker`).
Pick a folder containing a Brush dataset (or a single `.ply`) to start.

## Sketch

```js
import { BrushApp } from 'brush-js';

const app = new BrushApp();
await app.init(); // or app.initExisting(adapter, device, queue) to share a GPUDevice

const dir = await window.showDirectoryPicker();
const training = app.startTrainingFromDirectory(dir, async (initialConfig) => initialConfig);

while (true) {
  const msgs = await training.trainSteps(5);
  if (msgs.length === 0) break;
  // inspect msgs (TrainStep, RefineStep, EvalResult, ...)
}
```

`training.currentSplats().buffers()` gives you the live GPU buffers; see
`web/src/main.ts` for an end-to-end renderer.
