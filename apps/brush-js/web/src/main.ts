import { Button, Container, InfoBox, Label, NumericInput, Panel } from '@playcanvas/pcui';
import '@playcanvas/pcui/styles';

import {
  BrushApp,
  BrushMessageKind,
  type BrushMessage,
  type Training,
} from '../pkg/brush_js';

import { PointRenderer, type Camera } from './point_renderer';

// -------------------------------------------------------------------------------------------
// Sidebar UI
// -------------------------------------------------------------------------------------------

const root = new Container({ flex: true });
document.getElementById('ui')!.appendChild(root.dom);

function statRow(parent: Container, label: string): Label {
  const row = new Container({ flex: true, flexDirection: 'row', flexWrap: 'wrap' });
  row.dom.style.justifyContent = 'space-between';
  row.append(new Label({ text: label }));
  const value = new Label({ text: '—' });
  row.append(value);
  parent.append(row);
  return value;
}

const stats = (() => {
  const panel = new Panel({ headerText: 'Stats' });
  root.append(panel);
  const r = {
    status: statRow(panel, 'Status'),
    iter: statRow(panel, 'Iter'),
    splats: statRow(panel, 'Splats'),
    elapsed: statRow(panel, 'Elapsed'),
    stepsPerSec: statRow(panel, 'Steps/s'),
    psnr: statRow(panel, 'PSNR'),
    ssim: statRow(panel, 'SSIM'),
    trainViews: statRow(panel, 'Train views'),
    evalViews: statRow(panel, 'Eval views'),
  };
  r.status.text = 'idle';
  return r;
})();

const logEl = (() => {
  const panel = new Panel({ headerText: 'Log' });
  root.append(panel);
  const el = document.createElement('pre');
  el.style.cssText = `
    margin: 0;
    font: 12px/1.5 "JetBrains Mono", "SF Mono", ui-monospace, monospace;
    color: #c8ccd6;
    background: #0a0c10;
    border-radius: 4px;
    padding: 10px;
    max-height: 220px;
    overflow: auto;
    white-space: pre-wrap;
  `;
  panel.dom.appendChild(el);
  return el;
})();

function log(line: string) {
  const ts = new Date().toLocaleTimeString();
  logEl.textContent += `[${ts}] ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
}

const actionPanel = new Panel({ headerText: 'Train' });
root.append(actionPanel);

const trainBtn = new Button({ text: 'Pick folder…' });
const pauseBtn = new Button({ text: 'Pause', enabled: false });
const cancelBtn = new Button({ text: 'Cancel', enabled: false });
const trainRow = new Container({ flex: true, flexDirection: 'row' });
trainRow.dom.style.cssText = 'gap: 8px; flex-wrap: wrap;';
trainRow.append(trainBtn);
trainRow.append(pauseBtn);
trainRow.append(cancelBtn);
actionPanel.append(trainRow);

if (!('showDirectoryPicker' in window)) {
  actionPanel.append(
    new InfoBox({
      icon: 'E218',
      title: 'Directory picker unsupported',
      text: "This browser doesn't expose `showDirectoryPicker`. Open the page in a Chromium-based browser on localhost or HTTPS.",
    }),
  );
}

// -------------------------------------------------------------------------------------------
// Brush runtime
// -------------------------------------------------------------------------------------------

let app: BrushApp | null = null;
let training: Training | null = null;
let renderer: PointRenderer | null = null;
let device: GPUDevice | null = null;
const viewerWrap = document.getElementById('viewer-wrap') as HTMLDivElement;
const viewerCanvas = document.getElementById('viewer') as HTMLCanvasElement;

async function ensureApp(): Promise<BrushApp> {
  if (app) return app;
  stats.status.text = 'initializing…';
  log('Initializing Brush runtime');

  if (!('gpu' in navigator)) {
    throw new Error('navigator.gpu not available — this page needs a WebGPU-capable browser');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('Could not get a WebGPU adapter');

  // Brush needs the same adapter capabilities it would request itself when
  // owning its device — notably `subgroups` for the backward kernels, and
  // the maxed-out limits for large vertex / storage buffers.
  const features = [...(adapter.features as unknown as Iterable<GPUFeatureName>)].filter(
    (f) => f !== 'mappable-primary-buffers',
  );
  const requiredLimits: Record<string, number> = {};
  for (const k in adapter.limits) {
    const v = (adapter.limits as unknown as Record<string, number>)[k];
    if (typeof v === 'number') requiredLimits[k] = v;
  }
  device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
  log('Acquired host GPUDevice — handing it to Brush');

  const a = new BrushApp();
  a.initExisting(adapter, device, device.queue);
  app = a;
  stats.status.text = 'ready';

  renderer = new PointRenderer(device, viewerCanvas);
  renderLoop();
  return a;
}

// -------------------------------------------------------------------------------------------
// Config popup
// -------------------------------------------------------------------------------------------

interface ConfigDoc {
  // The field names match `TrainStreamConfig`'s flattened JSON shape (kebab-case).
  // We only edit a handful in the popup and pass the rest through unchanged.
  [k: string]: unknown;
  'total-train-iters': number;
  'max-resolution': number;
  'sh-degree': number;
  'seed': number;
  'eval-every': number;
}

function showConfigPopup(initial: ConfigDoc): Promise<ConfigDoc | null> {
  return new Promise((resolve) => {
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed; top: 0; right: 0; bottom: 0; left: 0;
      width: 100vw; height: 100vh;
      background: rgba(0, 0, 0, 0.6);
      display: flex; align-items: center; justify-content: center;
      z-index: 1000;
    `;

    const dialog = document.createElement('div');
    dialog.style.cssText = `
      width: min(440px, 92vw);
      background: #1a1d24;
      border: 1px solid #2a2f3a;
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5);
      box-sizing: border-box;
    `;

    const title = new Label({ text: 'Training settings' });
    title.dom.style.cssText = 'font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #e6e8ee; display: block;';
    dialog.appendChild(title.dom);

    const inputs = {
      'total-train-iters': new NumericInput({
        value: initial['total-train-iters'],
        min: 10,
        precision: 0,
      }),
      'max-resolution': new NumericInput({
        value: initial['max-resolution'],
        min: 64,
        precision: 0,
      }),
      'sh-degree': new NumericInput({
        value: initial['sh-degree'],
        min: 0,
        max: 3,
        precision: 0,
      }),
      'seed': new NumericInput({
        value: initial['seed'],
        min: 0,
        precision: 0,
      }),
      'eval-every': new NumericInput({
        value: initial['eval-every'],
        min: 0,
        precision: 0,
      }),
    } as const;

    const labels: Record<keyof typeof inputs, string> = {
      'total-train-iters': 'Total iterations',
      'max-resolution': 'Max resolution',
      'sh-degree': 'SH degree',
      'seed': 'Seed',
      'eval-every': 'Eval every',
    };

    for (const key of Object.keys(inputs) as (keyof typeof inputs)[]) {
      const row = new Container({ flex: true, flexDirection: 'row' });
      row.dom.style.cssText = 'gap: 8px; align-items: center; margin-bottom: 8px;';
      const label = new Label({ text: labels[key] });
      label.dom.style.cssText = 'min-width: 140px; color: #c8ccd6;';
      const input = inputs[key];
      // PCUI's NumericInput collapses to width:0 unless the parent flex
      // container grants it space — `flex: 1; min-width: 0` does exactly that.
      input.dom.style.flex = '1';
      input.dom.style.minWidth = '0';
      row.append(label);
      row.append(input);
      dialog.appendChild(row.dom);
    }

    const buttons = new Container({ flex: true, flexDirection: 'row' });
    buttons.dom.style.cssText = 'gap: 8px; justify-content: flex-end; margin-top: 12px;';
    const cancelBtn = new Button({ text: 'Cancel' });
    const startBtn = new Button({ text: 'Start training' });
    buttons.append(cancelBtn);
    buttons.append(startBtn);
    dialog.appendChild(buttons.dom);

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    const cleanup = () => overlay.remove();
    cancelBtn.on('click', () => {
      cleanup();
      resolve(null);
    });
    startBtn.on('click', () => {
      const out: ConfigDoc = { ...initial };
      for (const key of Object.keys(inputs) as (keyof typeof inputs)[]) {
        out[key] = inputs[key].value;
      }
      cleanup();
      resolve(out);
    });
  });
}

// -------------------------------------------------------------------------------------------
// Training flow
// -------------------------------------------------------------------------------------------

function setTrainingControlsActive(active: boolean) {
  pauseBtn.enabled = active;
  cancelBtn.enabled = active;
  if (!active) {
    pauseBtn.text = 'Pause';
  }
}

trainBtn.on('click', async () => {
  if (!('showDirectoryPicker' in window)) {
    log('Directory picker not available in this browser/context.');
    return;
  }
  trainBtn.enabled = false;
  try {
    // @ts-expect-error showDirectoryPicker is not in the standard TS lib yet.
    const dir = await window.showDirectoryPicker();
    const a = await ensureApp();
    log(`Opened folder: ${dir.name}`);

    // Drop any prior run synchronously so its stream is released.
    training?.free();
    training = null;

    let started = false;
    const t = await a.startTrainingFromDirectory(dir, async (initialConfig: ConfigDoc) => {
      const finalConfig = await showConfigPopup(initialConfig);
      if (!finalConfig) {
        // Returning null tells brush-process to abort the training stream
        // cleanly — see `bridge_config_callback` on the Rust side.
        log('Cancelled');
        stats.status.text = 'idle';
        return null;
      }
      viewerWrap.classList.add('active');
      log('Starting training');
      started = true;
      return finalConfig;
    });
    training = t;
    if (started) setTrainingControlsActive(true);
    await pumpMessages(t);
  } catch (e) {
    log(`Failed to start: ${e}`);
    stats.status.text = 'error';
  } finally {
    trainBtn.enabled = true;
    setTrainingControlsActive(false);
  }
});

pauseBtn.on('click', () => {
  const willPause = pauseBtn.text === 'Pause';
  setPaused(willPause);
  pauseBtn.text = willPause ? 'Resume' : 'Pause';
  log(willPause ? 'Paused' : 'Resumed');
  if (willPause) stats.status.text = 'paused';
});

cancelBtn.on('click', () => {
  if (!training) return;
  // Drop the training run — the pump loop will see `nextMessage` return null
  // on its next iteration and exit. Any in-flight Burn work is cancelled
  // when the underlying stream is dropped.
  training.free();
  training = null;
  // If we were paused, kick the pump so it observes the cleared training.
  setPaused(false);
  pauseBtn.text = 'Pause';
  log('Cancel requested');
});

// -------------------------------------------------------------------------------------------
// Splat <-> renderer plumbing
// -------------------------------------------------------------------------------------------

let pendingRebind = false;
function scheduleRebind() {
  if (!training || !renderer || pendingRebind) return;
  pendingRebind = true;
  const t = training;
  (async () => {
    try {
      const splats = await t.currentSplats();
      if (splats && splats.numSplats > 0 && renderer) {
        const bufs = splats.buffers();
        if (bufs) {
          const shCoeffsPerSplat = (splats.shDegree + 1) * (splats.shDegree + 1) * 3;
          renderer.bindExternal({
            transforms: bufs.transforms as GPUBuffer,
            shCoeffs: bufs.shCoeffs as GPUBuffer,
            rawOpacities: bufs.rawOpacities as GPUBuffer,
            count: splats.numSplats,
            shStride: shCoeffsPerSplat,
          });
        }
      }
    } finally {
      pendingRebind = false;
    }
  })();
}

// Sliding window of recent TrainStep arrivals, used for throughput stats.
const PERF_WINDOW = 32;
type StepSample = { iter: number; jsMs: number };
const stepSamples: StepSample[] = [];

function resetPerf() {
  stepSamples.length = 0;
  stats.stepsPerSec.text = '—';
}

function recordStep(iter: number) {
  stepSamples.push({ iter, jsMs: performance.now() });
  if (stepSamples.length > PERF_WINDOW) stepSamples.shift();
  if (stepSamples.length < 2) return;
  const a = stepSamples[0];
  const b = stepSamples[stepSamples.length - 1];
  const iters = b.iter - a.iter;
  if (iters <= 0) return;
  const wallMs = b.jsMs - a.jsMs;
  stats.stepsPerSec.text = ((iters * 1000) / wallMs).toFixed(1);
}

function applyMessage(msg: BrushMessage) {
  switch (msg.kind) {
    case BrushMessageKind.NewProcess:
      stats.status.text = 'starting';
      resetPerf();
      break;
    case BrushMessageKind.StartLoading:
      stats.status.text = `loading ${msg.name ?? ''}`;
      log(`Loading ${msg.name ?? ''}`);
      break;
    case BrushMessageKind.DatasetLoaded:
      stats.trainViews.text = String(msg.trainViews ?? 0);
      stats.evalViews.text = String(msg.evalViews ?? 0);
      log(`Dataset: ${msg.trainViews ?? 0} train / ${msg.evalViews ?? 0} eval views`);
      break;
    case BrushMessageKind.SplatsUpdated:
      if (msg.numSplats !== undefined) stats.splats.text = String(msg.numSplats);
      scheduleRebind();
      break;
    case BrushMessageKind.TrainStep:
      stats.status.text = 'training';
      if (msg.iter !== undefined) stats.iter.text = String(msg.iter);
      if (msg.elapsedMs !== undefined)
        stats.elapsed.text = `${(msg.elapsedMs / 1000).toFixed(1)}s`;
      if (msg.iter !== undefined) {
        recordStep(msg.iter);
      }
      scheduleRebind();
      break;
    case BrushMessageKind.RefineStep:
      if (msg.numSplats !== undefined) stats.splats.text = String(msg.numSplats);
      scheduleRebind();
      break;
    case BrushMessageKind.EvalResult:
      if (msg.psnr !== undefined) stats.psnr.text = msg.psnr.toFixed(2);
      if (msg.ssim !== undefined) stats.ssim.text = msg.ssim.toFixed(3);
      log(`Eval @ ${msg.iter}: PSNR ${msg.psnr?.toFixed(2)} SSIM ${msg.ssim?.toFixed(3)}`);
      break;
    case BrushMessageKind.DoneLoading:
      log('Loading done');
      break;
    case BrushMessageKind.DoneTraining:
      stats.status.text = 'done';
      log('Training done');
      break;
    case BrushMessageKind.Warning:
      log(`⚠️  ${msg.text ?? ''}`);
      break;
  }
}

// Simple JS-side pause: while `paused` is true, the pump loop awaits a
// promise that resolves when we resume. Brush back-pressures because the
// stream isn't being polled — no Rust state needed.
let paused = false;
let resume: (() => void) | null = null;

function setPaused(p: boolean) {
  paused = p;
  if (!paused && resume) {
    resume();
    resume = null;
  }
}

// Steps per `trainSteps` round trip. Larger amortizes the JS↔wasm boundary;
// smaller keeps the UI more responsive (and `Pause` more snappy).
const STEPS_PER_BATCH = 5;

async function pumpMessages(t: Training) {
  try {
    while (true) {
      while (paused) {
        await new Promise<void>((r) => {
          resume = r;
        });
      }
      // The user may have cancelled (and `free()`d this Training) while we
      // were paused — drop out so we don't call into a freed wasm object.
      if (training !== t) break;
      const msgs = await t.trainSteps(STEPS_PER_BATCH);
      if (msgs.length === 0) break; // stream exhausted
      for (const msg of msgs) applyMessage(msg);
    }
  } catch (e) {
    log(`Error: ${e}`);
    stats.status.text = 'error';
  }
}

// -------------------------------------------------------------------------------------------
// Camera + render loop
// -------------------------------------------------------------------------------------------

const camera: Camera = {
  position: [3, 2, 3],
  target: [0, 0, 0],
  up: [0, 1, 0],
  fovYRad: (50 * Math.PI) / 180,
  near: 0.01,
  far: 100,
};

const orbit = { yaw: Math.PI / 4, pitch: -0.4, radius: 4.5 };

function applyOrbit() {
  const cy = Math.cos(orbit.yaw), sy = Math.sin(orbit.yaw);
  const cp = Math.cos(orbit.pitch), sp = Math.sin(orbit.pitch);
  camera.position = [
    camera.target[0] + orbit.radius * cp * sy,
    camera.target[1] + orbit.radius * sp,
    camera.target[2] + orbit.radius * cp * cy,
  ];
}
applyOrbit();

let dragging = false;
let lastX = 0, lastY = 0;
viewerCanvas.addEventListener('pointerdown', (e) => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
  viewerCanvas.setPointerCapture(e.pointerId);
});
viewerCanvas.addEventListener('pointerup', (e) => {
  dragging = false;
  viewerCanvas.releasePointerCapture(e.pointerId);
});
viewerCanvas.addEventListener('pointermove', (e) => {
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  orbit.yaw -= dx * 0.005;
  orbit.pitch = Math.max(-1.5, Math.min(1.5, orbit.pitch - dy * 0.005));
  applyOrbit();
});
viewerCanvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  orbit.radius = Math.max(0.2, Math.min(50, orbit.radius * (1 + e.deltaY * 0.001)));
  applyOrbit();
}, { passive: false });

function renderLoop() {
  scheduleRebind();
  if (renderer) renderer.render(camera);
  requestAnimationFrame(renderLoop);
}

log('Page loaded — pick a folder to begin');
