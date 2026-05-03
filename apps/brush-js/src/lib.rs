//! brush-js — minimal JS / WebAssembly bindings for Brush training.

#![cfg(target_family = "wasm")]

use std::sync::Arc;

use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::slot::Slot;
use brush_process::{ProcessStream, burn_init_setup, create_process_from_vfs, cubecl_startup};
use brush_render::MainBackend;
use brush_render::gaussian_splats::Splats;
use brush_vfs::BrushVfs;
use std::pin::Pin;
use serde::Serialize;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
use wasm_bindgen::prelude::*;
use web_sys::js_sys;

#[wasm_bindgen]
#[derive(Clone, Copy, Debug)]
pub enum BrushMessageKind {
    NewProcess,
    StartLoading,
    SplatsUpdated,
    DatasetLoaded,
    TrainStep,
    RefineStep,
    EvalResult,
    DoneTraining,
    DoneLoading,
    Warning,
}

/// Opaque wrapper around the Rust [`ProcessMessage`] enum.
#[wasm_bindgen]
pub struct BrushMessage {
    inner: ProcessMessage,
}

#[wasm_bindgen]
impl BrushMessage {
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> BrushMessageKind {
        match &self.inner {
            ProcessMessage::NewProcess => BrushMessageKind::NewProcess,
            ProcessMessage::StartLoading { .. } => BrushMessageKind::StartLoading,
            ProcessMessage::SplatsUpdated { .. } => BrushMessageKind::SplatsUpdated,
            ProcessMessage::TrainMessage(t) => match t {
                TrainMessage::Dataset { .. } => BrushMessageKind::DatasetLoaded,
                TrainMessage::TrainStep { .. } => BrushMessageKind::TrainStep,
                TrainMessage::RefineStep { .. } => BrushMessageKind::RefineStep,
                TrainMessage::EvalResult { .. } => BrushMessageKind::EvalResult,
                TrainMessage::DoneTraining => BrushMessageKind::DoneTraining,
                // Filtered before reaching JS; arm exists only for exhaustiveness.
                TrainMessage::TrainConfig { .. } => BrushMessageKind::DoneLoading,
            },
            ProcessMessage::Warning { .. } => BrushMessageKind::Warning,
            ProcessMessage::DoneLoading => BrushMessageKind::DoneLoading,
        }
    }

    #[wasm_bindgen(getter)]
    #[allow(clippy::iter_not_returning_iterator)]
    pub fn iter(&self) -> Option<u32> {
        match &self.inner {
            ProcessMessage::TrainMessage(
                TrainMessage::TrainStep { iter, .. }
                | TrainMessage::RefineStep { iter, .. }
                | TrainMessage::EvalResult { iter, .. },
            ) => Some(*iter),
            _ => None,
        }
    }

    #[wasm_bindgen(getter, js_name = numSplats)]
    pub fn num_splats(&self) -> Option<u32> {
        match &self.inner {
            ProcessMessage::SplatsUpdated { num_splats, .. } => Some(*num_splats),
            ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                cur_splat_count, ..
            }) => Some(*cur_splat_count),
            _ => None,
        }
    }

    #[wasm_bindgen(getter, js_name = shDegree)]
    pub fn sh_degree(&self) -> Option<u32> {
        match &self.inner {
            ProcessMessage::SplatsUpdated { sh_degree, .. } => Some(*sh_degree),
            _ => None,
        }
    }

    #[wasm_bindgen(getter, js_name = elapsedMs)]
    pub fn elapsed_ms(&self) -> Option<f64> {
        match &self.inner {
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { total_elapsed, .. }) => {
                Some(total_elapsed.as_secs_f64() * 1000.0)
            }
            _ => None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn psnr(&self) -> Option<f32> {
        match &self.inner {
            ProcessMessage::TrainMessage(TrainMessage::EvalResult { avg_psnr, .. }) => {
                Some(*avg_psnr)
            }
            _ => None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn ssim(&self) -> Option<f32> {
        match &self.inner {
            ProcessMessage::TrainMessage(TrainMessage::EvalResult { avg_ssim, .. }) => {
                Some(*avg_ssim)
            }
            _ => None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Option<String> {
        match &self.inner {
            ProcessMessage::StartLoading { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    #[wasm_bindgen(getter, js_name = trainViews)]
    pub fn train_views(&self) -> Option<u32> {
        match &self.inner {
            ProcessMessage::TrainMessage(TrainMessage::Dataset { dataset }) => {
                Some(dataset.train.views.len() as u32)
            }
            _ => None,
        }
    }

    #[wasm_bindgen(getter, js_name = evalViews)]
    pub fn eval_views(&self) -> Option<u32> {
        match &self.inner {
            ProcessMessage::TrainMessage(TrainMessage::Dataset { dataset }) => {
                Some(dataset.eval.as_ref().map_or(0, |s| s.views.len() as u32))
            }
            _ => None,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn text(&self) -> Option<String> {
        match &self.inner {
            ProcessMessage::Warning { error } => Some(format!("{error:#}")),
            _ => None,
        }
    }
}

#[wasm_bindgen]
pub struct BrushSplats {
    inner: Splats<MainBackend>,
}

/// Snapshot of the GPU buffers backing a [`BrushSplats`]. All three buffers
/// live on the [`GPUDevice`] Brush is training on, so they can be bound
/// directly to render pipelines on that same device with no copies.
#[wasm_bindgen]
pub struct BrushSplatBuffers {
    transforms: JsValue,
    sh_coeffs: JsValue,
    raw_opacities: JsValue,
}

#[wasm_bindgen]
impl BrushSplatBuffers {
    /// Packed `[N, 10]` `GPUBuffer`. Each row is
    /// `means(3) | rotation_xyzw(4) | log_scales(3)`, stride 40 bytes.
    /// Bind as a vertex buffer with attributes at offsets 0 / 12 / 28.
    #[wasm_bindgen(getter)]
    pub fn transforms(&self) -> JsValue {
        self.transforms.clone()
    }
    /// Spherical-harmonics coefficients `[N, n_coeffs, 3]`, where
    /// `n_coeffs = (sh_degree + 1)^2`.
    #[wasm_bindgen(getter, js_name = shCoeffs)]
    pub fn sh_coeffs(&self) -> JsValue {
        self.sh_coeffs.clone()
    }
    /// Per-splat raw (pre-sigmoid) opacities `[N]`.
    #[wasm_bindgen(getter, js_name = rawOpacities)]
    pub fn raw_opacities(&self) -> JsValue {
        self.raw_opacities.clone()
    }
}

#[wasm_bindgen]
impl BrushSplats {
    #[wasm_bindgen(getter, js_name = numSplats)]
    pub fn num_splats(&self) -> u32 {
        self.inner.num_splats()
    }

    #[wasm_bindgen(getter, js_name = shDegree)]
    pub fn sh_degree(&self) -> u32 {
        self.inner.sh_degree()
    }

    /// All three GPU buffers backing this snapshot. Returns `null` if Brush
    /// isn't running on the WebGPU backend.
    pub fn buffers(&self) -> Option<BrushSplatBuffers> {
        Some(BrushSplatBuffers {
            transforms: tensor_buffer_js(self.inner.transforms.val())?,
            sh_coeffs: tensor_buffer_js(self.inner.sh_coeffs.val())?,
            raw_opacities: tensor_buffer_js(self.inner.raw_opacities.val())?,
        })
    }
}

// -------------------------------------------------------------------------------------------
// BrushApp + Training
// -------------------------------------------------------------------------------------------

/// Owns the Brush runtime state (wgpu device, panic hook, etc.). Holds nothing
/// per-training-run — each [`Self::start_training_from_directory`] call
/// returns a fresh [`Training`] you drive yourself.
#[wasm_bindgen]
pub struct BrushApp;

#[wasm_bindgen]
impl BrushApp {
    /// Construct a `BrushApp`. Installs the wasm panic hook + logger and
    /// applies the global `CubeCL` config. You must `await app.init()`
    /// (or call `app.initExisting(...)`) before starting any training.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        // Idempotent: hosts that re-init us (React StrictMode in dev, hot
        // reloads, etc.) would otherwise hit "logger already set" each time.
        static LOGGER_INIT: std::sync::Once = std::sync::Once::new();
        #[cfg(debug_assertions)]
        LOGGER_INIT.call_once(|| {
            wasm_logger::init(wasm_logger::Config::new(log::Level::Info));
        });
        cubecl_startup();
        Self
    }

    /// Initialize Brush with its own internal `GPUDevice`.
    pub async fn init(&self) -> Result<(), JsValue> {
        burn_init_setup().await;
        Ok(())
    }

    /// Initialize Brush against an existing `(GPUAdapter, GPUDevice, GPUQueue)`
    /// triple. Use this when the host app already has a WebGPU device and
    /// wants Brush to train on the same one — splat buffers exposed via
    /// [`BrushSplats::buffers`] then bind directly into the host's render
    /// pipelines without copies.
    #[wasm_bindgen(js_name = initExisting)]
    pub fn init_existing(
        &self,
        adapter: JsValue,
        device: JsValue,
        queue: JsValue,
    ) -> Result<(), JsValue> {
        let adapter = wgpu::webgpu_backend::WebAdapter::from_handle(adapter);
        let device = wgpu::webgpu_backend::WebDevice::from_handle(device);
        let queue = wgpu::webgpu_backend::WebQueue::from_handle(queue);
        brush_process::burn_init_external(
            wgpu::Adapter::from_webgpu(adapter),
            wgpu::Device::from_webgpu(device),
            wgpu::Queue::from_webgpu(queue),
        );
        Ok(())
    }

    /// Start training from a directory picked via `window.showDirectoryPicker()`.
    ///
    /// `config_fn` is called once with the initial [`TrainStreamConfig`] —
    /// loaded from `args.txt` in the chosen directory if present, defaults
    /// otherwise — serialized as a plain JS object. It must return a Promise
    /// resolving to the final config (or `null` to abort).
    ///
    /// The returned [`Training`] owns the underlying message stream. Drive it
    /// with `await training.trainSteps(N)`. To cancel, just drop it
    /// (`training.free()` synchronously, or let GC do it eventually) — Rust's
    /// normal future cancellation tears down any pending Burn work.
    ///
    /// To pause, just stop pumping; the training loop back-pressures
    /// because nothing is consuming messages.
    #[wasm_bindgen(js_name = startTrainingFromDirectory)]
    pub async fn start_training_from_directory(
        &self,
        handle: web_sys::FileSystemDirectoryHandle,
        config_fn: js_sys::Function,
    ) -> Result<Training, JsValue> {
        let display_name = handle.name();
        let dir = rrfd::wasm::DirectoryHandle::from_handle(handle);
        let vfs = Arc::new(BrushVfs::from_directory_handle(dir).await.map_err(js_err)?);

        let process = create_process_from_vfs(vfs, display_name, async move |init| {
            bridge_config_callback(config_fn, init).await
        });

        Ok(Training {
            stream: Mutex::new(process.stream),
            splat_view: process.splat_view,
        })
    }
}

/// A single training run. Owns the underlying brush-process stream + splat
/// view; dropping it cancels the run.
#[wasm_bindgen]
pub struct Training {
    stream: Mutex<Pin<Box<dyn ProcessStream>>>,
    splat_view: Slot<Splats<MainBackend>>,
}

#[wasm_bindgen]
impl Training {
    /// Drive the training stream until `steps` `TrainStep` events have been
    /// emitted (or the stream ends), and return every message produced along
    /// the way.
    ///
    /// On the first call this also yields the loading-phase messages
    /// (`StartLoading`, `Dataset`, initial `SplatsUpdated`, `DoneLoading`,
    /// then the first `steps` `TrainStep`s). Returns an empty array when
    /// the stream is fully exhausted — that's the JS host's "stop pumping"
    /// signal.
    ///
    /// Internal `TrainConfig` echoes are filtered (implementation detail of
    /// brush-process). Errors propagate as a Promise rejection; messages
    /// collected before the error are dropped — re-stream a fresh
    /// [`Training`] if recovery is needed.
    #[wasm_bindgen(js_name = trainSteps)]
    pub async fn train_steps(&self, steps: u32) -> Result<Vec<BrushMessage>, JsValue> {
        let mut stream = self.stream.lock().await;
        let mut out: Vec<BrushMessage> = Vec::new();
        let mut steps_taken: u32 = 0;
        loop {
            match stream.next().await {
                Some(Ok(ProcessMessage::TrainMessage(TrainMessage::TrainConfig { .. }))) => {
                    continue;
                }
                Some(Ok(msg)) => {
                    let is_step = matches!(
                        &msg,
                        ProcessMessage::TrainMessage(TrainMessage::TrainStep { .. })
                    );
                    out.push(BrushMessage { inner: msg });
                    if is_step {
                        steps_taken += 1;
                        if steps_taken >= steps {
                            return Ok(out);
                        }
                    }
                }
                Some(Err(e)) => return Err(js_err_str(&format!("{e:#}"))),
                None => return Ok(out),
            }
        }
    }

    /// Snapshot the current splats. Returns `null` if no splats have been
    /// produced yet.
    #[wasm_bindgen(js_name = currentSplats)]
    pub async fn current_splats(&self) -> Option<BrushSplats> {
        self.splat_view
            .clone_main()
            .await
            .map(|inner| BrushSplats { inner })
    }
}

// -------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------

/// Round-trip the initial `TrainStreamConfig` through a JS async callback.
/// Returns `None` to signal "user cancelled" — the JS callback can throw,
/// reject the promise, or explicitly return `null` / `undefined` to abort
/// the training stream cleanly.
async fn bridge_config_callback(
    config_fn: js_sys::Function,
    init: brush_process::config::TrainStreamConfig,
) -> Option<brush_process::config::TrainStreamConfig> {
    // `to_value` defaults to serializing structs as JS `Map`s — they don't
    // survive `{...obj}` spreads on the JS side. The JSON-compatible serializer
    // emits plain objects + arrays, which is what hosts expect.
    let serializer = serde_wasm_bindgen::Serializer::json_compatible();
    let init_js = match init.serialize(&serializer) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("Failed to serialize initial config: {e}");
            return Some(init);
        }
    };

    let promise = match config_fn.call1(&JsValue::NULL, &init_js) {
        Ok(p) => p,
        Err(e) => {
            log::warn!("config_fn threw: {e:?} — aborting");
            return None;
        }
    };

    let Ok(promise) = promise.dyn_into::<js_sys::Promise>() else {
        log::warn!("config_fn must return a Promise — aborting");
        return None;
    };

    let value = match wasm_bindgen_futures::JsFuture::from(promise).await {
        Ok(v) => v,
        Err(e) => {
            log::info!("config_fn rejected ({e:?}) — aborting");
            return None;
        }
    };

    if value.is_null() || value.is_undefined() {
        return None;
    }

    match serde_wasm_bindgen::from_value(value) {
        Ok(out) => Some(out),
        Err(e) => {
            log::warn!("Failed to parse config_fn result: {e} — aborting");
            None
        }
    }
}

/// Resolve a fusion-wrapped tensor down to its underlying [`wgpu::Buffer`]
/// and return it as a JS `GPUBuffer`. Returns `None` when Brush isn't on
/// the WebGPU backend (which is the only backend brush-js currently
/// supports anyway).
fn tensor_buffer_js<const D: usize>(
    tensor: burn::tensor::Tensor<MainBackend, D>,
) -> Option<JsValue> {
    use brush_render::MainBackendBase;

    let fusion_tensor = tensor.into_primitive().tensor();
    let cube_tensor = fusion_tensor
        .client
        .clone()
        .resolve_tensor_float::<MainBackendBase>(fusion_tensor);
    let resource = cube_tensor.client.get_resource(cube_tensor.handle).ok()?;
    resource.resource().buffer.as_webgpu().map(|w| w.raw_js())
}

fn js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&format!("{e}"))
}

fn js_err_str(s: &str) -> JsValue {
    JsValue::from_str(s)
}

impl Default for BrushApp {
    fn default() -> Self {
        Self::new()
    }
}
