pub mod args_file;
pub mod config;
pub mod message;
pub mod slot;
pub mod train_stream;

pub use brush_vfs::DataSource;
pub type ProcessDevice = burn::tensor::Device;

pub fn default_device() -> ProcessDevice {
    wgpu_device().into()
}

fn wgpu_device() -> WgpuDevice {
    configure_runtime();
    WgpuDevice::default().on(preferred_wgpu_backend())
}

/// Backend policy shared by training and the native viewer.
/// `Auto` leaves adapter selection to the caller; `BRUSH_WGPU_BACKEND`
/// overrides the platform default for diagnosis.
pub fn preferred_wgpu_backend() -> burn::cubecl::wgpu::WgpuBackend {
    use burn::cubecl::wgpu::WgpuBackend;

    // Keep the API in the device identity as well as its initial setup. On
    // Windows ARM64, the Adreno Vulkan path fails after splat refinement.
    let backend = if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        WgpuBackend::Dx12
    } else {
        WgpuBackend::Auto
    };

    // Allow backend comparisons without rebuilding; this also makes future
    // driver regressions reproducible from CLI logs.
    #[cfg(not(target_family = "wasm"))]
    let backend = match std::env::var("BRUSH_WGPU_BACKEND") {
        Ok(value) => match value.to_ascii_lowercase().as_str() {
            "auto" => WgpuBackend::Auto,
            "dx12" => WgpuBackend::Dx12,
            "vulkan" => WgpuBackend::Vulkan,
            "metal" => WgpuBackend::Metal,
            "gl" => WgpuBackend::Gl,
            _ => panic!("BRUSH_WGPU_BACKEND must be auto, dx12, vulkan, metal, or gl"),
        },
        Err(_) => backend,
    };

    backend
}

use burn_wgpu::{RuntimeOptions, WgpuDevice, graphics::AutoGraphicsApi};
use wgpu::{Adapter, Device, Queue};

use std::future::Future;
use std::pin::{Pin, pin};

use anyhow::Error;
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_train::train::{BOUND_PERCENTILE, get_splat_bounds};
use brush_vfs::SendNotWasm;
use tokio_stream::{Stream, StreamExt};

fn burn_options() -> RuntimeOptions {
    configure_runtime();
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

fn configure_runtime() {
    #[cfg(target_os = "windows")]
    {
        use burn::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let mut config = CubeClRuntimeConfig::from_current_dir();
            // CubeCL's Turso cache requests multiprocess WAL, unsupported by
            // Turso 0.8.2's default Windows IO backend. Keep autotuning's
            // in-memory results, without repeated failing database writes.
            config.autotune.disable_cache = true;
            config.throughput.disable_cache = true;
            // Session records use the same database independently of those
            // caches, and otherwise retry a failed open for every new kernel.
            config.environment.records.level = burn::cubecl::records::RecordLevel::Off;
            // An embedding application's existing configuration takes priority;
            // environment overrides also remain available for diagnosis.
            CubeClRuntimeConfig::try_set(config.override_from_env());
        });
    }
}

pub async fn burn_init_setup() -> ProcessDevice {
    let device = wgpu_device();
    let setup = burn_wgpu::init_setup_async::<AutoGraphicsApi>(&device, burn_options()).await;
    log::info!("GPU adapter: {:?}", setup.adapter.get_info());
    device.into()
}

pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> ProcessDevice {
    let backend = adapter.get_info().backend;
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend,
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    burn.into()
}

use crate::{
    message::ProcessMessage,
    slot::{Slot, SlotSender},
    train_stream::train_stream,
};

pub trait ProcessStream: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}
impl<T> ProcessStream for T where T: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}

pub struct RunningProcess {
    pub stream: Pin<Box<dyn ProcessStream>>,
    pub splat_view: Slot<Splats>,
    pub device: ProcessDevice,
}

/// Convenience alias for the emitter `try_fn_stream` hands us inside
/// the producer body — `try_fn_stream` itself drives the state
/// machine, so this is just the channel for `emit(msg).await`.
pub(crate) type Emitter = TryStreamEmitter<ProcessMessage, Error>;

/// Create a running process from a datasource and args.
///
/// The `config_fn` callback receives the initial config (loaded from
/// args.txt if present, otherwise defaults) and returns the final
/// config to use. This allows the caller to modify or override
/// settings as needed.
pub fn create_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>> + SendNotWasm,
>(
    source: DataSource,
    config_fn: Fun,
) -> RunningProcess {
    create_process_with_device(source, default_device(), config_fn)
}

pub fn create_process_with_device<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>> + SendNotWasm,
>(
    source: DataSource,
    device: ProcessDevice,
    config_fn: Fun,
) -> RunningProcess {
    let (splat_tx, splat_view) = crate::slot::channel();
    let process_device = device.clone();
    let stream = try_fn_stream(|emitter| async move {
        run_process(source, config_fn, &emitter, splat_tx, &process_device).await
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
        device,
    }
}

async fn run_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>>,
>(
    source: DataSource,
    config_fn: Fun,
    emitter: &Emitter,
    splat_view: SlotSender<Splats>,
    device: &ProcessDevice,
) -> Result<(), Error> {
    log::info!("Starting process with source {source:?}");
    emitter.emit(ProcessMessage::NewProcess).await;

    let vfs = source.clone().into_vfs().await?;
    let vfs_counts = vfs.file_count();

    if vfs_counts == 0 {
        return Err(anyhow::anyhow!("No files found."));
    }

    let ply_count = vfs.files_with_extension("ply").count();

    log::info!(
        "Mounted VFS with {} files. (plys: {})",
        vfs.file_count(),
        ply_count
    );

    let is_training = vfs_counts != ply_count;

    // Emit source info - just the display name
    let paths: Vec<_> = vfs.file_paths().collect();
    let source_name = if let Some(base_path) = vfs.base_path() {
        base_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(if is_training { "dataset" } else { "file" })
            .to_owned()
    } else if paths.len() == 1 {
        paths[0]
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("input.ply")
            .to_owned()
    } else {
        format!("{} files", paths.len())
    };

    let base_path = vfs.base_path();

    // Load initial config from args.txt via VFS if present
    let initial_config = args_file::load_config_from_vfs(&vfs).await;

    emitter
        .emit(ProcessMessage::StartLoading {
            name: source_name,
            source,
            training: is_training,
            base_path,
        })
        .await;

    if !is_training {
        let mut paths: Vec<_> = vfs.file_paths().collect();
        alphanumeric_sort::sort_path_slice(&mut paths);
        let total_frames = paths.len() as u32;

        for (frame, path) in paths.iter().enumerate() {
            log::info!("Loading single ply file");

            let mut splat_stream = pin!(brush_serde::stream_splat_from_ply(
                vfs.reader_at_path(path).await?,
                None,
                true,
            ));

            while let Some(message) = splat_stream.next().await {
                let message = message?;

                let degree = message
                    .data
                    .sh_coeffs
                    .as_ref()
                    .filter(|_| message.data.num_splats() > 0)
                    .map_or(0, |sh| {
                        brush_render::sh::sh_degree_from_coeffs(
                            (sh.len() / message.data.num_splats() / 3) as u32,
                        )
                    });
                let safe_max = brush_render::gpu_limits::max_splats(device, degree, false);
                let model_count =
                    (message.meta.total_splats as usize).max(message.data.num_splats());
                anyhow::ensure!(
                    model_count <= safe_max as usize,
                    "This model has {model_count} splats at SH degree {degree}, exceeding this GPU's safe limit of {safe_max}. Use a smaller model or a GPU with a larger safe buffer limit."
                );
                let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
                let splats = message.data.into_splats(device, mode);

                // As loading concatenates splats each time, memory usage tends to accumulate a lot
                // over time. Clear out memory after each step to prevent this buildup.
                device.memory_cleanup();

                // For the first frame of a new file, clear existing frames
                if frame == 0 {
                    splat_view.clear();
                }

                // Capture stats before moving splats
                let num_splats = splats.num_splats();
                let sh_degree = splats.sh_degree();
                let scene_scale = get_splat_bounds(splats.clone(), BOUND_PERCENTILE)
                    .await
                    .median_size();
                splat_view.set(frame, splats);

                emitter
                    .emit(ProcessMessage::SplatsUpdated {
                        up_axis: message.meta.up_axis,
                        frame: frame as u32,
                        total_frames,
                        num_splats,
                        sh_degree,
                        scene_scale,
                    })
                    .await;
            }
        }

        emitter.emit(ProcessMessage::DoneLoading).await;
    } else {
        // Pass initial config (from args.txt or defaults) to the callback.
        // Returning `None` from `config_fn` aborts cleanly without
        // surfacing as an error.
        let base_config = initial_config.unwrap_or_default();
        let Some(config) = config_fn(base_config).await else {
            log::info!("config_fn returned None — aborting before training");
            return Ok(());
        };
        train_stream(vfs, config, emitter, splat_view, device).await?;
    };

    Ok(())
}
