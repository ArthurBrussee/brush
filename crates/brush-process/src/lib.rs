pub mod args_file;
pub mod config;
pub mod message;
pub mod slot;
pub mod train_stream;

pub use brush_vfs::DataSource;

use burn_wgpu::{
    RuntimeOptions, WgpuDevice,
    graphics::{AutoGraphicsApi, GraphicsApi},
};
use wgpu::{Adapter, Device, Queue};

use std::future::Future;
use std::pin::{Pin, pin};

use anyhow::Error;
use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::SendNotWasm;
use tokio_stream::{Stream, StreamExt};

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

/// Open the compute device.
///
/// Its own device, separate from anything the viewer owns, so training never
/// contends with GUI work on the same queue. wgpu needs an explicit setup so
/// the adapter and limits are chosen before any kernel runs.
async fn open_device() -> burn::tensor::Device {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    WgpuDevice::DefaultDevice.into()
}

/// Open the compute device now, rather than waiting for the first thing that
/// needs it. Only worth calling to front-load the cost.
pub async fn burn_init_setup() -> burn::tensor::Device {
    device().await.clone()
}

/// Hand Brush a wgpu setup the host already owns, instead of letting it open
/// its own device. Useful when integrating with an existing wgpu/WebGPU
/// application: tensor buffers then bind directly into the host's render
/// pipelines without copies.
///
/// Must be called before anything touches the device, and does nothing if the
/// device is already open. Only meaningful for the wgpu backend.
pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    // A JS host can call `init()` and `initExisting()`, or a dev-mode double
    // mount can re-run setup. Whoever gets there first wins.
    let _ = DEVICE.set(burn.clone().into());
    burn
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
}

/// Convenience alias for the emitter `try_fn_stream` hands us inside
/// the producer body — `try_fn_stream` itself drives the state
/// machine, so this is just the channel for `emit(msg).await`.
pub(crate) type Emitter = TryStreamEmitter<ProcessMessage, Error>;

use tokio::sync::OnceCell;

/// Free cached GPU memory on whichever runtime the device belongs to.
///
/// `memory_cleanup` / `memory_usage` live on the cubecl client, which is
/// runtime-specific, so this has to branch on the dispatch variant. Only wgpu
/// is wired up today; another cubecl runtime would add an arm here.
pub fn device_memory_cleanup(device: &burn::tensor::Device) {
    use burn::backend::DispatchDevice;
    use burn::cubecl::Runtime;
    if let DispatchDevice::Wgpu(d) = device.as_dispatch() {
        burn_wgpu::WgpuRuntime::<burn_wgpu::AutoCompiler>::client(d).memory_cleanup();
    }
}

/// Bytes currently reserved by the runtime's memory pool, if it reports them.
pub fn device_memory_usage(device: &burn::tensor::Device) -> Option<burn::cubecl::MemoryUsage> {
    use burn::backend::DispatchDevice;
    use burn::cubecl::Runtime;
    match device.as_dispatch() {
        DispatchDevice::Wgpu(d) => {
            Some(burn_wgpu::WgpuRuntime::<burn_wgpu::AutoCompiler>::client(d).memory_usage())
        }
        // Autodiff wraps a device rather than being one; nothing to report.
        DispatchDevice::Autodiff(_) => None,
    }
}

static DEVICE: OnceCell<burn::tensor::Device> = OnceCell::const_new();

/// The compute device, opening it if this is the first call.
pub async fn device() -> &'static burn::tensor::Device {
    DEVICE.get_or_init(open_device).await
}

/// The compute device, but only if it is already open. For UI that wants to
/// report on the device without causing it to be created.
pub fn try_device() -> Option<&'static burn::tensor::Device> {
    DEVICE.get()
}

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
    let (splat_tx, splat_view) = crate::slot::channel();

    let stream =
        try_fn_stream(
            |emitter| async move { run_process(source, config_fn, &emitter, splat_tx).await },
        );

    RunningProcess {
        stream: Box::pin(stream),
        splat_view,
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
        let device = device().await;
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

                let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
                let splats = message.data.into_splats(device, mode);

                // As loading concatenates splats each time, memory usage tends to accumulate a lot
                // over time. Clear out memory after each step to prevent this buildup.
                device_memory_cleanup(device);

                // For the first frame of a new file, clear existing frames
                if frame == 0 {
                    splat_view.clear();
                }

                // Capture stats before moving splats
                let num_splats = splats.num_splats();
                let sh_degree = splats.sh_degree();
                splat_view.set(frame, splats);

                emitter
                    .emit(ProcessMessage::SplatsUpdated {
                        up_axis: message.meta.up_axis,
                        frame: frame as u32,
                        total_frames,
                        num_splats,
                        sh_degree,
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
        train_stream(vfs, config, emitter, splat_view).await?;
    };

    Ok(())
}
