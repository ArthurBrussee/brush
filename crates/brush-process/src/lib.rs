#![recursion_limit = "256"]

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
use async_fn_stream::try_fn_stream;
use brush_render::MainBackend;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::SendNotWasm;
use burn_cubecl::cubecl::Runtime;
use burn_cubecl::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, streaming::StreamingConfig};
use burn_wgpu::WgpuRuntime;
use tokio_stream::{Stream, StreamExt};

/// Set `CubeCL` global runtime config. Idempotent — `CubeClRuntimeConfig::set`
/// panics if called twice, but in browsers React's `StrictMode` and similar
/// host conventions can re-init us, so guard with [`std::sync::Once`].
pub fn cubecl_startup() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        CubeClRuntimeConfig::set(CubeClRuntimeConfig {
            streaming: StreamingConfig {
                max_streams: 1,
                ..Default::default()
            },
            ..Default::default()
        });
    });
}

fn burn_options() -> RuntimeOptions {
    RuntimeOptions {
        tasks_max: 64,
        memory_config: burn_wgpu::MemoryConfiguration::ExclusivePages,
    }
}

pub async fn burn_init_setup() -> WgpuDevice {
    burn_wgpu::init_setup_async::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, burn_options())
        .await;
    connect_device(WgpuDevice::DefaultDevice);
    WgpuDevice::DefaultDevice
}

/// Initialize Burn with a wgpu setup the host already owns. Useful when
/// integrating with an existing wgpu/WebGPU application that wants to share
/// its device with Brush so tensor buffers can flow back into the host's
/// render pipeline without copies.
pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    connect_device(burn.clone());
    burn
}

use crate::{message::ProcessMessage, slot::Slot, train_stream::train_stream};

pub trait ProcessStream: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}
impl<T> ProcessStream for T where T: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}

pub struct RunningProcess {
    pub stream: Pin<Box<dyn ProcessStream>>,
    pub splat_view: Slot<Splats<MainBackend>>,
}

use tokio::sync::SetOnce;

static DEVICE: SetOnce<WgpuDevice> = SetOnce::const_new();

pub(crate) fn connect_device(device: WgpuDevice) {
    DEVICE.set(device).unwrap();
}

pub async fn wait_for_device() -> &'static WgpuDevice {
    DEVICE.wait().await
}

/// Create a running process from a datasource and args.
///
/// The `config_fn` callback receives the initial config (loaded from args.txt if present,
/// otherwise defaults) and returns the final config to use. This allows the caller to
/// modify or override settings as needed.
pub fn create_process<
    Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + SendNotWasm + 'static,
    Fut: Future<Output = Option<crate::config::TrainStreamConfig>> + SendNotWasm,
>(
    source: DataSource,
    config_fn: Fun,
) -> RunningProcess {
    let splat_view = Slot::default();
    let splat_state_cl = splat_view.clone();

    let stream = try_fn_stream(|emitter| async move {
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
            let device = wait_for_device().await;
            let mut paths: Vec<_> = vfs.file_paths().collect();
            alphanumeric_sort::sort_path_slice(&mut paths);
            let client = WgpuRuntime::client(device);
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
                    client.memory_cleanup();

                    // For the first frame of a new file, clear existing frames
                    if frame == 0 {
                        splat_view.clear().await;
                    }

                    // Capture stats before moving splats
                    let num_splats = splats.num_splats();
                    let sh_degree = splats.sh_degree();
                    splat_view.set_at(frame, splats).await;

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
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view: splat_state_cl,
    }
}
