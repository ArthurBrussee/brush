#![recursion_limit = "256"]

use burn_wgpu::{
    RuntimeOptions, WgpuDevice,
    graphics::{AutoGraphicsApi, GraphicsApi},
};
use wgpu::{Adapter, Device, Queue};

pub mod config;
pub mod message;

#[cfg(feature = "training")]
pub mod train_stream;

#[cfg(feature = "training")]
pub mod args_file;

pub mod slot;

use std::pin::{Pin, pin};

use anyhow::Error;
use async_fn_stream::try_fn_stream;
use brush_render::MainBackend;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::{DataSource, SendNotWasm};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::WgpuRuntime;
use tokio_stream::{Stream, StreamExt};

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

pub fn burn_init_device(adapter: Adapter, device: Device, queue: Queue) -> WgpuDevice {
    let setup = burn_wgpu::WgpuSetup {
        instance: wgpu::Instance::new(&wgpu::InstanceDescriptor::default()), // unused... need to fix this in Burn.
        adapter,
        device,
        queue,
        backend: AutoGraphicsApi::backend(),
    };
    let burn = burn_wgpu::init_device(setup, burn_options());
    connect_device(burn.clone());
    burn
}

use crate::{message::ProcessMessage, slot::Slot};

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

/// Create a running process from a datasource and args.
///
/// The `config_fn` callback receives the initial config (loaded from args.txt if present,
/// otherwise defaults) and returns the final config to use. This allows the caller to
/// modify or override settings as needed.
pub fn create_process<
    #[cfg(feature = "training")] Fun: FnOnce(crate::config::TrainStreamConfig) -> Fut + Send + 'static,
    #[cfg(feature = "training")] Fut: std::future::Future<Output = crate::config::TrainStreamConfig> + Send,
>(
    source: DataSource,
    #[cfg(feature = "training")] config_fn: Fun,
) -> RunningProcess {
    let splat_view = Slot::default();
    let splat_state_cl = splat_view.clone();

    let stream = try_fn_stream(|emitter| async move {
        log::info!("Starting process with source {source:?}");
        emitter.emit(ProcessMessage::NewProcess).await;

        // Wait until the devise is set.
        let device = DEVICE.wait().await.clone();

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
        #[cfg(feature = "training")]
        let initial_config = crate::args_file::load_config_from_vfs(&vfs).await;

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
            let client = WgpuRuntime::client(&device);
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
                    let splats = message.data.into_splats(&device, mode);

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
            #[cfg(feature = "training")]
            {
                // Pass initial config (from args.txt or defaults) to the callback
                let base_config = initial_config.unwrap_or_default();
                let config = config_fn(base_config).await;
                crate::train_stream::train_stream(vfs, config, device, emitter, splat_view).await?;
            }

            #[cfg(not(feature = "training"))]
            anyhow::bail!("Training is not enabled in Brush, cannot load dataset.");
        };

        Ok(())
    });

    RunningProcess {
        stream: Box::pin(stream),
        splat_view: splat_state_cl,
    }
}
