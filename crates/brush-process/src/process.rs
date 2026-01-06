use std::pin::{Pin, pin};

use anyhow::Error;
use async_fn_stream::try_fn_stream;
use brush_render::MainBackend;
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_vfs::{DataSource, SendNotWasm};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};

#[allow(unused)]
use brush_serde;
use tokio_stream::{Stream, StreamExt};

use crate::{message::ProcessMessage, slot::Slot};

use crate::config::TrainStreamConfig;

pub trait ProcessStream: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}
impl<T> ProcessStream for T where T: Stream<Item = Result<ProcessMessage, Error>> + SendNotWasm {}

pub struct RunningProcess {
    pub stream: Pin<Box<dyn ProcessStream>>,
    pub splat_view: Slot<Splats<MainBackend>>,
}

/// Create a running process from a datasource and args.
pub fn create_process(
    source: DataSource,
    #[allow(unused)] config: impl Future<Output = TrainStreamConfig> + Send + 'static,
    device: WgpuDevice,
    splat_view: Slot<Splats<MainBackend>>,
) -> RunningProcess {
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

        emitter
            .emit(ProcessMessage::StartLoading {
                name: source_name,
                source,
                training: is_training,
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
                        splat_view.clear();
                    }

                    // Ensure we have space up to this frame index and set it
                    {
                        let mut guard = splat_view.lock();
                        if guard.len() <= frame {
                            guard.resize(frame + 1, splats.clone());
                        }
                        guard[frame] = splats;
                    }

                    emitter
                        .emit(ProcessMessage::SplatsUpdated {
                            up_axis: message.meta.up_axis,
                            frame: frame as u32,
                            total_frames,
                        })
                        .await;
                }
            }

            emitter.emit(ProcessMessage::DoneLoading).await;
        } else {
            #[cfg(feature = "training")]
            {
                let config = config.await;
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
