use std::pin::{Pin, pin};

use async_fn_stream::{TryStreamEmitter, try_fn_stream};
use brush_render::gaussian_splats::SplatRenderMode;
use brush_vfs::DataSource;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};

#[allow(unused)]
use brush_serde;
use tokio_stream::{Stream, StreamExt};

use crate::{
    message::{ProcessMessage, SplatView},
    slot::Slot,
};

use crate::config::TrainStreamConfig;

pub struct RunningProcess {
    pub stream: Pin<Box<dyn Stream<Item = Result<ProcessMessage, anyhow::Error>> + Send>>,
    pub splat_view: Slot<SplatView>,
}

/// Create a running process from a datasource and args.
pub fn create_process<CF, CFut>(
    source: DataSource,
    config: CF,
    device: WgpuDevice,
    splat_view: Slot<SplatView>,
) -> RunningProcess
where
    CF: FnOnce() -> CFut + Send + 'static,
    CFut: Future<Output = TrainStreamConfig> + Send,
{
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

            for (i, path) in paths.iter().enumerate() {
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

                    // If there's multiple ply files in a zip, don't support animated plys, that would
                    // get rather mind bending.
                    let (frame, total_frames) = if paths.len() == 1 {
                        (message.meta.current_frame, message.meta.frame_count)
                    } else {
                        (i as u32, paths.len() as u32)
                    };

                    // As loading concatenates splats each time, memory usage tends to accumulate a lot
                    // over time. Clear out memory after each step to prevent this buildup.
                    client.memory_cleanup();

                    update_splat_state(
                        &emitter,
                        &splat_view,
                        SplatView::new(splats, message.meta.up_axis, frame, total_frames),
                    )
                    .await;
                }
            }

            emitter.emit(ProcessMessage::DoneLoading).await;
        } else {
            #[cfg(feature = "training")]
            {
                let config = config().await;
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

pub(crate) async fn update_splat_state(
    emitter: &TryStreamEmitter<ProcessMessage, anyhow::Error>,
    splat_state: &Slot<SplatView>,
    view: SplatView,
) {
    splat_state.put(view);
    emitter.emit(ProcessMessage::SplatsUpdated).await;
}
