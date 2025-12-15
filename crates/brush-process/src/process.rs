use async_fn_stream::try_fn_stream;
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use tokio::sync::oneshot::Receiver;
use tokio_stream::Stream;

#[allow(unused)]
use brush_serde;

use crate::{config::TrainStreamConfig, message::ProcessMessage, view_stream::view_stream};

pub fn create_process(
    source: DataSource,
    process_args: Receiver<TrainStreamConfig>,
    device: WgpuDevice,
) -> impl Stream<Item = Result<ProcessMessage, anyhow::Error>> + 'static {
    try_fn_stream(|emitter| async move {
        log::info!("Starting process with source {source:?}");
        emitter.emit(ProcessMessage::NewProcess).await;

        let vfs = source.into_vfs().await?;

        let vfs_counts = vfs.file_count();
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
            .emit(ProcessMessage::NewSource { name: source_name })
            .await;

        if !is_training {
            drop(process_args);
            view_stream(vfs, device, emitter).await?;
        } else {
            #[cfg(feature = "training")]
            crate::train_stream::train_stream(vfs, process_args, device, emitter).await?;
            #[cfg(not(feature = "training"))]
            anyhow::bail!("Training is not enabled in Brush, cannot load dataset.");
        };

        Ok(())
    })
}
