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
        emitter.emit(ProcessMessage::NewSource).await;

        let vfs = source.into_vfs().await?;

        let vfs_counts = vfs.file_count();
        let ply_count = vfs.files_with_extension("ply").count();

        log::info!(
            "Mounted VFS with {} files. (plys: {})",
            vfs.file_count(),
            ply_count
        );

        if vfs_counts == ply_count {
            drop(process_args);
            view_stream(vfs, device, emitter).await?;
        } else {
            #[cfg(feature = "training")]
            crate::train_stream::train_stream(vfs, process_args, device, emitter).await?;
        };

        Ok(())
    })
}
