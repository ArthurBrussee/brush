use std::sync::Arc;

use async_fn_stream::try_fn_stream;
use brush_vfs::DataSource;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use tokio::sync::oneshot::Receiver;
use tokio_stream::Stream;

#[allow(unused)]
use brush_dataset::splat_export;

use crate::{
    config::ProcessArgs, message::ProcessMessage, train_stream::train_stream,
    view_stream::view_stream,
};

pub fn process_stream(
    source: DataSource,
    process_args: Receiver<ProcessArgs>,
    device: WgpuDevice,
) -> impl Stream<Item = Result<ProcessMessage, anyhow::Error>> + 'static {
    try_fn_stream(|emitter| async move {
        log::info!("Starting process with source {source:?}");
        emitter.emit(ProcessMessage::NewSource).await;

        let vfs = Arc::new(source.into_vfs().await?);

        let client = WgpuRuntime::client(&device);
        // Start with memory cleared out.
        client.memory_cleanup();

        let vfs_counts = vfs.file_count();
        let ply_count = vfs.files_with_extension("ply").count();

        log::info!(
            "Mounted VFS with {} files. (plys: {})",
            vfs.file_count(),
            ply_count
        );

        log::info!("Start of view stream");

        if vfs_counts == ply_count {
            drop(process_args);
            view_stream(vfs, device, emitter).await?;
        } else {
            // Receive the processing args.
            train_stream(vfs, process_args, device, emitter).await?;
        };

        Ok(())
    })
}
