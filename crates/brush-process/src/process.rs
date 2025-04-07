use std::sync::Arc;

use async_fn_stream::try_fn_stream;
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use tokio_stream::Stream;

#[allow(unused)]
use brush_dataset::splat_export;

use crate::{
    config::ProcessArgs, message::ProcessMessage, train_stream::train_stream,
    view_stream::view_stream,
};

pub fn process_stream(
    source: DataSource,
    process_args: ProcessArgs,
    device: WgpuDevice,
) -> impl Stream<Item = Result<ProcessMessage, anyhow::Error>> + 'static {
    try_fn_stream(|emitter| async move {
        log::info!("Starting process with source {source:?}");
        let vfs = Arc::new(source.into_vfs().await?);

        let paths: Vec<_> = vfs.file_names().collect();
        log::info!("Mounted VFS with {} files", paths.len());

        if paths
            .iter()
            .all(|p| p.extension().is_some_and(|p| p == "ply"))
        {
            view_stream(vfs, device, emitter).await?;
        } else {
            train_stream(vfs, process_args, device, emitter).await?;
        };
        Ok(())
    })
}
