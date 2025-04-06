use std::sync::Arc;

use crate::process_loop::view_stream::view_stream;
use async_fn_stream::try_fn_stream;
use brush_msg::{DataSource, ProcessMessage, config::ProcessArgs};
use burn_wgpu::WgpuDevice;
use tokio_stream::Stream;

#[allow(unused)]
use brush_dataset::splat_export;

use super::train_stream::train_stream;

pub fn process_stream(
    source: DataSource,
    process_args: ProcessArgs,
    device: WgpuDevice,
) -> impl Stream<Item = Result<ProcessMessage, anyhow::Error>> + 'static {
    try_fn_stream(|emitter| async move {
        log::info!("Starting process with source {source:?}");

        emitter.emit(ProcessMessage::NewSource).await;

        let vfs = source.into_vfs().await;

        let vfs = match vfs {
            Ok(vfs) => Arc::new(vfs),
            Err(e) => {
                anyhow::bail!(e);
            }
        };

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
