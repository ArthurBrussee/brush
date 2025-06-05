use crate::message::ProcessMessage;

use std::sync::Arc;

use async_fn_stream::TryStreamEmitter;
use brush_dataset::splat_import;
use brush_vfs::BrushVfs;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use tokio_stream::StreamExt;

pub(crate) async fn view_stream(
    vfs: Arc<BrushVfs>,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
) -> anyhow::Result<()> {
    let mut paths: Vec<_> = vfs.file_paths().collect();
    alphanumeric_sort::sort_path_slice(&mut paths);
    let client = WgpuRuntime::client(&device);

    for (i, path) in paths.iter().enumerate() {
        log::info!("Loading single ply file");

        emitter
            .emit(ProcessMessage::StartLoading { training: false })
            .await;

        let sub_sample = None; // Subsampling a trained ply doesn't really make sense.
        let splat_stream = splat_import::load_splat_from_ply(
            vfs.reader_at_path(path).await?,
            sub_sample,
            device.clone(),
        );

        let mut splat_stream = std::pin::pin!(splat_stream);

        while let Some(message) = splat_stream.next().await {
            let message = message?;

            // If there's multiple ply files in a zip, don't support animated plys, that would
            // get rather mind bending.
            let (frame, total_frames) = if paths.len() == 1 {
                (message.meta.current_frame, message.meta.frame_count)
            } else {
                (i as u32, paths.len() as u32)
            };

            let view_splat_msg = ProcessMessage::ViewSplats {
                up_axis: message.meta.up_axis,
                splats: Box::new(message.splats),
                frame,
                total_frames,
            };

            // As loading concatenates splats each time, memory usage tends to accumulate a lot
            // over time. Clear out memory after each step to prevent this buildup.
            client.memory_cleanup();

            emitter.emit(view_splat_msg).await;
        }
    }

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Clear out memory after loading is fully done.
    client.memory_cleanup();

    Ok(())
}
