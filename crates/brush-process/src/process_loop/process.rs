use async_fn_stream::try_fn_stream;
use burn::tensor::backend::AutodiffBackend;
use web_time::Duration;

use crate::{data_source::DataSource, process_loop::view_stream::view_stream};
use brush_dataset::Dataset;
use brush_render::gaussian_splats::Splats;
use brush_train::train::{RefineStats, TrainBack, TrainStepStats};
use burn_wgpu::WgpuDevice;
use glam::Vec3;
use tokio_stream::Stream;

#[allow(unused)]
use brush_dataset::splat_export;

use super::{ProcessArgs, train_stream::train_stream};

pub enum ProcessMessage {
    NewSource,
    StartLoading {
        training: bool,
    },
    /// Loaded a splat from a ply file.
    ///
    /// Nb: This includes all the intermediately loaded splats.
    /// Nb: Animated splats will have the 'frame' number set.
    ViewSplats {
        up_axis: Option<Vec3>,
        splats: Box<Splats<<TrainBack as AutodiffBackend>::InnerBackend>>,
        frame: u32,
        total_frames: u32,
    },
    /// Loaded a bunch of viewpoints to train on.
    Dataset {
        data: Dataset,
    },
    /// Splat, or dataset and initial splat, are done loading.
    #[allow(unused)]
    DoneLoading {
        training: bool,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    TrainStep {
        splats: Box<Splats<<TrainBack as AutodiffBackend>::InnerBackend>>,
        stats: Box<TrainStepStats<TrainBack>>,
        iter: u32,
        total_elapsed: Duration,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    RefineStep {
        stats: Box<RefineStats>,
        cur_splat_count: u32,
        iter: u32,
    },
    /// Eval was run successfully with these results.
    #[allow(unused)]
    EvalResult {
        iter: u32,
        avg_psnr: f32,
        avg_ssim: f32,
    },
}

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
            Ok(vfs) => vfs,
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
