/// A default training loop for Brush.
use async_fn_stream::try_fn_stream;

use brush_dataset::{Dataset, scene_loader::SceneLoader};
use brush_render::gaussian_splats::Splats;
use brush_train::train::TrainBack;
use brush_train::train::{RefineStats, SplatTrainer, TrainConfig, TrainStepStats};

use burn::{module::AutodiffModule, tensor::backend::AutodiffBackend};
use burn_wgpu::WgpuDevice;
use tokio_stream::Stream;
use web_time::Instant;

pub enum TrainMessage {
    TrainStep {
        splats: Box<Splats<<TrainBack as AutodiffBackend>::InnerBackend>>,
        stats: Box<TrainStepStats<TrainBack>>,
        iter: u32,
        timestamp: Instant,
    },
    RefineStep {
        stats: Box<RefineStats>,
        iter: u32,
    },
}

// False positive: need to pass in TrainConfig by value to keep lifetimes sane.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn train_stream(
    dataset: Dataset,
    initial_splats: Splats<TrainBack>,
    config: TrainConfig,
    device: WgpuDevice,
    start_iter: u32,
) -> impl Stream<Item = anyhow::Result<TrainMessage>> {
    try_fn_stream(|emitter| async move {
        let mut splats = initial_splats;

        let train_scene = dataset.train.clone();

        let mut dataloader = SceneLoader::new(&train_scene, 42, &device);

        let scene_extent = train_scene.estimate_extent().unwrap_or(1.0);
        let mut trainer = SplatTrainer::new(&config, &device);

        let mut iter = start_iter;

        #[allow(clippy::infinite_loop)]
        loop {
            let batch = dataloader.next_batch().await;

            let (new_splats, stats) = trainer.step(scene_extent, iter, batch, splats);
            let (new_splats, refine) = trainer
                .refine_if_needed(iter, new_splats, scene_extent)
                .await;
            splats = new_splats;

            emitter
                .emit(TrainMessage::TrainStep {
                    splats: Box::new(splats.valid()),
                    stats: Box::new(stats),
                    iter,
                    timestamp: Instant::now(),
                })
                .await;

            if let Some(refine) = refine {
                emitter
                    .emit(TrainMessage::RefineStep {
                        stats: Box::new(refine),
                        iter,
                    })
                    .await;
            }

            iter += 1;
        }
    })
}
