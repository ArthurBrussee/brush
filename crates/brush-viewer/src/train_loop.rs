use async_fn_stream::try_fn_stream;

use brush_dataset::{
    brush_vfs::BrushVfs, scene_loader::SceneLoader, Dataset, LoadDatasetArgs, LoadInitArgs,
};
use brush_render::gaussian_splats::{RandomSplatsConfig, Splats};
use brush_train::train::{SplatTrainer, TrainConfig};
use burn::module::AutodiffModule;
use burn_jit::cubecl::Runtime;
use burn_wgpu::{Wgpu, WgpuDevice, WgpuRuntime};
use rand::SeedableRng;
use tokio::sync::mpsc::{error::TryRecvError, Receiver};
use tokio_stream::{Stream, StreamExt};
use web_time::Instant;

use crate::viewer::ProcessMessage;

const UPDATE_EVERY: u32 = 5;

#[derive(Debug, Clone)]
pub enum TrainMessage {
    Paused(bool),
    Eval { view_count: Option<usize> },
}

pub(crate) fn train_loop(
    vfs: BrushVfs,
    device: WgpuDevice,
    mut receiver: Receiver<TrainMessage>,
    load_data_args: LoadDatasetArgs,
    load_init_args: LoadInitArgs,
    config: TrainConfig,
) -> impl Stream<Item = anyhow::Result<ProcessMessage>> {
    try_fn_stream(|emitter| async move {
        let batch_size = 1;

        // Maybe good if the seed would be configurable.
        let seed = 42;
        <Wgpu as burn::prelude::Backend>::seed(seed);
        let mut rng = rand::rngs::StdRng::from_seed([seed as u8; 32]);

        // Load initial splats if included
        let mut initial_splats = None;

        let mut dataset = Dataset::empty();
        let (mut splat_stream, mut data_stream) =
            brush_dataset::load_dataset(vfs.clone(), &load_data_args, &device).await?;

        // Read initial splats if any.
        while let Some(message) = splat_stream.next().await {
            let message = message?;
            let splats = message.splats.with_sh_degree(load_init_args.sh_degree);
            let msg = ProcessMessage::ViewSplats {
                up_axis: message.meta.up_axis,
                splats: Box::new(splats.valid()),
                frame: 0,
                total_frames: 0,
            };
            emitter.emit(msg).await;
            initial_splats = Some(splats);
        }

        // Read dataset stream.
        while let Some(d) = data_stream.next().await {
            dataset = d?;

            emitter
                .emit(ProcessMessage::Dataset {
                    data: dataset.clone(),
                })
                .await;
        }
        emitter
            .emit(ProcessMessage::DoneLoading { training: true })
            .await;

        let mut splats = if let Some(splats) = initial_splats {
            splats
        } else {
            // By default, spawn the splats in bounds.
            let bounds = dataset.train.bounds();
            let bounds_extent = bounds.extent.length();
            // Arbitrarly assume area of interest is 0.2 - 0.75 of scene bounds.
            // Somewhat specific to the blender scenes
            let adjusted_bounds = dataset
                .train
                .adjusted_bounds(bounds_extent * 0.25, bounds_extent);

            let config = RandomSplatsConfig::new().with_sh_degree(load_init_args.sh_degree);
            Splats::from_random_config(config, adjusted_bounds, &mut rng, &device)
        };

        let train_scene = dataset.train.clone();
        let eval_scene = dataset.eval.clone();

        let mut dataloader = SceneLoader::new(&train_scene, batch_size, seed, &device);
        let mut trainer = SplatTrainer::new(&splats, &config, &device);

        let mut is_paused = false;

        let mut iter = 0;

        loop {
            let message = if is_paused {
                // When paused, wait for a message async and handle it. The "default" train iteration
                // won't be hit.
                receiver.recv().await
            } else {
                // Otherwise, check for messages, and if there isn't any just proceed training.
                match receiver.try_recv() {
                    Ok(message) => Some(message),
                    Err(TryRecvError::Empty) => None, // Nothing special to do.
                    Err(TryRecvError::Disconnected) => break, // If channel is closed, stop.
                }
            };

            match message {
                Some(TrainMessage::Paused(paused)) => {
                    is_paused = paused;
                }
                Some(TrainMessage::Eval { view_count }) => {
                    if let Some(eval_scene) = eval_scene.as_ref() {
                        let eval = brush_train::eval::eval_stats(
                            splats.valid(),
                            eval_scene,
                            view_count,
                            &mut rng,
                            &device,
                        )
                        .await;

                        emitter
                            .emit(ProcessMessage::EvalResult { iter, eval })
                            .await;
                    }
                }
                // By default, continue training.
                None => {
                    let batch = dataloader.next_batch().await;
                    let extent = batch.scene_extent;

                    let (new_splats, stats) = trainer.step(iter, batch, splats)?;
                    let (new_splats, refine) =
                        trainer.refine_if_needed(iter, new_splats, extent).await;

                    iter += 1;

                    splats = new_splats;

                    if iter % UPDATE_EVERY == 0 {
                        emitter
                            .emit(ProcessMessage::TrainStep {
                                splats: Box::new(splats.valid()),
                                stats: Box::new(stats),
                                iter,
                                timestamp: Instant::now(),
                            })
                            .await;
                    }

                    if let Some(refine) = refine {
                        emitter
                            .emit(ProcessMessage::RefineStep {
                                stats: Box::new(refine),
                                iter,
                            })
                            .await;
                    }
                }
            }

            // On the first iteration, wait for the backend to catch up. It likely kicks off a flurry of autotuning,
            // and on web where this isn't cached causes a real slowdown. Autotuning takes forever as the GPU is
            // busy with our work. This is only needed on wasm - on native autotuning is
            // synchronous anyway.
            if cfg!(target_family = "wasm") && iter < 5 {
                // Wait for them all to be done.
                let client = WgpuRuntime::client(&device);
                client.sync().await;
            }
        }

        Ok(())
    })
}
