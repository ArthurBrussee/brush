use crate::{
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
};
use anyhow::Context;
use async_fn_stream::TryStreamEmitter;
use brush_dataset::{load_dataset, scene::Scene, scene_loader::SceneLoader, splat_data_to_splats};
use brush_render::{MainBackend, gaussian_splats::Splats};
use brush_rerun::{RerunConfig, visualize_tools::VisualizeTools};
use brush_train::{
    RandomSplatsConfig, create_random_splats,
    eval::eval_stats,
    msg::{RefineStats, TrainStepStats},
    splats_into_autodiff,
    train::SplatTrainer,
};
use brush_vfs::BrushVfs;
use burn::{backend::Autodiff, module::AutodiffModule, prelude::Backend};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use rand::SeedableRng;
use std::{path::PathBuf, sync::Arc};

#[allow(unused)]
use std::path::Path;

use tokio::sync::oneshot::Receiver;
use tokio_with_wasm::alias as tokio_wasm;
use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

pub(crate) async fn train_stream(
    vfs: Arc<BrushVfs>,
    process_args: Receiver<TrainStreamConfig>,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
) -> anyhow::Result<()> {
    log::info!("Start of training stream");

    emitter
        .emit(ProcessMessage::StartLoading { training: true })
        .await;

    // Now wait for the process args.
    let train_stream_args = process_args.await?;

    let visualize = tracing::trace_span!("Create rerun")
        .in_scope(|| VisualizeTools::new(train_stream_args.rerun_config.rerun_enabled));

    let process_config = &train_stream_args.process_config;
    log::info!("Using seed {}", process_config.seed);
    <MainBackend as Backend>::seed(&device, process_config.seed);
    let mut rng = rand::rngs::StdRng::from_seed([process_config.seed as u8; 32]);

    log::info!("Loading dataset");
    let (initial_splats, dataset) = load_dataset(vfs.clone(), &train_stream_args.load_config)
        .instrument(trace_span!("Load dataset"))
        .await?;

    log::info!("Log scene to rerun");
    if let Err(error) = visualize.log_scene(
        &dataset.train,
        train_stream_args.rerun_config.rerun_max_img_size,
    ) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    if let Err(error) = visualize.log_scene(
        &dataset.train,
        train_stream_args.rerun_config.rerun_max_img_size,
    ) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    log::info!("Dataset loaded");
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
            dataset: dataset.clone(),
        }))
        .await;

    let estimated_up = dataset.estimate_up();
    log::info!("Loading initial splats if any.");

    // Convert SplatData to Splats using KNN initialization
    let initial_splats = initial_splats.map(|msg| {
        let splats = splat_data_to_splats(msg.data, &device);
        (msg.meta, splats)
    });

    if let Some((meta, splats)) = &initial_splats {
        emitter
            .emit(ProcessMessage::ViewSplats {
                // If the metadata has an up axis prefer that, otherwise estimate
                // the up direction.
                up_axis: meta.up_axis.or(Some(estimated_up)),
                splats: Box::new(splats.clone()),
                frame: 0,
                total_frames: 0,
                progress: meta.progress,
            })
            .await;
    }

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Start with memory cleared out.
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let splats = if let Some((_, splats)) = initial_splats {
        splats
    } else {
        log::info!("Starting with random splat config.");
        // Create a bounding box the size of all the cameras plus a bit.
        let mut bounds = dataset.train.bounds();
        bounds.extent *= 1.25;
        let config = RandomSplatsConfig::new();
        create_random_splats(&config, bounds, &mut rng, &device)
    };

    let splats = splats.with_sh_degree(train_stream_args.model_config.sh_degree);
    let mut splats = splats_into_autodiff(splats);

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(&dataset.train, 42);
    let mut trainer =
        SplatTrainer::new(&train_stream_args.train_config, &device, splats.clone()).await;

    let export_path = if let Some(base_path) = vfs.base_path() {
        base_path.join("exports")
    } else {
        // Defaults to CWD.
        PathBuf::from("./")
    };

    let export_path = export_path.join(&train_stream_args.process_config.export_path);
    // Normalize path components
    let export_path: PathBuf = export_path.components().collect();

    log::info!("Start training loop.");
    for iter in
        train_stream_args.process_config.start_iter..train_stream_args.train_config.total_steps
    {
        let step_time = Instant::now();

        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;
        let (new_splats, stats) = trainer.step(batch, splats);
        splats = new_splats;
        let (new_splats, refine) = trainer
            .refine_if_needed(iter, splats)
            .instrument(trace_span!("Refine splats"))
            .await;
        splats = new_splats;

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == train_stream_args.train_config.total_steps;

        // Add up time from this step.
        train_duration += step_time.elapsed();

        // Check if we want to evaluate _next iteration_. Small detail, but this ensures we evaluate
        // before doing a refine.
        if (iter % process_config.eval_every == 0 || is_last_step)
            && let Some(eval_scene) = eval_scene.as_mut()
        {
            let save_path = train_stream_args
                .process_config
                .eval_save_to_disk
                .then(|| export_path.clone());

            let res = run_eval(
                &device,
                &emitter,
                &visualize,
                splats.valid(),
                iter,
                eval_scene,
                save_path,
            )
            .await
            .with_context(|| format!("Failed evaluation at iteration {iter}"));

            if let Err(error) = res {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
        }

        #[cfg(not(target_family = "wasm"))]
        if iter % process_config.export_every == 0 || is_last_step {
            let res = export_checkpoint(
                splats.valid(),
                &export_path,
                &process_config.export_name,
                iter,
                train_stream_args.train_config.total_steps,
            )
            .await
            .with_context(|| format!("Export at iteration {iter} failed"));

            if let Err(error) = res {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
        }

        let res = rerun_log(
            &train_stream_args.rerun_config,
            &visualize,
            splats.clone(),
            &stats,
            iter,
            is_last_step,
            &device,
            refine.as_ref(),
        )
        .await
        .context("Rerun visualization failed");

        if let Err(error) = res {
            emitter.emit(ProcessMessage::Warning { error }).await;
        }

        if refine.is_some() {
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                    cur_splat_count: splats.num_splats(),
                    iter,
                }))
                .await;
        }

        // How frequently to update the UI after a training step.
        const UPDATE_EVERY: u32 = 5;
        if iter % UPDATE_EVERY == 0 || is_last_step {
            let message = ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                splats: Box::new(splats.valid()),
                iter,
                total_steps: train_stream_args.train_config.total_steps,
                total_elapsed: train_duration,
            });
            emitter.emit(message).await;
        }
    }

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::DoneTraining))
        .await;
    Ok(())
}

async fn run_eval(
    device: &WgpuDevice,
    emitter: &TryStreamEmitter<ProcessMessage, anyhow::Error>,
    visualize: &VisualizeTools,
    splats: Splats<MainBackend>,
    iter: u32,
    eval_scene: &Scene,
    save_path: Option<PathBuf>,
) -> Result<(), anyhow::Error> {
    let mut psnr = 0.0;
    let mut ssim = 0.0;
    let mut count = 0;
    log::info!("Running evaluation for iteration {iter}");

    for (i, view) in eval_scene.views.iter().enumerate() {
        tokio_wasm::task::yield_now().await;

        let eval_img = view.image.load().await?;
        let sample = eval_stats(
            &splats,
            &view.camera,
            eval_img,
            view.image.alpha_mode(),
            device,
        )
        .context("Failed to run eval for sample.")?;

        count += 1;
        psnr += sample.psnr.clone().into_scalar_async().await?;
        ssim += sample.ssim.clone().into_scalar_async().await?;

        #[cfg(not(target_family = "wasm"))]
        if let Some(path) = &save_path {
            let img_name = view.image.img_name();
            let path = path
                .join(format!("eval_{iter}"))
                .join(format!("{img_name}.png"));
            sample.save_to_disk(&path).await?;
        }

        #[cfg(target_family = "wasm")]
        let _ = save_path;

        visualize.log_eval_sample(iter, i as u32, sample).await?;
    }
    psnr /= count as f32;
    ssim /= count as f32;
    visualize.log_eval_stats(iter, psnr, ssim)?;
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::EvalResult {
            iter,
            avg_psnr: psnr,
            avg_ssim: ssim,
        }))
        .await;

    Ok(())
}

// TODO: Want to support this on WASM somehow. Maybe have user pick a file once,
// and write to it repeatedly?
#[cfg(not(target_family = "wasm"))]
async fn export_checkpoint(
    splats: Splats<MainBackend>,
    export_path: &Path,
    export_name: &str,
    iter: u32,
    total_steps: u32,
) -> Result<(), anyhow::Error> {
    tokio::fs::create_dir_all(&export_path)
        .await
        .with_context(|| format!("Creating export directory {}", export_path.display()))?;
    let digits = ((total_steps as f64).log10().floor() as usize) + 1;
    let export_name = export_name.replace("{iter}", &format!("{iter:0digits$}"));
    let splat_data = brush_serde::splat_to_ply(splats)
        .await
        .context("Serializing splat data")?;
    tokio::fs::write(export_path.join(&export_name), splat_data)
        .await
        .context(format!("Failed to export ply {export_path:?}"))?;
    Ok(())
}

async fn rerun_log(
    rerun_config: &RerunConfig,
    visualize: &VisualizeTools,
    splats: Splats<Autodiff<MainBackend>>,
    stats: &TrainStepStats<MainBackend>,
    iter: u32,
    is_last_step: bool,
    device: &WgpuDevice,
    refine: Option<&RefineStats>,
) -> Result<(), anyhow::Error> {
    visualize.log_splat_stats(iter, &splats)?;

    if let Some(every) = rerun_config.rerun_log_splats_every
        && (iter.is_multiple_of(every) || is_last_step)
    {
        visualize.log_splats(iter, splats.valid()).await?;
    }
    // Log out train stats.
    if iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step {
        visualize.log_train_stats(iter, stats.clone()).await?;
    }
    let client = WgpuRuntime::client(device);
    visualize.log_memory(iter, &client.memory_usage())?;
    // Emit some messages. Important to not count these in the training time (as this might pause).
    if let Some(stats) = refine {
        visualize.log_refine_stats(iter, stats)?;
    }
    Ok(())
}
