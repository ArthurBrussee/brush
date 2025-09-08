use crate::{
    config::{ProcessArgs, ProcessConfig, RerunConfig},
    emit_warnings::WarningEmitter,
    eval_export::eval_save_to_disk,
    message::ProcessMessage,
    visualize_tools::VisualizeTools,
};
use anyhow::Context;
use async_fn_stream::TryStreamEmitter;
use brush_dataset::{load_dataset, scene::Scene, scene_loader::SceneLoader};
use brush_render::{
    MainBackend,
    gaussian_splats::{RandomSplatsConfig, Splats},
};
use brush_train::{
    eval::eval_stats,
    msg::{RefineStats, TrainStepStats},
    train::SplatTrainer,
};
use brush_vfs::BrushVfs;
use burn::{backend::Autodiff, module::AutodiffModule, prelude::Backend};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use rand::SeedableRng;
use std::{path::Path, sync::Arc};
use tokio::sync::oneshot::Receiver;
use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

pub(crate) async fn train_stream(
    vfs: Arc<BrushVfs>,
    process_args: Receiver<ProcessArgs>,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
) -> anyhow::Result<()> {
    log::info!("Start of training stream");

    let warner = WarningEmitter::new(&emitter);

    emitter
        .emit(ProcessMessage::StartLoading { training: true })
        .await;

    // Now wait for the process args.
    let process_args = process_args.await?;

    log::info!("Create rerun {}", process_args.rerun_config.rerun_enabled);
    let visualize = VisualizeTools::new(process_args.rerun_config.rerun_enabled);

    let process_config = &process_args.process_config;
    log::info!("Using seed {}", process_config.seed);
    <MainBackend as Backend>::seed(&device, process_config.seed);
    let mut rng = rand::rngs::StdRng::from_seed([process_config.seed as u8; 32]);

    log::info!("Loading dataset");
    let (initial_splats, dataset) =
        load_dataset(vfs.clone(), &process_args.load_config, &device).await?;

    warner
        .warn_if_err(
            visualize.log_scene(&dataset.train, process_args.rerun_config.rerun_max_img_size),
        )
        .await;

    log::info!("Dataset loaded");
    emitter
        .emit(ProcessMessage::Dataset {
            dataset: dataset.clone(),
        })
        .await;

    let estimated_up = dataset.estimate_up();
    log::info!("Loading initial splats if any.");

    if let Some(init) = &initial_splats {
        emitter
            .emit(ProcessMessage::ViewSplats {
                // If the metadata has an up axis prefer that, otherwise estimate
                // the up direction.
                up_axis: init.meta.up_axis.or(Some(estimated_up)),
                splats: Box::new(init.splats.clone()),
                frame: 0,
                total_frames: 0,
                progress: init.meta.progress,
            })
            .await;
    }

    emitter.emit(ProcessMessage::DoneLoading).await;

    let splats = if let Some(init_msg) = initial_splats {
        init_msg.splats
    } else {
        log::info!("Starting with random splat config.");
        // Create a bounding box the size of all the cameras plus a bit.
        let mut bounds = dataset.train.bounds();
        bounds.extent *= 1.25;
        let config = RandomSplatsConfig::new();
        Splats::from_random_config(&config, bounds, &mut rng, &device)
    };

    let splats = splats.with_sh_degree(process_args.model_config.sh_degree);
    let mut splats = splats.into_autodiff();

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(&dataset.train, 42, &device);
    let mut trainer = SplatTrainer::new(&process_args.train_config, &device, splats.clone()).await;

    log::info!("Start training loop.");
    for iter in process_args.process_config.start_iter..process_args.train_config.total_steps {
        let step_time = Instant::now();

        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;

        let (new_splats, stats) = trainer.step(iter, &batch, splats);
        splats = new_splats;
        let (new_splats, refine) = trainer
            .refine_if_needed(iter, splats)
            .instrument(trace_span!("Refine splats"))
            .await;
        splats = new_splats;

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == process_args.train_config.total_steps;

        // Add up time from this step.
        train_duration += step_time.elapsed();

        // Check if we want to evaluate _next iteration_. Small detail, but this ensures we evaluate
        // before doing a refine.
        if (iter % process_config.eval_every == 0 || is_last_step)
            && let Some(eval_scene) = eval_scene.as_mut()
        {
            let res = run_eval(
                &device,
                &emitter,
                &visualize,
                process_config,
                splats.valid(),
                iter,
                eval_scene,
            )
            .await;
            warner
                .warn_if_err(res.context(format!("Failed evaluation at iteration {iter}")))
                .await;
        }

        if iter % process_config.export_every == 0 || is_last_step {
            let res = export_checkpoint(&process_args, process_config, splats.valid(), iter).await;
            warner
                .warn_if_err(res.context(format!("Export at iteration {iter} failed")))
                .await;
        }

        let res = rerun_log(
            &process_args.rerun_config,
            &visualize,
            splats.clone(),
            &stats,
            iter,
            is_last_step,
            &device,
            refine.as_ref(),
        )
        .await;

        warner
            .warn_if_err(res.context("Rerun visualization failed"))
            .await;

        if let Some(stats) = refine {
            emitter
                .emit(ProcessMessage::RefineStep {
                    stats: Box::new(stats),
                    cur_splat_count: splats.num_splats(),
                    iter,
                })
                .await;
        }

        // How frequently to update the UI after a training step.
        const UPDATE_EVERY: u32 = 5;
        if iter % UPDATE_EVERY == 0 || is_last_step {
            let message = ProcessMessage::TrainStep {
                splats: Box::new(splats.valid()),
                stats: Box::new(stats),
                iter,
                total_elapsed: train_duration,
            };
            emitter.emit(message).await;
        }
    }

    Ok(())
}

async fn run_eval(
    device: &WgpuDevice,
    emitter: &TryStreamEmitter<ProcessMessage, anyhow::Error>,
    visualize: &VisualizeTools,
    process_config: &ProcessConfig,
    splats: Splats<MainBackend>,
    iter: u32,
    eval_scene: &Scene,
) -> Result<(), anyhow::Error> {
    let mut psnr = 0.0;
    let mut ssim = 0.0;
    let mut count = 0;
    log::info!("Running evaluation for iteration {iter}");

    for (i, view) in eval_scene.views.iter().enumerate() {
        let eval_img = view.image.load().await?;
        let sample = eval_stats(
            &splats,
            &view.camera,
            eval_img,
            view.image.is_masked(),
            device,
        )
        .context("Failed to run eval for sample.")?;

        count += 1;
        psnr += sample.psnr.clone().into_scalar_async().await;
        ssim += sample.ssim.clone().into_scalar_async().await;

        let export_path = Path::new(&process_config.export_path).to_owned();
        if process_config.eval_save_to_disk {
            let img_name = Path::new(&view.image.path)
                .file_stem()
                .expect("No file name for eval view.")
                .to_string_lossy();
            let path = Path::new(&export_path)
                .join(format!("eval_{iter}"))
                .join(format!("{img_name}.png"));
            eval_save_to_disk(&sample, &path).await?;
        }

        visualize.log_eval_sample(iter, i as u32, sample).await?;
    }
    psnr /= count as f32;
    ssim /= count as f32;
    visualize.log_eval_stats(iter, psnr, ssim)?;
    emitter
        .emit(ProcessMessage::EvalResult {
            iter,
            avg_psnr: psnr,
            avg_ssim: ssim,
        })
        .await;

    Ok(())
}

async fn export_checkpoint(
    process_args: &ProcessArgs,
    process_config: &ProcessConfig,
    splats: Splats<MainBackend>,
    iter: u32,
) -> Result<(), anyhow::Error> {
    // TODO: Want to support this on WASM somehow. Maybe have user pick a file once,
    // and write to it repeatedly?
    #[cfg(not(target_family = "wasm"))]
    {
        use tokio::fs;
        let total_steps = process_args.train_config.total_steps;
        let digits = ((total_steps as f64).log10().floor() as usize) + 1;
        let export_name = process_config
            .export_name
            .replace("{iter}", &format!("{iter:0digits$}"));
        let export_path = Path::new(&process_config.export_path).to_owned();
        fs::create_dir_all(&export_path)
            .await
            .context("Creating export directory")?;
        let splat_data = brush_serde::splat_to_ply(splats)
            .await
            .context("Serializing splat data")?;
        fs::write(export_path.join(&export_name), splat_data)
            .await
            .context(format!("Failed to export ply {export_path:?}"))?;
    }
    #[cfg(target_family = "wasm")]
    {
        let _ = process_args;
        let _ = process_config;
        let _ = splats;
        let _ = iter;
    }
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
