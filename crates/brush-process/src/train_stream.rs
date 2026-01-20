use crate::{
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::Slot,
};
use anyhow::Context;
use async_fn_stream::TryStreamEmitter;
use brush_dataset::{load_dataset, scene::Scene, scene_loader::SceneLoader};
use brush_render::{
    MainBackend,
    gaussian_splats::{SplatRenderMode, Splats},
};
use brush_rerun::visualize_tools::VisualizeTools;
use brush_train::{
    RandomSplatsConfig, create_random_splats,
    eval::eval_stats,
    msg::RefineStats,
    splats_into_autodiff, to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_vfs::BrushVfs;
use burn::{module::AutodiffModule, prelude::Backend};
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use rand::SeedableRng;
use std::{path::PathBuf, sync::Arc};

#[allow(unused)]
use std::path::Path;

use tokio_with_wasm::alias as tokio_wasm;
use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

pub(crate) async fn train_stream(
    vfs: Arc<BrushVfs>,
    train_stream_config: TrainStreamConfig,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
    splat_slot: Slot<Splats<MainBackend>>,
) -> anyhow::Result<()> {
    log::info!("Start of training stream");

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
            config: Box::new(train_stream_config.clone()),
        }))
        .await;

    let visualize = tracing::trace_span!("Create rerun")
        .in_scope(|| VisualizeTools::new(train_stream_config.rerun_config.rerun_enabled));

    let process_config = &train_stream_config.process_config;
    log::info!("Using seed {}", process_config.seed);

    <MainBackend as Backend>::seed(&device, process_config.seed);
    let mut rng = rand::rngs::StdRng::from_seed([process_config.seed as u8; 32]);

    log::info!("Loading dataset");
    let load_result = load_dataset(vfs.clone(), &train_stream_config.load_config)
        .instrument(trace_span!("Load dataset"))
        .await?;

    // Emit any warnings from dataset loading.
    for warning in load_result.warnings {
        emitter
            .emit(ProcessMessage::Warning {
                error: anyhow::anyhow!("{warning}"),
            })
            .await;
    }

    let dataset = load_result.dataset;

    log::info!("Log scene to rerun");
    if let Err(error) = visualize.log_scene(
        &dataset.train,
        train_stream_config.rerun_config.rerun_max_img_size,
    ) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    if let Err(error) = visualize.log_scene(
        &dataset.train,
        train_stream_config.rerun_config.rerun_max_img_size,
    ) {
        emitter.emit(ProcessMessage::Warning { error }).await;
    }

    log::info!("Dataset loaded");
    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::Dataset {
            dataset: dataset.clone(),
        }))
        .await;

    log::info!("Loading initial splats if any.");
    let estimated_up = dataset.estimate_up();

    // Convert SplatData to Splats using KNN initialization
    let (up_axis, init_splats) = if let Some(msg) = load_result.init_splat {
        // Use loaded splats with KNN init
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .or(msg.meta.render_mode)
            .unwrap_or(SplatRenderMode::Default);
        let splats = to_init_splats(msg.data, render_mode, &device);
        (msg.meta.up_axis, splats)
    } else {
        // Default: just use random splats
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);
        log::info!("Starting with random splat config.");
        // Create a bounding box the size of all the cameras plus a bit.
        let mut bounds = dataset.train.bounds();
        bounds.extent *= 1.25;
        let config = RandomSplatsConfig::new();
        let splats = create_random_splats(&config, bounds, &mut rng, render_mode, &device);
        (None, splats)
    };

    let init_splats = init_splats.with_sh_degree(train_stream_config.model_config.sh_degree);

    // If the metadata has an up axis prefer that, otherwise estimate the up direction.
    let up_axis = up_axis.or(Some(estimated_up));

    splat_slot.set(init_splats.clone()).await;
    emitter
        .emit(ProcessMessage::SplatsUpdated {
            up_axis,
            frame: 0,
            total_frames: 1,
            num_splats: init_splats.num_splats(),
            sh_degree: init_splats.sh_degree(),
        })
        .await;

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Start with memory cleared out.
    let client = WgpuRuntime::client(&device);
    client.memory_cleanup();

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(&dataset.train, 42);
    let bounds = get_splat_bounds(init_splats.clone(), BOUND_PERCENTILE).await;
    let mut trainer = SplatTrainer::new(&train_stream_config.train_config, &device, bounds);

    // Get the dataset name from the base path (if available) for interpolation.
    let dataset_name = vfs
        .base_path()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "dataset".to_owned());

    // Interpolate {dataset} in the export path.
    let export_path_str = train_stream_config
        .process_config
        .export_path
        .replace("{dataset}", &dataset_name);

    // Resolve relative to the dataset's parent directory if available, otherwise CWD.
    let base_path = vfs
        .base_path()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let export_path = base_path.join(&export_path_str);
    // Normalize path components
    let export_path: PathBuf = export_path.components().collect();
    let sh_degree = init_splats.sh_degree();

    log::info!("Start training loop.");
    for iter in
        train_stream_config.process_config.start_iter..train_stream_config.train_config.total_steps
    {
        let step_time = Instant::now();

        // Wait for next batch.
        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;

        let stats = splat_slot
            .act(0, |splats| async {
                let (new_splats, stats) = trainer.step(batch, splats_into_autodiff(splats)).await;
                (new_splats.valid(), stats)
            })
            .await
            .unwrap();

        let train_t =
            (iter as f32 / train_stream_config.train_config.total_steps as f32).clamp(0.0, 1.0);

        let refine = if iter > 0
            && iter.is_multiple_of(train_stream_config.train_config.refine_every)
            && train_t <= 0.95
        {
            splat_slot
                .act(0, async |splats| trainer.refine(iter, splats).await)
                .await
                .unwrap()
        } else {
            let new_total: u32 = splat_slot.map(0, |s| s.num_splats()).await.unwrap();
            RefineStats {
                num_added: 0,
                num_pruned: 0,
                total_splats: new_total,
            }
        };

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == train_stream_config.train_config.total_steps;

        // Add up time from this step.
        train_duration += step_time.elapsed();

        // Check if we want to evaluate _next iteration_. Small detail, but this ensures we evaluate
        // before doing a refine.
        if (iter % process_config.eval_every == 0 || is_last_step)
            && let Some(eval_scene) = eval_scene.as_mut()
        {
            let save_path = train_stream_config
                .process_config
                .eval_save_to_disk
                .then(|| export_path.clone());

            let res = run_eval(
                &device,
                &emitter,
                &visualize,
                splat_slot.clone_main().await.unwrap(),
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
                splat_slot.clone_main().await.unwrap(),
                &export_path,
                &process_config.export_name,
                iter,
                train_stream_config.train_config.total_steps,
            )
            .await
            .with_context(|| format!("Export at iteration {iter} failed"));

            if let Err(error) = res {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
        }

        {
            let rerun_config = &train_stream_config.rerun_config;
            visualize
                .log_splat_stats(iter, refine.total_splats)
                .unwrap();

            if let Some(every) = rerun_config.rerun_log_splats_every
                && (iter.is_multiple_of(every) || is_last_step)
            {
                let splats = splat_slot.clone_main().await.unwrap();
                visualize.log_splats(iter, splats).await.unwrap();
            }

            // Log out train stats.
            if iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step {
                visualize
                    .log_train_stats(iter, stats.clone())
                    .await
                    .unwrap();
            }

            visualize.log_memory(iter, &WgpuRuntime::client(&device).memory_usage())?;
            if refine.num_added > 0 {
                visualize.log_refine_stats(iter, &refine).unwrap();
            }
        }

        if refine.num_added > 0 {
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                    cur_splat_count: refine.total_splats,
                    iter,
                }))
                .await;
        }

        // How frequently to update the UI after a training step.
        const UPDATE_EVERY: u32 = 5;
        if iter % UPDATE_EVERY == 0 || is_last_step {
            emitter
                .emit(ProcessMessage::SplatsUpdated {
                    up_axis: None,
                    frame: 0,
                    total_frames: 1,
                    num_splats: refine.total_splats,
                    sh_degree,
                })
                .await;

            // Send lightweight message with training progress.
            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter,
                    total_steps: train_stream_config.train_config.total_steps,
                    total_elapsed: train_duration,
                }))
                .await;
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
            splats.clone(),
            &view.camera,
            eval_img,
            view.image.alpha_mode(),
            device,
        )
        .await
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
