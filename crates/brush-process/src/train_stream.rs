use crate::{
    Emitter,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
    wait_for_device,
};
use anyhow::Context;
use brush_dataset::{load_dataset, scene::Scene, scene_loader::SceneLoader};
use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_rerun::visualize_tools::VisualizeTools;
use brush_train::{
    RandomSplatsConfig, create_random_splats,
    eval::eval_stats,
    lod::{compute_pup_scores, decimate_to_count},
    msg::RefineStats,
    to_init_splats,
    train::{BOUND_PERCENTILE, SplatTrainer, get_splat_bounds},
};
use brush_vfs::BrushVfs;
use burn::module::AutodiffModule;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{AutoCompiler, WgpuRuntime};
use rand::SeedableRng;
use std::{path::PathBuf, sync::Arc};

#[allow(unused)]
use std::path::Path;

use tracing::{Instrument, trace_span};
use web_time::{Duration, Instant};

#[allow(clippy::large_stack_frames)]
pub(crate) async fn train_stream(
    vfs: Arc<BrushVfs>,
    train_stream_config: TrainStreamConfig,
    emitter: &Emitter,
    slot: SlotSender<Splats>,
) -> anyhow::Result<()> {
    log::info!("Start of training stream");

    let visualize = VisualizeTools::new(train_stream_config.rerun_config.rerun_enabled).await;

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::TrainConfig {
            config: Box::new(train_stream_config.clone()),
        }))
        .await;

    let process_config = &train_stream_config.process_config;
    log::info!("Using seed {}", process_config.seed);

    let wgpu_device = wait_for_device().await;
    // Splats live on the inner (non-autodiff) device between steps; each
    // training step lifts them via [`lift_splats_to_autodiff`] then strips
    // back via `.valid()`. Going through `Module::train()` would hit
    // burn-dispatch's `from_inner` checkpointing bug.
    let device: burn::tensor::Device = wgpu_device.clone().into();
    device.seed(process_config.seed);
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

    let num_eval_views = dataset.eval.as_ref().map_or(0, |s| s.views.len());
    if let Err(error) = visualize.send_default_blueprint(num_eval_views) {
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
        let max_splats = train_stream_config.train_config.max_splats as usize;
        let original = msg.data.num_splats();
        let data = msg.data.subsample(max_splats);
        if data.num_splats() < original {
            emitter
                .emit(ProcessMessage::Warning {
                    error: anyhow::anyhow!(
                        "Initial point cloud has {original} points, exceeding --max-splats ({max_splats}). Subsampled to {}; the remaining points were discarded. Raise --max-splats to keep more.",
                        data.num_splats()
                    ),
                })
                .await;
        }
        let splats = to_init_splats(data, render_mode, &device);
        (msg.meta.up_axis, splats)
    } else {
        // Default: just use random splats
        let render_mode = train_stream_config
            .train_config
            .render_mode
            .unwrap_or(SplatRenderMode::Default);
        log::info!("Starting with random splat config.");
        let cameras: Vec<_> = dataset.train.views.iter().map(|v| v.camera).collect();
        let config = RandomSplatsConfig::new();
        let scene_scale = train_stream_config.train_config.random_init_scene_scale;
        let splats = create_random_splats(
            &config,
            &cameras,
            scene_scale,
            &mut rng,
            render_mode,
            &device,
        );
        (None, splats)
    };

    let init_splats = init_splats.with_sh_degree(train_stream_config.model_config.sh_degree);

    // If the metadata has an up axis prefer that, otherwise estimate the up direction.
    let up_axis = up_axis.or(Some(estimated_up));

    // The trainer owns its working `splats` locally and publishes a
    // clone to the `Slot` after every modification (train
    // step, refine, LOD decimation).
    let mut splats: Splats = init_splats.clone();
    slot.set(0, splats.clone());
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
    let client = WgpuRuntime::<AutoCompiler>::client(wgpu_device);
    client.memory_cleanup();

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(&dataset.train, 42, &train_stream_config.load_config);
    let bounds = get_splat_bounds(init_splats.clone(), BOUND_PERCENTILE).await;

    // Per-train-view (world center, focal-px at native res) for the
    // Mip-Splatting 3D filter (always on).
    let mut view_cams: Vec<(glam::Vec3, f32)> = Vec::with_capacity(dataset.train.views.len());
    for view in dataset.train.views.iter() {
        let (w, h) = view.image.dimensions().await.unwrap_or((1, 1));
        let focal = view.camera.focal(glam::uvec2(w, h)).x;
        view_cams.push((view.camera.position, focal));
    }

    let mut trainer = SplatTrainer::new(&train_stream_config.train_config, &device, bounds);
    trainer.set_view_cams(view_cams.clone());

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

    let training_steps = train_stream_config.train_config.total_train_iters;
    let lod_levels = train_stream_config.train_config.lod_levels;
    let lod_refine_steps = train_stream_config.train_config.lod_refine_steps;
    let mut current_lod: u32 = 0;

    let process_config = &train_stream_config.process_config;

    log::info!("Start training loop.");
    for iter in process_config.start_iter..train_stream_config.train_config.total_iters() {
        let target_lod = if lod_levels == 0 || iter < training_steps {
            0u32
        } else {
            ((iter - training_steps) / lod_refine_steps + 1).min(lod_levels)
        };

        if target_lod > current_lod {
            #[cfg(not(target_family = "wasm"))]
            {
                let (name, exp_iter, exp_total) = if current_lod == 0 {
                    (process_config.export_name.clone(), iter, training_steps)
                } else {
                    let lod_name = process_config
                        .export_name
                        .replace(".ply", &format!("_lod{current_lod}.ply"));
                    (lod_name, lod_refine_steps, lod_refine_steps)
                };
                let res =
                    export_checkpoint(splats.clone(), &export_path, &name, exp_iter, exp_total)
                        .await
                        .with_context(|| "Export at LOD boundary failed");

                if let Err(error) = res {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }

            current_lod = target_lod;
            let lod_keep_pct = train_stream_config.train_config.lod_decimation_keep;
            let lod_img_pct = train_stream_config.train_config.lod_image_scale;

            log::info!("LOD {current_lod}/{lod_levels}: Decimating (keep {lod_keep_pct}%)");

            let before = splats.num_splats();
            let target_count = (before as f32 * lod_keep_pct as f32 / 100.0).max(1.0) as u32;

            log::info!("LOD {current_lod}/{lod_levels}: Computing sensitivity scores...");
            let scores = compute_pup_scores(splats.clone(), &dataset.train, &device).await;
            splats = decimate_to_count(splats, &scores, target_count).await;
            slot.set(0, splats.clone());

            let after = splats.num_splats();
            log::info!("LOD {current_lod}/{lod_levels}: {before} -> {after} splats");

            let client = WgpuRuntime::<AutoCompiler>::client(wgpu_device);
            client.memory_cleanup();

            let cumulative_scale = (lod_img_pct as f32 / 100.0).powi(current_lod as i32);
            dataloader = if lod_img_pct < 100 {
                let lod_scene = dataset.train.clone().with_image_scale(cumulative_scale);
                SceneLoader::new(&lod_scene, 42, &train_stream_config.load_config)
            } else {
                SceneLoader::new(&dataset.train, 42, &train_stream_config.load_config)
            };

            let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
            trainer = SplatTrainer::new(&train_stream_config.train_config, &device, bounds);
            trainer.set_view_cams(view_cams.clone());

            log::info!(
                "LOD {current_lod}/{lod_levels}: Training for {lod_refine_steps} steps (image scale {:.0}%)",
                cumulative_scale * 100.0
            );
        }

        let step_time = Instant::now();

        let batch = dataloader
            .next_batch()
            .instrument(trace_span!("Wait for next data batch"))
            .await;

        // Lift splats onto the autodiff graph for this step, run training,
        // then strip back to inner so the viewer slot sees plain splats.
        // `step` immediately replaces `splats` with the returned value, so we
        // can move it here instead of cloning every iteration.
        let diff_splats = brush_render_bwd::burn_glue::lift_splats_to_autodiff(splats);
        let (new_diff_splats, stats) = trainer.step(batch, diff_splats).await;
        splats = new_diff_splats.valid();

        // Phase-local iteration for refine gating
        let phase_iter = if current_lod == 0 {
            iter
        } else {
            (iter - training_steps) % lod_refine_steps
        };
        let phase_total = if current_lod == 0 {
            training_steps
        } else {
            lod_refine_steps
        };
        let phase_progress = (phase_iter as f32 / phase_total as f32).clamp(0.0, 1.0);

        let refine_start = Instant::now();
        let refine = if phase_iter > 0
            && phase_iter.is_multiple_of(train_stream_config.train_config.refine_every)
            && phase_progress <= 0.95
        {
            let (new_splats, refine_stats) = trainer.refine(iter, splats).await;
            splats = new_splats;
            refine_stats
        } else {
            RefineStats {
                num_added: 0,
                num_split_oversized: 0,
                num_split_high_grad: 0,
                num_pruned: 0,
                num_pruned_non_finite: 0,
                total_splats: splats.num_splats(),
            }
        };
        slot.set(0, splats.clone());
        let refine_dur = refine_start.elapsed();

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == train_stream_config.train_config.total_iters();

        let step_dur = step_time.elapsed();
        train_duration += step_dur;

        // Do evals. We skip this for LODs as it'd be confusing for rerun, but, could
        // revisit this.
        if current_lod == 0
            && (iter % process_config.eval_every == 0 || iter == training_steps)
            && let Some(eval_scene) = eval_scene.as_mut()
        {
            let save_path = train_stream_config
                .process_config
                .eval_save_to_disk
                .then(|| export_path.clone());

            let eval = run_eval(
                &device,
                emitter,
                &visualize,
                splats.clone(),
                iter,
                eval_scene,
                save_path,
                train_stream_config.rerun_config.rerun_max_img_size,
            )
            .await
            .with_context(|| format!("Failed evaluation at iteration {iter}"));

            if let Err(error) = eval {
                emitter.emit(ProcessMessage::Warning { error }).await;
            }
        }

        // Export checkpoints
        #[cfg(not(target_family = "wasm"))]
        {
            let should_export = if current_lod == 0 {
                iter % process_config.export_every == 0 || (is_last_step && lod_levels == 0)
            } else {
                is_last_step
            };
            if should_export {
                let (name, exp_iter, exp_total) = if current_lod == 0 {
                    (process_config.export_name.clone(), iter, training_steps)
                } else {
                    let lod_name = process_config
                        .export_name
                        .replace(".ply", &format!("_lod{current_lod}.ply"));
                    (lod_name, lod_refine_steps, lod_refine_steps)
                };
                let res =
                    export_checkpoint(splats.clone(), &export_path, &name, exp_iter, exp_total)
                        .await
                        .with_context(|| format!("Export at iteration {iter} failed"));

                if let Err(error) = res {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }
        }

        // --- Rerun logging ---
        {
            let rerun_config = &train_stream_config.rerun_config;
            visualize
                .log_splat_stats(iter, refine.total_splats)
                .unwrap();

            if let Some(every) = rerun_config.rerun_log_splats_every
                && (iter.is_multiple_of(every) || is_last_step)
            {
                visualize.log_splats(iter, splats.clone()).await.unwrap();
            }

            if iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step {
                visualize
                    .log_train_stats(iter, &stats, step_dur)
                    .await
                    .unwrap();
            }

            // The memory query goes through the compute server and stalls
            // behind all queued GPU work — keep it off the hot path unless
            // rerun is actually recording, and then only on the stats cadence.
            if rerun_config.rerun_enabled
                && (iter.is_multiple_of(rerun_config.rerun_log_train_stats_every) || is_last_step)
            {
                visualize.log_memory(
                    iter,
                    &WgpuRuntime::<AutoCompiler>::client(wgpu_device).memory_usage()?,
                )?;
            }

            if refine.num_added > 0 {
                visualize
                    .log_refine_stats(iter, &refine, refine_dur)
                    .unwrap();
            }

            // Distribution stats need a GPU read-back, so sample them on a
            // coarser cadence than the per-refine stats.
            if iter.is_multiple_of(rerun_config.rerun_log_distribution_every) || is_last_step {
                visualize
                    .log_splat_distribution_stats(iter, splats.clone())
                    .await
                    .unwrap();
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

            let lod_progress = if current_lod > 0 {
                Some((current_lod, lod_levels))
            } else {
                None
            };

            emitter
                .emit(ProcessMessage::TrainMessage(TrainMessage::TrainStep {
                    iter,
                    total_elapsed: train_duration,
                    lod_progress,
                }))
                .await;
        }

        brush_async::yield_now().await;
    }

    emitter
        .emit(ProcessMessage::TrainMessage(TrainMessage::DoneTraining))
        .await;

    Ok(())
}

async fn run_eval(
    device: &burn::tensor::Device,
    emitter: &Emitter,
    visualize: &VisualizeTools,
    splats: Splats,
    iter: u32,
    eval_scene: &Scene,
    save_path: Option<PathBuf>,
    rerun_max_img_size: u32,
) -> Result<(), anyhow::Error> {
    if eval_scene.views.is_empty() {
        return Ok(());
    }

    let mut psnr = 0.0;
    let mut ssim = 0.0;
    let mut count = 0;
    log::info!("Running evaluation for iteration {iter}");

    for (i, view) in eval_scene.views.iter().enumerate() {
        brush_async::yield_now().await;

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
        psnr += sample.psnr.clone().into_scalar_async::<f32>().await?;
        ssim += sample.ssim.clone().into_scalar_async::<f32>().await?;

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

        visualize
            .log_eval_sample(iter, i as u32, sample, rerun_max_img_size)
            .await?;
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
    splats: Splats,
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
