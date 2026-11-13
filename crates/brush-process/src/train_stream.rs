use crate::{
    Emitter,
    config::TrainStreamConfig,
    message::{ProcessMessage, TrainMessage},
    slot::SlotSender,
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
use burn::module::Module;
use rand::SeedableRng;
use std::{collections::VecDeque, path::PathBuf, sync::Arc};

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
    device: &crate::ProcessDevice,
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

    device.seed(process_config.seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(process_config.seed);

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

    // A full checkpoint (from --checkpoint-every) trumps any other
    // initialization: it restores the splats *and* the optimizer state, so
    // training continues as if it was never interrupted.
    let resume_ckpt = load_resume_checkpoint(&vfs, &train_stream_config, device).await?;

    // Convert SplatData to Splats using KNN initialization
    let (up_axis, init_splats) = if let Some(ckpt) = &resume_ckpt {
        if load_result.init_splat.is_some() {
            log::info!("Both a checkpoint and an init ply are present; resuming from the checkpoint.");
        }
        // A resumed model keeps its own SH degree and render mode.
        (None, ckpt.splats.clone())
    } else if let Some(msg) = load_result.init_splat {
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
        let splats = to_init_splats(data, render_mode, device);
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
            device,
        );
        (None, splats)
    };

    // Fresh inits grow to the configured SH degree; a resumed model already
    // carries its trained coefficients and stays as-is.
    let init_splats = if resume_ckpt.is_some() {
        init_splats
    } else {
        init_splats.with_sh_degree(train_stream_config.model_config.sh_degree)
    };

    // If the metadata has an up axis prefer that, otherwise estimate the up direction.
    let up_axis = up_axis.or(Some(estimated_up));

    // The trainer owns its working `splats` locally and publishes a
    // clone to the `Slot` after every modification (train
    // step, refine, LOD decimation).
    let mut splats: Splats = init_splats.clone();
    let bounds = get_splat_bounds(init_splats.clone(), BOUND_PERCENTILE).await;
    slot.set(0, splats.clone());
    emitter
        .emit(ProcessMessage::SplatsUpdated {
            up_axis,
            frame: 0,
            total_frames: 1,
            num_splats: init_splats.num_splats(),
            sh_degree: init_splats.sh_degree(),
            scene_scale: bounds.median_size(),
        })
        .await;

    emitter.emit(ProcessMessage::DoneLoading).await;

    // Start with memory cleared out.
    device.memory_cleanup();

    let mut eval_scene = dataset.eval;

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(
        &dataset.train,
        process_config.seed,
        &train_stream_config.load_config,
    );
    // Per-train-view (world center, focal-px at native res) for the
    // Mip-Splatting 3D filter (always on).
    let mut view_cams: Vec<(glam::Vec3, f32)> = Vec::with_capacity(dataset.train.views.len());
    for view in dataset.train.views.iter() {
        let (w, h) = view.image.dimensions().await.unwrap_or((1, 1));
        let focal = view.camera.focal(glam::uvec2(w, h)).x;
        view_cams.push((view.camera.position, focal));
    }

    let (mut trainer, start_iter) = if let Some(ckpt) = resume_ckpt {
        // The restored trainer carries the optimizer momentum, LR-schedule
        // position and refine bounds from the checkpoint.
        if process_config.start_iter != 0 && process_config.start_iter != ckpt.next_iter {
            log::info!(
                "--start-iter {} is ignored; resuming at iteration {} from the checkpoint.",
                process_config.start_iter,
                ckpt.next_iter
            );
        }
        let current_args = crate::args_file::config_to_args(&train_stream_config).join(" ");
        if let Some(saved_args) = &ckpt.config_args
            && saved_args != &current_args
        {
            emitter
                .emit(ProcessMessage::Warning {
                    error: anyhow::anyhow!(
                        "Resuming with different settings than the checkpoint was trained with.\nCheckpoint: {saved_args}\nCurrent: {current_args}"
                    ),
                })
                .await;
        }
        log::info!("Resumed training from checkpoint at iteration {}.", ckpt.next_iter);
        (ckpt.trainer, ckpt.next_iter)
    } else {
        let trainer = SplatTrainer::new_seeded(
            &train_stream_config.train_config,
            device,
            bounds,
            process_config.seed,
        );
        (trainer, process_config.start_iter)
    };
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

    let mut vram_guard = VramGuard::new(
        process_config.vram_dump_above_gb,
        process_config.vram_dump_slow_factor,
    );

    log::info!("Start training loop.");
    for iter in start_iter..train_stream_config.train_config.total_iters() {
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
                let res = export_checkpoint(
                    splats.clone(),
                    &export_path,
                    &name,
                    exp_iter,
                    exp_total,
                    up_axis,
                    train_stream_config.load_config.units_per_meter,
                )
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
            let scores = compute_pup_scores(splats.clone(), &dataset.train, device).await;
            splats = decimate_to_count(splats, &scores, target_count).await;
            slot.set(0, splats.clone());

            let after = splats.num_splats();
            log::info!("LOD {current_lod}/{lod_levels}: {before} -> {after} splats");

            device.memory_cleanup();

            let cumulative_scale = (lod_img_pct as f32 / 100.0).powi(current_lod as i32);
            // Only rebuild the loader when the images actually changed size.
            // A rebuild throws away a warm batch cache and re-decodes the
            // whole dataset, which at 100% scale buys nothing.
            if lod_img_pct < 100 {
                let lod_scene = dataset.train.clone().with_image_scale(cumulative_scale);
                dataloader = SceneLoader::new(
                    &lod_scene,
                    process_config.seed,
                    &train_stream_config.load_config,
                );
            }

            let bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
            trainer = SplatTrainer::new_seeded(
                &train_stream_config.train_config,
                device,
                bounds,
                process_config.seed,
            );
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
        let diff_splats = splats.train();
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
        let did_refine = phase_iter > 0
            && phase_iter.is_multiple_of(train_stream_config.train_config.refine_every)
            && phase_progress <= 0.95;
        let refine = if did_refine {
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
                device,
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
                let res = export_checkpoint(
                    splats.clone(),
                    &export_path,
                    &name,
                    exp_iter,
                    exp_total,
                    up_axis,
                    train_stream_config.load_config.units_per_meter,
                )
                .await
                .with_context(|| format!("Export at iteration {iter} failed"));

                if let Err(error) = res {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }

            // Full training checkpoints for resuming. Only in the main phase:
            // LOD phases restart from a decimated model anyway.
            let should_checkpoint = process_config.checkpoint_every > 0
                && current_lod == 0
                && (iter.is_multiple_of(process_config.checkpoint_every) || is_last_step);
            if should_checkpoint {
                let res =
                    write_checkpoint(&trainer, &splats, iter, &export_path, &train_stream_config)
                        .await
                        .with_context(|| format!("Checkpoint at iteration {iter} failed"));

                if let Err(error) = res {
                    emitter.emit(ProcessMessage::Warning { error }).await;
                }
            }
        }

        // Auto VRAM dump-and-reload: when the memory pool outgrows its budget
        // or steps get sustainably slower than their own baseline (VRAM
        // spilling into system memory), checkpoint in place, release every
        // GPU allocation, and reload. Main phase only, like checkpoints.
        if vram_guard.enabled() && current_lod == 0 {
            if !did_refine {
                vram_guard.record(step_dur);
            }
            if let Some(reason) = vram_guard.should_dump(iter, || {
                device.memory_pool_usage().unwrap_or_default().bytes_reserved
            }) {
                log::info!("{reason}; dumping and reloading GPU memory at iteration {iter}.");
                let dump_start = Instant::now();
                let res = dump_and_reload(
                    &mut splats,
                    &mut trainer,
                    &slot,
                    &train_stream_config,
                    device,
                    iter,
                    &view_cams,
                    &export_path,
                )
                .await
                .with_context(|| format!("VRAM dump at iteration {iter} failed"));
                match res {
                    Ok(()) => {
                        vram_guard.mark_dumped(iter);
                        log::info!(
                            "GPU memory reloaded in {:.1}s.",
                            dump_start.elapsed().as_secs_f32()
                        );
                    }
                    Err(error) => emitter.emit(ProcessMessage::Warning { error }).await,
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
                visualize.log_memory(iter, &device.memory_pool_usage().unwrap_or_default())?;
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
                    scene_scale: trainer.bounds().median_size(),
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

#[cfg(not(target_family = "wasm"))]
fn eval_output_path(base: &Path, iter: u32, image_path: &Path) -> anyhow::Result<PathBuf> {
    let mut relative = PathBuf::new();
    for component in image_path.components() {
        match component {
            std::path::Component::Normal(part) => relative.push(part),
            std::path::Component::CurDir => {}
            _ => anyhow::bail!(
                "Eval image path must be relative and cannot traverse directories: {}",
                image_path.display()
            ),
        }
    }
    if relative.as_os_str().is_empty() {
        anyhow::bail!("Eval image path is empty");
    }
    relative.set_extension("png");
    Ok(base.join(format!("eval_{iter}")).join(relative))
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[test]
    fn eval_paths_are_safe_and_distinct() {
        let base = Path::new("output");
        let first = eval_output_path(base, 42, Path::new("images/a/frame.jpg")).unwrap();
        let second = eval_output_path(base, 42, Path::new("images/b/frame.jpg")).unwrap();

        assert_eq!(first, Path::new("output/eval_42/images/a/frame.png"));
        assert_eq!(second, Path::new("output/eval_42/images/b/frame.png"));
        assert_ne!(first, second);

        for invalid in ["../frame.jpg", "/tmp/frame.jpg", "."] {
            assert!(eval_output_path(base, 1, Path::new(invalid)).is_err());
        }
    }

    /// Feed `guard` `n` plain-step samples of `ms` milliseconds.
    fn feed(guard: &mut VramGuard, n: usize, ms: u64) {
        for _ in 0..n {
            guard.record(Duration::from_millis(ms));
        }
    }

    #[test]
    fn vram_guard_disabled_by_default() {
        let mut guard = VramGuard::new(0.0, 0.0);
        assert!(!guard.enabled(), "both triggers off means disabled");
        feed(&mut guard, 200, 500);
        assert!(
            guard.should_dump(100, || u64::MAX).is_none(),
            "disabled guard never dumps"
        );
    }

    #[test]
    fn vram_guard_triggers_on_sustained_slowdown() {
        let mut guard = VramGuard::new(0.0, 2.0);
        assert!(guard.enabled(), "slow factor alone enables the guard");
        // Warmup + baseline at 100 ms.
        feed(
            &mut guard,
            VRAM_WARMUP_STEPS as usize + VRAM_BASELINE_LEN,
            100,
        );
        // A fast rolling window must not trigger.
        feed(&mut guard, VRAM_WINDOW_LEN, 150);
        assert!(
            guard.should_dump(1000, || 0).is_none(),
            "1.5x is under the 2x threshold"
        );
        // A sustained 3x slowdown must trigger.
        feed(&mut guard, VRAM_WINDOW_LEN, 300);
        let reason = guard.should_dump(1001, || 0);
        assert!(reason.is_some(), "3x slowdown over a full window triggers");

        // After a dump the baseline re-learns: the same slow pace becomes the
        // new normal instead of a dump loop.
        guard.mark_dumped(1001);
        feed(
            &mut guard,
            VRAM_WARMUP_STEPS as usize + VRAM_BASELINE_LEN + VRAM_WINDOW_LEN,
            300,
        );
        assert!(
            guard.should_dump(2000, || 0).is_none(),
            "steady state after re-baseline must not re-trigger"
        );
    }

    #[test]
    fn vram_guard_triggers_on_memory_budget() {
        let mut guard = VramGuard::new(1.0, 0.0);
        assert!(
            guard.should_dump(150, || u64::MAX).is_none(),
            "memory is only checked on the sparse cadence"
        );
        assert!(
            guard
                .should_dump(VRAM_MEM_CHECK_EVERY, || 2_000_000_000)
                .is_some(),
            "2 GB reserved over a 1 GB budget triggers"
        );
        assert!(
            guard
                .should_dump(VRAM_MEM_CHECK_EVERY, || 500_000_000)
                .is_none(),
            "0.5 GB reserved under a 1 GB budget does not trigger"
        );
    }

    #[test]
    fn vram_guard_respects_cooldown() {
        let mut guard = VramGuard::new(1.0, 0.0);
        guard.mark_dumped(1000);
        assert!(
            guard
                .should_dump(1000 + VRAM_MEM_CHECK_EVERY, || u64::MAX)
                .is_none(),
            "within cooldown no dump may trigger"
        );
        let after = 1000 + VRAM_DUMP_COOLDOWN.next_multiple_of(VRAM_MEM_CHECK_EVERY);
        assert!(
            guard.should_dump(after, || u64::MAX).is_some(),
            "after cooldown the memory trigger works again"
        );
    }
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
            let path = eval_output_path(path, iter, view.image.path())?;
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.with_context(|| {
                    format!(
                        "Failed to create eval output directory {}",
                        parent.display()
                    )
                })?;
            }
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

/// Look for a resume checkpoint: an explicit `--resume` path, or a
/// `checkpoint.safetensors` inside the dataset. Returns the restored state,
/// or `None` to start fresh. A checkpoint that exists but fails to load is an
/// error rather than a silent restart from scratch.
async fn load_resume_checkpoint(
    vfs: &BrushVfs,
    config: &TrainStreamConfig,
    device: &crate::ProcessDevice,
) -> anyhow::Result<Option<brush_train::checkpoint::LoadedCheckpoint>> {
    use brush_train::checkpoint::{CHECKPOINT_FILE_NAME, checkpoint_from_bytes};

    let bytes: Option<Vec<u8>> = if let Some(path) = &config.process_config.resume {
        #[cfg(target_family = "wasm")]
        {
            anyhow::bail!(
                "--resume with a file path isn't supported on the web; add {CHECKPOINT_FILE_NAME} to the dataset files instead (ignored path: {path})."
            );
        }
        #[cfg(not(target_family = "wasm"))]
        {
            Some(
                tokio::fs::read(path)
                    .await
                    .with_context(|| format!("Reading checkpoint {path}"))?,
            )
        }
    } else if let Some(ckpt_path) = vfs
        .file_paths()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(CHECKPOINT_FILE_NAME))
    {
        use tokio::io::AsyncReadExt;
        let mut reader = vfs.reader_at_path(&ckpt_path).await?;
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .with_context(|| format!("Reading checkpoint {}", ckpt_path.display()))?;
        Some(bytes)
    } else {
        None
    };

    let Some(bytes) = bytes else {
        return Ok(None);
    };

    let ckpt = checkpoint_from_bytes(
        &bytes,
        &config.train_config,
        device,
        config.process_config.seed,
    )?;
    anyhow::ensure!(
        ckpt.next_iter <= config.train_config.total_train_iters,
        "Checkpoint is at iteration {} but --total-train-iters is {}. Raise --total-train-iters to continue training this model.",
        ckpt.next_iter,
        config.train_config.total_train_iters,
    );
    Ok(Some(ckpt))
}

/// Write a full training checkpoint atomically (via a temp file + rename), so
/// a crash mid-write never destroys the previous checkpoint.
#[cfg(not(target_family = "wasm"))]
async fn write_checkpoint(
    trainer: &SplatTrainer,
    splats: &Splats,
    next_iter: u32,
    export_path: &Path,
    config: &TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    use brush_train::checkpoint::{META_CONFIG_ARGS, checkpoint_to_bytes};

    let args = crate::args_file::config_to_args(config).join(" ");
    let bytes = checkpoint_to_bytes(
        trainer,
        splats,
        next_iter,
        &[(META_CONFIG_ARGS.to_owned(), args)],
    )
    .await?;
    write_checkpoint_bytes(&bytes, export_path).await?;
    log::info!("Wrote checkpoint at iteration {next_iter}.");
    Ok(())
}

/// Atomically place already-serialized checkpoint bytes as
/// `checkpoint.safetensors` in the export directory.
#[cfg(not(target_family = "wasm"))]
async fn write_checkpoint_bytes(bytes: &[u8], export_path: &Path) -> Result<(), anyhow::Error> {
    use brush_train::checkpoint::CHECKPOINT_FILE_NAME;

    tokio::fs::create_dir_all(&export_path)
        .await
        .with_context(|| format!("Creating export directory {}", export_path.display()))?;
    let final_path = export_path.join(CHECKPOINT_FILE_NAME);
    let tmp_path = export_path.join(format!("{CHECKPOINT_FILE_NAME}.tmp"));
    tokio::fs::write(&tmp_path, bytes)
        .await
        .with_context(|| format!("Writing checkpoint {}", tmp_path.display()))?;
    tokio::fs::rename(&tmp_path, &final_path)
        .await
        .with_context(|| format!("Moving checkpoint into place at {}", final_path.display()))?;
    Ok(())
}

/// Checkpoint the full training state to host memory (and disk, where there
/// is one), release every GPU allocation the trainer and the published splats
/// hold, and reload through the same path a disk resume takes. Training
/// continues as if uninterrupted; the burn memory pool gets to start over
/// compact, which counters fragmentation and VRAM overcommit.
async fn dump_and_reload(
    splats: &mut Splats,
    trainer: &mut SplatTrainer,
    slot: &SlotSender<Splats>,
    train_stream_config: &TrainStreamConfig,
    device: &crate::ProcessDevice,
    next_iter: u32,
    view_cams: &[(glam::Vec3, f32)],
    export_path: &Path,
) -> Result<(), anyhow::Error> {
    use brush_train::checkpoint::{META_CONFIG_ARGS, checkpoint_from_bytes, checkpoint_to_bytes};

    let args = crate::args_file::config_to_args(train_stream_config).join(" ");
    let bytes = checkpoint_to_bytes(
        trainer,
        splats,
        next_iter,
        &[(META_CONFIG_ARGS.to_owned(), args)],
    )
    .await?;

    // Crash-safety copy on disk while the bytes are in hand; best-effort.
    #[cfg(not(target_family = "wasm"))]
    if let Err(e) = write_checkpoint_bytes(&bytes, export_path).await {
        log::warn!("Could not write checkpoint during VRAM dump: {e:?}");
    }
    #[cfg(target_family = "wasm")]
    let _ = export_path;

    // Swap the published splats for a tiny placeholder so the viewer slot
    // doesn't pin the old buffers, drop the optimizer momentum, and hand the
    // pool's pages back.
    let placeholder = Splats::from_raw(
        vec![0.0; 3],
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0; 3],
        vec![0.0; 3],
        vec![0.0],
        SplatRenderMode::Default,
        device,
    );
    *splats = placeholder.clone();
    slot.set(0, placeholder);
    trainer.release_gpu_state();
    device.memory_cleanup();

    let ckpt = checkpoint_from_bytes(
        &bytes,
        &train_stream_config.train_config,
        device,
        train_stream_config.process_config.seed,
    )?;
    *splats = ckpt.splats;
    *trainer = ckpt.trainer;
    trainer.set_view_cams(view_cams.to_vec());
    slot.set(0, splats.clone());
    device.memory_cleanup();
    Ok(())
}

/// Decides when to dump-and-reload GPU memory. Two independent triggers:
/// the burn memory pool reserving more than a configured budget, and training
/// steps getting sustainably slower than their own baseline (VRAM overcommit
/// spilling into system memory shows up exactly this way). Timing samples
/// only count plain steps — refine iterations are excluded by the caller, and
/// a warmup is skipped after every (re)start so shader compilation and cache
/// warming don't skew the baseline.
struct VramGuard {
    budget_bytes: u64,
    slow_factor: f64,
    /// Plain-step samples still to ignore before collecting.
    warmup_left: u32,
    /// First `BASELINE_LEN` post-warmup samples; the reference speed.
    baseline: Vec<f64>,
    /// Rolling window of the most recent samples.
    window: VecDeque<f64>,
    last_dump_iter: Option<u32>,
}

/// Samples forming the reference speed after each (re)start.
const VRAM_BASELINE_LEN: usize = 50;
/// Rolling window compared against the baseline.
const VRAM_WINDOW_LEN: usize = 25;
/// Plain steps ignored after every (re)start.
const VRAM_WARMUP_STEPS: u32 = 20;
/// The pool-size query stalls behind queued GPU work, so it runs sparsely.
const VRAM_MEM_CHECK_EVERY: u32 = 100;
/// Minimum iterations between two dumps.
const VRAM_DUMP_COOLDOWN: u32 = 500;

fn median(samples: impl IntoIterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = samples.into_iter().collect();
    v.sort_by(f64::total_cmp);
    if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

impl VramGuard {
    fn new(budget_gb: f32, slow_factor: f32) -> Self {
        Self {
            budget_bytes: (f64::from(budget_gb.max(0.0)) * 1e9) as u64,
            slow_factor: f64::from(slow_factor.max(0.0)),
            warmup_left: VRAM_WARMUP_STEPS,
            baseline: Vec::new(),
            window: VecDeque::new(),
            last_dump_iter: None,
        }
    }

    fn enabled(&self) -> bool {
        self.budget_bytes > 0 || self.slow_factor > 0.0
    }

    /// Record one plain (non-refine) step's duration.
    fn record(&mut self, duration: Duration) {
        if self.warmup_left > 0 {
            self.warmup_left -= 1;
            return;
        }
        let secs = duration.as_secs_f64();
        if self.baseline.len() < VRAM_BASELINE_LEN {
            self.baseline.push(secs);
            return;
        }
        self.window.push_back(secs);
        if self.window.len() > VRAM_WINDOW_LEN {
            self.window.pop_front();
        }
    }

    /// A dump just ran: start a cooldown and re-learn the timing baseline, so
    /// an unfixable slowdown becomes the new normal instead of a dump loop.
    fn mark_dumped(&mut self, iter: u32) {
        self.last_dump_iter = Some(iter);
        self.warmup_left = VRAM_WARMUP_STEPS;
        self.baseline.clear();
        self.window.clear();
    }

    /// Returns a human-readable reason when a dump should happen now.
    /// `pool_reserved` is only queried on the sparse check cadence.
    fn should_dump(&mut self, iter: u32, pool_reserved: impl FnOnce() -> u64) -> Option<String> {
        if let Some(last) = self.last_dump_iter
            && iter < last.saturating_add(VRAM_DUMP_COOLDOWN)
        {
            return None;
        }
        if self.slow_factor > 0.0
            && self.baseline.len() == VRAM_BASELINE_LEN
            && self.window.len() == VRAM_WINDOW_LEN
        {
            let base = median(self.baseline.iter().copied());
            let rolling = median(self.window.iter().copied());
            if base > 0.0 && rolling > base * self.slow_factor {
                return Some(format!(
                    "Training steps slowed to {:.0} ms (baseline {:.0} ms, over {:.1}x)",
                    rolling * 1000.0,
                    base * 1000.0,
                    self.slow_factor
                ));
            }
        }
        if self.budget_bytes > 0 && iter.is_multiple_of(VRAM_MEM_CHECK_EVERY) {
            let reserved = pool_reserved();
            if reserved > self.budget_bytes {
                return Some(format!(
                    "GPU memory pool at {:.2} GB exceeds the {:.2} GB budget",
                    reserved as f64 / 1e9,
                    self.budget_bytes as f64 / 1e9
                ));
            }
        }
        None
    }
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
    up_axis: Option<glam::Vec3>,
    units_per_meter: f32,
) -> Result<(), anyhow::Error> {
    tokio::fs::create_dir_all(&export_path)
        .await
        .with_context(|| format!("Creating export directory {}", export_path.display()))?;
    let digits = ((total_steps as f64).log10().floor() as usize) + 1;
    let export_name = export_name.replace("{iter}", &format!("{iter:0digits$}"));
    // Training runs in metres; write the file back in the dataset's units.
    let splat_data = brush_serde::splat_to_ply(splats.scaled(units_per_meter), up_axis)
        .await
        .context("Serializing splat data")?;
    tokio::fs::write(export_path.join(&export_name), splat_data)
        .await
        .context(format!("Failed to export ply {export_path:?}"))?;
    Ok(())
}
