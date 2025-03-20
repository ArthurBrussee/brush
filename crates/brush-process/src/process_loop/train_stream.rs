use std::path::Path;

use anyhow::Context;
/// A default training loop for Brush.
use async_fn_stream::TryStreamEmitter;

use brush_dataset::brush_vfs::BrushVfs;
use brush_dataset::{Dataset, scene_loader::SceneLoader};
use brush_render::gaussian_splats::{RandomSplatsConfig, Splats};
use brush_train::train::SplatTrainer;
use brush_train::train::TrainBack;

use burn::module::AutodiffModule;
use burn::prelude::Backend;
use burn_cubecl::cubecl::Runtime;
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use rand::SeedableRng;
use tokio_stream::StreamExt;
use web_time::{Duration, Instant};

use crate::rerun_tools::VisualizeTools;

use super::{ProcessArgs, ProcessMessage};

pub(crate) async fn train_stream(
    vfs: BrushVfs,
    process_args: ProcessArgs,
    device: WgpuDevice,
    emitter: TryStreamEmitter<ProcessMessage, anyhow::Error>,
) -> anyhow::Result<()> {
    let process_config = &process_args.process_config;

    emitter
        .emit(ProcessMessage::StartLoading { training: true })
        .await;

    <TrainBack as Backend>::seed(process_config.seed);
    let mut rng = rand::rngs::StdRng::from_seed([process_config.seed as u8; 32]);

    // Load initial splats if included
    let mut initial_splats = None;

    let mut dataset = Dataset::empty();
    let (mut splat_stream, mut data_stream) =
        brush_dataset::load_dataset(vfs.clone(), &process_args.load_config, &device).await?;

    let visualize = VisualizeTools::new(process_args.rerun_config.rerun_enabled);

    // Read dataset stream.
    while let Some(d) = data_stream.next().await {
        dataset = d.context("Failed to parse dataset. \n")?;
        emitter
            .emit(ProcessMessage::Dataset {
                data: dataset.clone(),
            })
            .await;
    }

    visualize.log_scene(&dataset.train, process_args.rerun_config.rerun_max_img_size)?;

    let estimated_up = dataset.estimate_up();

    // Read initial splats if any.
    while let Some(message) = splat_stream.next().await {
        let message = message?;
        let msg = ProcessMessage::ViewSplats {
            // If the metadata has an up axis prefer that, otherwise estimate
            // the up direction.
            up_axis: message.meta.up_axis.or(Some(estimated_up)),
            splats: Box::new(message.splats.valid()),
            frame: 0,
            total_frames: 0,
        };
        emitter.emit(msg).await;
        initial_splats = Some(message.splats);
    }

    emitter
        .emit(ProcessMessage::DoneLoading { training: true })
        .await;

    let splats = if let Some(splats) = initial_splats {
        splats
    } else {
        // By default, spawn the splats in bounds.
        let bounds = dataset.train.bounds();
        let bounds_extent = bounds.extent.length();
        // Arbitrarily assume area of interest is 0.2 - 0.75 of scene bounds.
        // Somewhat specific to the blender scenes
        let adjusted_bounds = dataset
            .train
            .adjusted_bounds(bounds_extent * 0.25, bounds_extent);
        let config = RandomSplatsConfig::new();
        Splats::from_random_config(&config, adjusted_bounds, &mut rng, &device)
    };

    let mut splats = splats.with_sh_degree(process_args.model_config.sh_degree);

    let eval_scene = dataset.eval.clone();
    let train_scene = dataset.train.clone();
    let scene_extent = train_scene.estimate_extent().unwrap_or(1.0);

    let mut train_duration = Duration::from_secs(0);
    let mut dataloader = SceneLoader::new(&train_scene, 42, &device);
    let mut trainer = SplatTrainer::new(&process_args.train_config, &device);

    for iter in process_args.process_config.start_iter..process_args.train_config.total_steps {
        let step_time = Instant::now();

        let batch = dataloader.next_batch().await;
        let (new_splats, stats) = trainer.step(scene_extent, iter, batch, splats);
        splats = new_splats;
        let (new_splats, refine) = trainer.refine_if_needed(iter, splats).await;
        splats = new_splats;

        if let Some(stats) = refine {
            visualize.log_refine_stats(iter, &stats)?;
            emitter
                .emit(ProcessMessage::RefineStep {
                    stats: Box::new(stats),
                    cur_splat_count: splats.num_splats(),
                    iter,
                })
                .await;
        }

        #[allow(unused)]
        let export_path =
            Path::new(process_config.export_path.as_deref().unwrap_or(".")).to_owned();

        // We just finished iter 'iter', now starting iter + 1.
        let iter = iter + 1;
        let is_last_step = iter == process_args.train_config.total_steps;

        // Check if we want to evaluate _next iteration_. Small detail, but this ensures we evaluate
        // before doing a refine.
        if iter % process_config.eval_every == 0 || is_last_step {
            if let Some(eval_scene) = eval_scene.as_ref() {
                let mut psnr = 0.0;
                let mut ssim = 0.0;
                let mut count = 0;

                log::info!("Running evaluation for iteration {iter}");

                for sample in brush_train::eval::eval_stats(
                    splats.valid(),
                    eval_scene,
                    None,
                    &mut rng,
                    &device,
                ) {
                    count += 1;
                    psnr += sample.psnr.clone().into_scalar_async().await;
                    ssim += sample.ssim.clone().into_scalar_async().await;
                    visualize.log_eval_sample(iter, &sample).await?;

                    #[cfg(not(target_family = "wasm"))]
                    if process_args.process_config.eval_save_to_disk {
                        log::info!("Saving eval image to disk.");

                        let eval_render = brush_train::image::tensor_into_image(
                            sample.rendered.clone().into_data_async().await,
                        );
                        let rendered: image::DynamicImage = eval_render.to_rgb8().into();

                        let img_name = Path::new(&sample.view.path)
                            .file_stem()
                            .expect("No file name for eval view.")
                            .to_string_lossy();

                        let path = Path::new(&export_path)
                            .join(format!("eval_{iter}"))
                            .join(format!("{img_name}.png"));

                        let parent = path.parent().expect("Eval must have a filename");
                        tokio::fs::create_dir_all(parent).await?;

                        log::info!("Saving eval view to {path:?}");

                        rendered.save(path)?;
                    }
                }

                psnr /= count as f32;
                ssim /= count as f32;

                visualize.log_eval_stats(iter, psnr, ssim)?;

                let message = ProcessMessage::EvalResult {
                    iter,
                    avg_psnr: psnr,
                    avg_ssim: ssim,
                };

                emitter.emit(message).await;
            }
        }

        let client = WgpuRuntime::client(&device);
        visualize.log_memory(iter, &client.memory_usage())?;

        // TODO: Support this on WASM somehow. Maybe have user pick a file once,
        // and write to it repeatedly?
        #[cfg(not(target_family = "wasm"))]
        if iter % process_config.export_every == 0 || is_last_step {
            let total_steps = process_args.train_config.total_steps;

            // Ad-hoc format string.
            let digits = (total_steps as f64).log10().ceil() as usize;
            let export_name = process_config
                .export_name
                .replace("{iter}", &format!("{iter:0digits$}"));

            tokio::fs::create_dir_all(&export_path).await?;

            let splat_data = brush_dataset::splat_export::splat_to_ply(splats.valid()).await?;
            tokio::fs::write(export_path.join(&export_name), splat_data)
                .await
                .with_context(|| format!("Failed to export ply {export_path:?}"))?;
        }

        if let Some(every) = process_args.rerun_config.rerun_log_splats_every {
            if iter % every == 0 || is_last_step {
                visualize.log_splats(iter, splats.valid()).await?;
            }
        }

        visualize.log_splat_stats(iter, &splats)?;

        // How frequently to update the UI after a training step.
        const UPDATE_EVERY: u32 = 5;

        if iter % UPDATE_EVERY == 0 || is_last_step {
            let message = ProcessMessage::TrainStep {
                splats: Box::new(splats.valid()),
                stats: Box::new(stats.clone()),
                iter,
                total_elapsed: train_duration,
            };
            emitter.emit(message).await;
        }

        // Log out train stats.
        if iter % process_args.rerun_config.rerun_log_train_stats_every == 0 || is_last_step {
            visualize.log_train_stats(iter, stats).await?;
        }

        if is_last_step {
            break;
        }

        // Add up time from this step.
        train_duration += step_time.elapsed();
    }

    Ok(())
}
