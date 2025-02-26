use std::path::Path;

use anyhow::Context;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn_cubecl::cubecl::Runtime;
use web_time::Instant;

use crate::{data_source::DataSource, rerun_tools::VisualizeTools};
use brush_dataset::{Dataset, brush_vfs::BrushVfs, splat_import};
use brush_render::gaussian_splats::{RandomSplatsConfig, Splats};
use brush_train::train::{RefineStats, TrainBack, TrainStepStats};
use burn::{backend::Autodiff, module::AutodiffModule};
use burn_wgpu::{Wgpu, WgpuDevice, WgpuRuntime};
use glam::Vec3;
use rand::SeedableRng;
use tokio::sync::mpsc::{Receiver, unbounded_channel};
use tokio::sync::mpsc::{Sender, UnboundedReceiver};
use tokio::sync::mpsc::{UnboundedSender, channel};
use tokio_stream::StreamExt;

#[allow(unused)]
use brush_dataset::splat_export;

use super::{
    ProcessArgs,
    train_stream::{self, train_stream},
};

pub enum ProcessMessage {
    NewSource,
    StartLoading {
        training: bool,
    },
    /// Some process errored out, and want to display this error
    /// to the user.
    Error(anyhow::Error),
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
        timestamp: Instant,
    },
    /// Some number of training steps are done.
    #[allow(unused)]
    RefineStep {
        stats: Box<RefineStats>,
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

#[derive(Debug, Clone)]
pub enum ControlMessage {
    Paused(bool),
}

async fn process_loop(
    source: DataSource,
    output: Sender<ProcessMessage>,
    args: ProcessArgs,
    device: WgpuDevice,
    control_receiver: UnboundedReceiver<ControlMessage>,
) {
    if output.send(ProcessMessage::NewSource).await.is_err() {
        return;
    }

    let vfs = source.into_vfs().await;

    let vfs = match vfs {
        Ok(vfs) => vfs,
        Err(e) => {
            let _ = output.send(ProcessMessage::Error(e)).await;
            return;
        }
    };

    let paths: Vec<_> = vfs.file_names().collect();
    log::info!("Mounted VFS with {} files", paths.len());

    let result = if paths
        .iter()
        .all(|p| p.extension().is_some_and(|p| p == "ply"))
    {
        view_process_loop(paths, output.clone(), vfs, device).await
    } else {
        train_process_loop(output.clone(), vfs, device, control_receiver, &args).await
    };

    if let Err(e) = result {
        let _ = output.send(ProcessMessage::Error(e)).await;
    }
}

async fn view_process_loop(
    paths: Vec<std::path::PathBuf>,
    output: Sender<ProcessMessage>,
    vfs: BrushVfs,
    device: WgpuDevice,
) -> Result<(), anyhow::Error> {
    let mut vfs = vfs;

    for (i, path) in paths.iter().enumerate() {
        log::info!("Loading single ply file");

        if output
            .send(ProcessMessage::StartLoading { training: false })
            .await
            .is_err()
        {
            return Ok(());
        }

        let sub_sample = None; // Subsampling a trained ply doesn't really make sense.
        let splat_stream = splat_import::load_splat_from_ply(
            vfs.open_path(path).await?,
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

            if output
                .send(ProcessMessage::ViewSplats {
                    up_axis: message.meta.up_axis,
                    splats: Box::new(message.splats),
                    frame,
                    total_frames,
                })
                .await
                .is_err()
            {
                return Ok(());
            }
        }
    }

    let _ = output
        .send(ProcessMessage::DoneLoading { training: false })
        .await;
    Ok(())
}

async fn train_process_loop(
    output: Sender<ProcessMessage>,
    vfs: BrushVfs,
    device: WgpuDevice,
    control_receiver: UnboundedReceiver<ControlMessage>,
    process_args: &ProcessArgs,
) -> Result<(), anyhow::Error> {
    let process_config = &process_args.process_config;

    let _ = output
        .send(ProcessMessage::StartLoading { training: true })
        .await;

    <Autodiff<Wgpu> as Backend>::seed(process_config.seed);
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

        let _ = output
            .send(ProcessMessage::Dataset {
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
        if output.send(msg).await.is_err() {
            return Ok(());
        }
        initial_splats = Some(message.splats);
    }

    let _ = output
        .send(ProcessMessage::DoneLoading { training: true })
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

    let splats = splats.with_sh_degree(process_args.model_config.sh_degree);

    let mut control_receiver = control_receiver;

    let eval_scene = dataset.eval.clone();
    let stream = train_stream(
        dataset,
        splats,
        process_args.train_config.clone(),
        device.clone(),
        process_args.process_config.start_iter,
    );
    let mut stream = std::pin::pin!(stream);

    let mut train_paused = false;

    loop {
        let control = if train_paused {
            control_receiver.recv().await
        } else {
            control_receiver.try_recv().ok()
        };

        if let Some(control) = control {
            match control {
                ControlMessage::Paused(paused) => {
                    train_paused = paused;
                }
            }
        }

        let msg = stream.next().await;

        let Some(msg) = msg else {
            break;
        };

        // Bubble up errors in message.
        let msg = msg?;

        match msg {
            train_stream::TrainMessage::TrainStep {
                splats,
                stats,
                iter,
                timestamp,
            } => {
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
                            *splats.clone(),
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

                        if output
                            .send(ProcessMessage::EvalResult {
                                iter,
                                avg_psnr: psnr,
                                avg_ssim: ssim,
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                }

                let client = WgpuRuntime::client(&device);
                visualize.log_memory(iter, &client.memory_usage())?;

                // TODO: Support this on WASM somehow. Maybe have user pick a file once,
                // and write to it repeatedly?
                #[cfg(not(target_family = "wasm"))]
                if iter % process_config.export_every == 0 || is_last_step {
                    let splats = *splats.clone();
                    let output_send = output.clone();

                    let total_steps = process_args.train_config.total_steps;

                    // Ad-hoc format string.
                    let digits = (total_steps as f64).log10().ceil() as usize;
                    let export_name = process_config
                        .export_name
                        .replace("{iter}", &format!("{iter:0digits$}"));

                    tokio::fs::create_dir_all(&export_path).await?;

                    // Nb: this COULD easily be done in the spawned future as well,
                    // but for memory reasons it's not great to keep another copy of the
                    // field.
                    let splat_data = splat_export::splat_to_ply(splats).await?;

                    tokio::task::spawn(async move {
                        if let Err(e) = tokio::fs::write(export_path.join(&export_name), splat_data)
                            .await
                            .with_context(|| format!("Failed to export ply {export_path:?}"))
                        {
                            let _ = output_send.send(ProcessMessage::Error(e)).await;
                        }
                    });
                }

                if let Some(every) = process_args.rerun_config.rerun_log_splats_every {
                    if iter % every == 0 || is_last_step {
                        visualize.log_splats(iter, *splats.clone()).await?;
                    }
                }

                visualize.log_splat_stats(iter, &splats)?;

                // Log out train stats.
                if iter % process_args.rerun_config.rerun_log_train_stats_every == 0 || is_last_step
                {
                    visualize.log_train_stats(iter, *stats.clone()).await?;
                }

                // How frequently to update the UI after a training step.
                const UPDATE_EVERY: u32 = 5;

                if (iter % UPDATE_EVERY == 0 || is_last_step)
                    && output
                        .send(ProcessMessage::TrainStep {
                            splats,
                            stats,
                            iter,
                            timestamp,
                        })
                        .await
                        .is_err()
                {
                    break;
                }

                if is_last_step {
                    break;
                }
            }
            train_stream::TrainMessage::RefineStep { stats, iter } => {
                visualize.log_refine_stats(iter, &stats)?;

                if output
                    .send(ProcessMessage::RefineStep { stats, iter })
                    .await
                    .is_err()
                {
                    break;
                }
            }
        }
    }

    Ok(())
}

pub struct RunningProcess {
    pub start_args: ProcessArgs,
    pub messages: Receiver<ProcessMessage>,
    pub control: UnboundedSender<ControlMessage>,
}

pub fn start_process(source: DataSource, args: ProcessArgs, device: WgpuDevice) -> RunningProcess {
    log::info!("Starting process with source {:?}", source);

    // Create a small channel. We don't want 10 updated splats to be stuck in the queue eating up memory!
    // Bigger channels could mean the train loop spends less time waiting for the UI though.
    // create a channel for the train loop.
    let (sender, receiver) = channel(1);
    let (train_sender, train_receiver) = unbounded_channel();

    let args_loop = args.clone();
    tokio_with_wasm::alias::task::spawn(async move {
        process_loop(source, sender, args_loop, device, train_receiver).await;
    });

    RunningProcess {
        start_args: args,
        messages: receiver,
        control: train_sender,
    }
}
