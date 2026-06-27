#![recursion_limit = "256"]
#![cfg(not(target_family = "wasm"))]

use brush_async::Actor;
use brush_process::DataSource;
use brush_process::RunningProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::create_process;
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;

use clap::{Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::trace_span;

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    next_line_help = true,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub animation: AnimationArgs,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,
}

/// Options for rendering an animation (instead of training). The positional
/// `PATH_OR_URL` is reused as the splat source: a single `.ply`, or a folder of
/// `.ply` files (one video is rendered per file).
#[derive(clap::Args, Clone)]
#[command(next_help_heading = "Animation")]
pub struct AnimationArgs {
    /// Render an animation instead of training: path to an animation config
    /// (JSON) saved from the viewer's "Save…" button.
    #[arg(long, value_name = "ANIM_CONFIG")]
    pub render_anim: Option<PathBuf>,

    /// Output for the rendered animation. For a single ply, this is the MP4
    /// path. For a folder source, set this to a directory to collect the videos
    /// there (named after each ply); otherwise each video is written next to
    /// its ply.
    #[arg(long, value_name = "OUT", default_value = "animation.mp4")]
    pub anim_out: PathBuf,

    /// Output width for the rendered animation.
    #[arg(long, default_value_t = 1920)]
    pub anim_width: u32,

    /// Output height for the rendered animation.
    #[arg(long, default_value_t = 1080)]
    pub anim_height: u32,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if self.animation.render_anim.is_some() {
            if self.source.is_none() {
                return Err(Error::raw(
                    ErrorKind::MissingRequiredArgument,
                    "--render-anim requires a PATH_OR_URL source (a .ply file or a folder of plys)",
                ));
            }
            return Ok(self);
        }
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

/// Build the training process described by `args`, or `None` if no source was
/// given. Shared by the standalone CLI binary and brush-app's headless path.
pub fn build_process(args: &Cli) -> Option<RunningProcess> {
    let source = args.source.clone()?;
    let cli_config = args.train_stream.clone();
    Some(create_process(source, async move |init| {
        Some(brush_process::args_file::merge_configs(&init, &cli_config))
    }))
}

/// Render an animation headlessly from the `--render-anim` config. The
/// positional source is a single `.ply` (written to `--anim-out`) or a folder
/// of plys (one video each, into `--anim-out` when it is a directory, otherwise
/// next to each ply).
pub async fn run_render_animation(args: &Cli) -> Result<(), anyhow::Error> {
    use anyhow::Context;

    let config_path = args
        .animation
        .render_anim
        .as_ref()
        .expect("render_anim must be set in render mode");

    // The positional source is reused as the splat path; only local paths work.
    let source = args.source.as_ref().expect("validate guarantees a source");
    let DataSource::Path(source_path) = source else {
        anyhow::bail!("--render-anim needs a local .ply file or folder, not {source}");
    };
    let source_path = std::path::PathBuf::from(source_path);

    let config_str = tokio::fs::read_to_string(config_path)
        .await
        .with_context(|| format!("reading {}", config_path.display()))?;
    let mut config =
        brush_anim::AnimationConfig::from_json(&config_str).context("parsing animation config")?;
    // The viewer keeps this invariant live; a hand-edited config might not, and
    // a too-short timeline underflows the wrap math in `pose_at_frame`.
    config.num_frames = config.num_frames.max(config.min_frames());

    let device: burn::tensor::Device = brush_process::burn_init_setup().await.into();

    let (width, height) = (args.animation.anim_width, args.animation.anim_height);
    // Cameras only depend on the config + resolution, so build them once.
    let cameras = config.render_cameras(width, height);

    // Build the (ply, output) work list.
    let jobs: Vec<(std::path::PathBuf, std::path::PathBuf)> = if source_path.is_dir() {
        let mut plys: Vec<std::path::PathBuf> = std::fs::read_dir(&source_path)
            .with_context(|| format!("reading directory {}", source_path.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("ply")))
            .collect();
        plys.sort();
        if plys.is_empty() {
            anyhow::bail!("no .ply files found in {}", source_path.display());
        }

        // An `--anim-out` that is (or looks like) a directory collects the
        // videos there; otherwise each video is written next to its ply.
        let out = &args.animation.anim_out;
        let out_dir = if out.is_dir() || out.extension().is_none() {
            std::fs::create_dir_all(out)
                .with_context(|| format!("creating output directory {}", out.display()))?;
            Some(out.clone())
        } else {
            None
        };

        plys.into_iter()
            .map(|ply| {
                let mp4 = ply.with_extension("mp4");
                let out = match &out_dir {
                    Some(dir) => dir.join(mp4.file_name().expect("ply has a file name")),
                    None => mp4,
                };
                (ply, out)
            })
            .collect()
    } else {
        vec![(source_path.clone(), args.animation.anim_out.clone())]
    };

    for (ply, out) in jobs {
        render_one(&device, &config, &cameras, &ply, &out, width, height)
            .await
            .with_context(|| format!("rendering {}", ply.display()))?;
    }
    Ok(())
}

/// Renders a single ply to `out` using already-built `cameras`.
async fn render_one(
    device: &burn::tensor::Device,
    config: &brush_anim::AnimationConfig,
    cameras: &[brush_render::camera::Camera],
    ply: &std::path::Path,
    out: &std::path::Path,
    width: u32,
    height: u32,
) -> Result<(), anyhow::Error> {
    use anyhow::Context;
    use brush_render::gaussian_splats::SplatRenderMode;

    let ply_bytes = tokio::fs::read(ply)
        .await
        .with_context(|| format!("reading {}", ply.display()))?;
    let message = brush_serde::load_splat_from_ply(std::io::Cursor::new(ply_bytes), None)
        .await
        .context("loading splat from ply")?;
    let mode = message.meta.render_mode.unwrap_or(SplatRenderMode::Default);
    let splats = message.data.into_splats(device, mode);

    let bar = ProgressBar::new(cameras.len() as u64)
        .with_style(
            ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .expect("Invalid indicatif config")
                .progress_chars("◍○○"),
        )
        .with_message(format!(
            "frames → {}",
            out.file_name().unwrap_or_default().to_string_lossy()
        ));

    brush_anim::render_to_mp4(
        splats,
        cameras.to_vec(),
        config.fps,
        config.background,
        config.splat_scale,
        width,
        height,
        out,
        |done, _total| bar.set_position(done as u64),
    )
    .await?;

    bar.finish_and_clear();
    log::info!("Wrote animation to {}", out.display());
    Ok(())
}

/// Initialize the backend, then drive `process` to completion on the CLI UI.
pub async fn run_headless(
    process: RunningProcess,
    train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    brush_process::burn_init_setup().await;
    run_cli_ui(process, train_stream_config).await
}

/// Run the CLI: pin the trainer stream to a dedicated [`Actor`] thread,
/// drive the indicatif UI on the main task.
pub async fn run_cli_ui(
    mut process: RunningProcess,
    #[allow(unused)] train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    // Pump the trainer stream from a dedicated Actor thread; the
    // indicatif UI loop below consumes its output on the main task.
    let (tx, mut messages) = mpsc::unbounded_channel();
    let trainer = Actor::new("cli-trainer");
    trainer
        .run(move || async move {
            while let Some(msg) = process.stream.next().await {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        })
        .detach();

    // Hold the actor for the lifetime of the UI loop; dropping it
    // would kill the pump.
    let _trainer = trainer;

    // Initialize the logger with indicatif integration to prevent
    // progress bars from clobbering log output.
    let sp = {
        let mut builder = env_logger::builder();
        builder.target(env_logger::Target::Stdout);
        let logger = builder.build();
        let level = logger.filter();
        let multi = MultiProgress::new();

        LogWrapper::new(multi.clone(), logger)
            .try_init()
            .expect("Failed to initialize logger");
        log::set_max_level(level);

        multi
    };

    let main_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indacitif config")
            .tick_strings(&[
                "🖌️      ",
                "█🖌️     ",
                "▓█🖌️    ",
                "░▓█🖌️   ",
                "•░▓█🖌️  ",
                "·•░▓█🖌️ ",
                " ·•░▓🖌️ ",
                "  ·•░🖌️ ",
                "   ·•🖌️ ",
                "    ·🖌️ ",
                "     🖌️ ",
                "    🖌️ █",
                "   🖌️ █▓",
                "  🖌️ █▓░",
                " 🖌️ █▓░•",
                "🖌️ █▓░•·",
                "🖌️ ▓░•· ",
                "🖌️ ░•·  ",
                "🖌️ •·   ",
                "🖌️ ·    ",
                "🖌️      ",
            ]),
    );

    let stats_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["ℹ️", "ℹ️"]),
    );

    let train_progress = {
        let tc = &train_stream_config.train_config;
        let bar = ProgressBar::new(tc.total_iters() as u64)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({per_sec}, {eta} remaining)",
            )
            .expect("Invalid indicatif config").progress_chars("◍○○"),
        )
        .with_message("Steps");
        sp.add(bar)
    };

    let main_spinner = sp.add(main_spinner);
    main_spinner.enable_steady_tick(Duration::from_millis(120));

    let eval_spinner = sp.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .expect("Invalid indicatif config")
                .tick_strings(&["✅", "✅"]),
        ),
    );

    eval_spinner.set_message("waiting for dataset...");

    let stats_spinner = sp.add(stats_spinner);
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        let _ =
            sp.println("ℹ️  running in debug mode, compile with --release for best performance");
    }

    #[allow(unused_mut)]
    let mut duration = Duration::from_secs(0);

    while let Some(msg) = messages.recv().await {
        let _span = trace_span!("CLI UI").entered();

        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                let _ = sp.println("❌ Encountered an error");
                return Err(error);
            }
        };

        match msg {
            ProcessMessage::NewProcess => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { name, training, .. } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    break;
                }
                main_spinner.set_message(format!("Loading {name}..."));
            }
            ProcessMessage::SplatsUpdated { .. } => {}
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainConfig { .. } => {}
                TrainMessage::Dataset { dataset } => {
                    let train_views = dataset.train.views.len();
                    let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                    log::info!(
                        "Loaded dataset with {train_views} training, {eval_views} eval views",
                    );
                    main_spinner.set_message(format!(
                        "Loading dataset with {train_views} training, {eval_views} eval views",
                    ));
                    if eval_views > 0 {
                        eval_spinner.set_message(format!(
                            "evaluating {} views every {} steps",
                            eval_views, train_stream_config.process_config.eval_every,
                        ));
                    } else {
                        eval_spinner.finish_and_clear();
                    }
                }
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    lod_progress,
                    ..
                } => {
                    if let Some((lod, total_lods)) = lod_progress {
                        main_spinner.set_message(format!("LOD {lod}/{total_lods}"));
                    } else {
                        main_spinner.set_message("Training");
                    }
                    train_progress.set_position(iter as u64);
                    duration = total_elapsed;
                }
                TrainMessage::RefineStep {
                    cur_splat_count,
                    iter,
                    ..
                } => {
                    stats_spinner.set_message(format!("Current splat count {cur_splat_count}"));
                    log::info!("Refine iter {iter}, {cur_splat_count} splats.");
                }
                TrainMessage::EvalResult {
                    iter,
                    avg_psnr,
                    avg_ssim,
                } => {
                    log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");

                    eval_spinner.set_message(format!(
                        "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                    ));
                }
                TrainMessage::DoneTraining => {}
            },
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
                main_spinner.set_message("Completed loading");
                stats_spinner.set_message("Completed loading");
            }
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
                sp.println(format!("⚠️: {error}"))?;
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    let duration_secs = Duration::from_secs(duration.as_secs());
    let _ = sp.println(format!(
        "Training took {}",
        humantime::format_duration(duration_secs)
    ));

    log::info!(
        "Done training! Took {:?}.",
        humantime::format_duration(duration_secs)
    );

    Ok(())
}
