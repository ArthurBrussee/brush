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
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::trace_span;

pub mod mesh_eval;
pub mod mesh_extract;
pub mod mesh_render;

#[derive(Parser, Clone)]
#[command(
    author,
    version,
    arg_required_else_help = false,
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

    /// Extract a triangle mesh from a trained splat (GOF-style mesh
    /// extraction). Loads the dataset's cameras and the splat PLY, then
    /// writes the mesh to `--out-mesh`.
    #[arg(long, help_heading = "Mesh extraction", conflicts_with = "with_viewer")]
    pub extract_mesh: bool,

    /// Output path for the extracted mesh (binary PLY).
    #[arg(long, help_heading = "Mesh extraction", default_value = "mesh.ply")]
    pub out_mesh: std::path::PathBuf,

    /// Near plane for frustum culling seed points.
    #[arg(long, help_heading = "Mesh extraction", default_value = "0.02")]
    pub mesh_near: f32,

    /// Far plane for frustum culling seed points.
    #[arg(long, help_heading = "Mesh extraction", default_value = "1e6")]
    pub mesh_far: f32,

    /// Subsample the input splat PLY by this stride before extracting.
    /// Each Gaussian produces 9 seed points; with ~1M splats and stride 1
    /// the CPU Delaunay handles 9M points, which is workable but slow —
    /// use a higher stride for fast iteration.
    #[arg(long, help_heading = "Mesh extraction")]
    pub splat_subsample: Option<u32>,

    /// Iso-value for the level set. GOF default 0.5 — carves the surface
    /// where transmittance has dropped to half, the principled "actual
    /// material boundary" choice. Higher values (0.9+) fill more
    /// background on diffuse-trained splats but fatten foreground
    /// geometry into a speckled halo, so PSNR can be misleading.
    #[arg(long, help_heading = "Mesh extraction", default_value = "0.5")]
    pub iso_value: f32,

    /// Skip writing the extracted mesh as PLY to disk. The mesh PLY for
    /// a 10M-face garden is ~30 MB and even with buffering takes a few
    /// seconds to flush; when iterating on the renderer / colour eval
    /// we just want the eval-render PNGs, not the PLY.
    #[arg(long, help_heading = "Mesh extraction")]
    pub skip_mesh_write: bool,

    /// After extracting the mesh, render it at each training-camera
    /// viewpoint via `f3d` and report mean PSNR vs the ground-truth
    /// images. Renders land in `{out-mesh dir}/eval_renders/`. Set
    /// `--eval-views=N` to evaluate just the first N views (default 0
    /// = skip eval).
    #[arg(long, help_heading = "Mesh extraction", default_value = "0")]
    pub eval_views: usize,

    /// Target fraction of the scene's all-views directional coverage
    /// to fill before stopping view subsetting, in `[0, 1]`. Default
    /// `0.8` — heavily-redundant captures drop most views, near-
    /// optimal ones drop only a few. See `brush_mesh::view_select`.
    /// Set to `1.0` to disable subsetting entirely.
    #[arg(long, help_heading = "Mesh extraction", default_value = "0.8")]
    pub view_coverage: f32,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if self.extract_mesh && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "--extract-mesh requires a --source pointing at a dataset directory containing a PLY",
            ));
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
