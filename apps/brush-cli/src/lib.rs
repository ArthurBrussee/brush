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

#[derive(Parser)]
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

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
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

    log::info!("Compute backend: {:?}", process.device);

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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_source_and_overrides() {
        let cli = Cli::try_parse_from([
            "brush-cli",
            "some/dataset/path",
            "--total-train-iters",
            "50",
            "--eval-split-every",
            "2",
            "--max-resolution",
            "512",
            "--sh-degree",
            "2",
            "--seed",
            "7",
        ])
        .unwrap();

        assert!(matches!(
            &cli.source,
            Some(DataSource::Path(p)) if p == "some/dataset/path"
        ));
        // Passing a source flips the viewer default off.
        assert!(!cli.with_viewer);

        let ts = &cli.train_stream;
        assert_eq!(ts.train_config.total_train_iters, 50);
        assert_eq!(ts.train_config.total_iters(), 50); // No LOD levels by default.
        assert_eq!(ts.load_config.eval_split_every, Some(2));
        assert_eq!(ts.load_config.max_resolution, 512);
        assert_eq!(ts.model_config.sh_degree, 2);
        assert_eq!(ts.process_config.seed, 7);

        // A source without a viewer is a valid combination.
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn parses_url_source() {
        let cli = Cli::try_parse_from(["brush-cli", "https://example.com/data.zip"]).unwrap();
        assert!(matches!(
            &cli.source,
            Some(DataSource::Url(u)) if u == "https://example.com/data.zip"
        ));
    }

    #[test]
    fn defaults_to_viewer_without_source() {
        let cli = Cli::try_parse_from(["brush-cli"]).unwrap();
        assert!(cli.source.is_none());
        assert!(cli.with_viewer);
        // Viewer without a source is valid (brush-app's default mode).
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn viewer_flag_with_source() {
        let cli = Cli::try_parse_from(["brush-cli", "some/path", "--with-viewer"]).unwrap();
        assert!(cli.with_viewer);
        assert!(cli.source.is_some());
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn validate_rejects_headless_without_source() {
        let mut cli = Cli::try_parse_from(["brush-cli"]).unwrap();
        cli.with_viewer = false;
        let Err(err) = cli.validate() else {
            panic!("expected validation error")
        };
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn rejects_unknown_flag() {
        assert!(Cli::try_parse_from(["brush-cli", "--not-a-real-flag"]).is_err());
    }
}
