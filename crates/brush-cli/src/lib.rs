#![recursion_limit = "256"]

use brush_process::message::ProcessMessage;
#[cfg(feature = "training")]
use brush_process::message::TrainMessage;

use brush_process::{config::TrainStreamConfig, process::create_process};
use brush_vfs::DataSource;
use burn_wgpu::WgpuDevice;
use clap::{Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
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

pub async fn run_cli_ui(
    source: DataSource,
    train_stream_config: TrainStreamConfig,
    device: WgpuDevice,
) -> Result<(), anyhow::Error> {
    // TODO: Find a way to make logging and indicatif to play nicely with eachother.
    // let mut stream = std::pin::pin!(stream);
    // while let Some(msg) = stream.next().await {
    //     let _ = msg?;
    // }

    let (tx, rx) = tokio::sync::oneshot::channel::<TrainStreamConfig>();
    let _ = tx.send(train_stream_config.clone());

    let stream = create_process(source, rx, device);
    let mut stream = std::pin::pin!(stream);

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

    let sp = indicatif::MultiProgress::new();

    #[cfg(feature = "training")]
    let train_progress = {
        let bar = ProgressBar::new(train_stream_config.train_config.total_steps as u64)
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

    #[cfg(feature = "training")]
    let eval_spinner = sp.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .expect("Invalid indicatif config")
                .tick_strings(&["✅", "✅"]),
        ),
    );

    #[cfg(feature = "training")]
    eval_spinner.set_message(format!(
        "evaluating every {} steps",
        train_stream_config.process_config.eval_every,
    ));

    let stats_spinner = sp.add(stats_spinner);
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        let _ =
            sp.println("ℹ️  running in debug mode, compile with --release for best performance");
    }

    #[allow(unused_mut)]
    let mut duration = Duration::from_secs(0);

    while let Some(msg) = stream.next().await {
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
            ProcessMessage::NewSource { name } => {
                log::info!("Loading: {name}");
                main_spinner.set_message(format!("Loading {name}..."));
            }
            ProcessMessage::StartLoading { training } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    break;
                }
                main_spinner.set_message("Loading data...");
            }
            ProcessMessage::ViewSplats { .. } => {}
            #[cfg(feature = "training")]
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::Dataset { dataset } => {
                    let train_views = dataset.train.views.len();
                    let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                    log::info!(
                        "Loaded dataset with {train_views} training, {eval_views} eval views",
                    );
                    main_spinner.set_message(format!(
                        "Loading dataset with {train_views} training, {eval_views} eval views",
                    ));
                    if let Some(val) = dataset.eval.as_ref() {
                        eval_spinner.set_message(format!(
                            "evaluating {} views every {} steps",
                            val.views.len(),
                            train_stream_config.process_config.eval_every,
                        ));
                    }
                }
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    ..
                } => {
                    main_spinner.set_message("Training");
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
                sp.println("⚠️: {error}")?;
            }
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
