#![recursion_limit = "256"]

use brush_process::{config::ProcessArgs, message::ProcessMessage};
use brush_vfs::DataSource;
use clap::{Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;
use tokio_stream::{Stream, StreamExt};

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
    pub process: ProcessArgs,
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

pub async fn process_ui(
    stream: impl Stream<Item = anyhow::Result<ProcessMessage>>,
    process_args: ProcessArgs,
) -> Result<(), anyhow::Error> {
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

    let eval_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["✅", "✅"]),
    );

    let train_progress = ProgressBar::new(process_args.train_config.total_steps as u64)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({per_sec}, {eta} remaining)",
            )
            .expect("Invalid indicatif config").progress_chars("◍○○"),
        )
        .with_message("Steps");

    let sp = indicatif::MultiProgress::new();
    let main_spinner = sp.add(main_spinner);
    let train_progress = sp.add(train_progress);
    let eval_spinner = sp.add(eval_spinner);
    let stats_spinner = sp.add(stats_spinner);

    main_spinner.enable_steady_tick(Duration::from_millis(120));
    eval_spinner.set_message(format!(
        "evaluating every {} steps",
        process_args.process_config.eval_every,
    ));
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        let _ =
            sp.println("ℹ️  running in debug mode, compile with --release for best performance");
    }

    let mut stream = std::pin::pin!(stream);
    let mut duration = Duration::from_secs(0);

    // TODO: Unify logging & CLI UI somehow.
    while let Some(msg) = stream.next().await {
        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                let _ = sp.println("❌ Encountered an error");
                return Err(error);
            }
        };

        match msg {
            ProcessMessage::NewSource => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { training } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    break;
                }
                main_spinner.set_message("Loading data...");
            }
            ProcessMessage::ViewSplats { .. } => {
                // I guess we're already showing a warning.
            }
            ProcessMessage::Dataset { dataset } => {
                let train_views = dataset.train.views.len();
                let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                log::info!("Loaded dataset with {train_views} training, {eval_views} eval views",);
                main_spinner.set_message(format!(
                    "Loading dataset with {train_views} training, {eval_views} eval views",
                ));
                if let Some(val) = dataset.eval.as_ref() {
                    eval_spinner.set_message(format!(
                        "evaluating {} views every {} steps",
                        val.views.len(),
                        process_args.process_config.eval_every,
                    ));
                }
            }
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
                main_spinner.set_message("Completed loading");
                stats_spinner.set_message("Completed loading");
            }
            ProcessMessage::TrainStep {
                iter,
                total_elapsed,
                ..
            } => {
                main_spinner.set_message("Training");
                train_progress.set_position(iter as u64);
                duration = total_elapsed;
            }
            ProcessMessage::RefineStep {
                cur_splat_count,
                iter,
                ..
            } => {
                stats_spinner.set_message(format!("Current splat count {cur_splat_count}"));
                log::info!("Refine iter {iter}, {cur_splat_count} splats.");
            }
            ProcessMessage::EvalResult {
                iter,
                avg_psnr,
                avg_ssim,
            } => {
                log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");

                eval_spinner.set_message(format!(
                    "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                ));
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
