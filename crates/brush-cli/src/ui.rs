use std::time::Duration;

use brush_process::process_loop::{ProcessMessage, RunningProcess};
use indicatif::{ProgressBar, ProgressStyle};

pub async fn process_ui(process: RunningProcess) {
    let mut process = process;

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

    let eval_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["✅", "✅"]),
    );

    let train_progress = ProgressBar::new(process.start_args.train_config.total_steps as u64)
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

    main_spinner.enable_steady_tick(Duration::from_millis(120));

    eval_spinner.set_message(format!(
        "evaluating every {} steps",
        process.start_args.process_config.eval_every,
    ));

    while let Some(msg) = process.messages.recv().await {
        match msg {
            ProcessMessage::NewSource => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { training } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI");
                    break;
                }
                main_spinner.set_message("Loading data...");
            }
            ProcessMessage::Error(error) => {
                let _ = sp.println(format!("❌ Error: {error}"));
                break;
            }
            ProcessMessage::ViewSplats { .. } => {
                // I guess we're already showing a warning.
            }
            ProcessMessage::Dataset { data } => {
                main_spinner.set_message(format!(
                    "Loading data... {} training, {} eval views",
                    data.train.views.len(),
                    data.eval.as_ref().map_or(0, |v| v.views.len()),
                ));

                if let Some(val) = data.eval.as_ref() {
                    eval_spinner.set_message(format!(
                        "evaluating {} views every {} steps",
                        val.views.len(),
                        process.start_args.process_config.eval_every,
                    ));
                }
            }
            ProcessMessage::DoneLoading { .. } => {
                main_spinner.set_message("Dataset loaded");
            }
            ProcessMessage::TrainStep {
                splats: _,
                stats: _,
                iter,
                timestamp: _,
            } => {
                main_spinner.set_message("Training");
                train_progress.set_position(iter as u64);
                // Progress bar.
            }
            ProcessMessage::RefineStep { .. } => {
                // Do we show this info somewhere?
            }
            ProcessMessage::EvalResult { iter, eval } => {
                let psnr = eval.avg_psnr();
                let ssim = eval.avg_ssim();
                eval_spinner.set_message(format!("Eval iter {iter}: PSNR {psnr}, ssim {ssim}"));
                // Show eval results.
            }
        }
    }
}
