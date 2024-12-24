use brush_dataset::{LoadDataseConfig, ModelConfig};
use brush_train::train::TrainConfig;
use burn::config::Config;
use clap::Args;

#[derive(Config, Args)]
pub struct ProcessConfig {
    /// Random seed.
    #[config(default = 42)]
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,
    /// Eval every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "1000")]
    #[config(default = 1000)]
    pub eval_every: u32,
    /// Export every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "5000")]
    #[config(default = 5000)]
    pub export_every: u32,
    /// Export path
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./export_{iter}.ply"
    )]
    #[config(default = "String::from(\"./export_{iter}.ply\")")]
    pub export_path: String,
}

#[derive(Config, Args)]
pub struct RerunConfig {
    #[arg(long, help_heading = "Rerun options", default_value = "false")]
    #[config(default = false)]
    pub rerun_enabled: bool,
    #[arg(long, help_heading = "Rerun options", default_value = "50")]
    #[config(default = 50)]
    pub rerun_log_train_stats_every: u32,
    #[arg(long, help_heading = "Rerun options")]
    pub rerun_log_splats_every: Option<u32>,
}

#[derive(Config, Args)]
pub struct ProcessArgs {
    #[clap(flatten)]
    pub train_config: TrainConfig,
    #[clap(flatten)]
    pub model_config: ModelConfig,
    #[clap(flatten)]
    pub load_config: LoadDataseConfig,
    #[clap(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    pub rerun_config: RerunConfig,
}

impl Default for ProcessArgs {
    fn default() -> Self {
        Self {
            train_config: TrainConfig::new(),
            model_config: ModelConfig::new(),
            load_config: LoadDataseConfig::new(),
            process_config: ProcessConfig::new(),
            rerun_config: RerunConfig::new(),
        }
    }
}
