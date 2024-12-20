use brush_dataset::{LoadDataseConfig, ModelConfig};
use brush_train::train::TrainConfig;
use burn::config::Config;
use clap::Args;

#[derive(Config, Default, Args)]
pub struct ProcessConfig {
    /// Random seed.
    #[config(default = 42)]
    #[arg(long, help_heading = "Training options", default_value = "42")]
    pub seed: u64,
    /// Eval every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "1000")]
    #[config(default = 1000)]
    pub eval_every: u32,
    /// Eval every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "1000")]
    #[config(default = 1000)]
    pub export_every: u32,
}

#[derive(Clone, Default, Args)]
pub struct ProcessArgs {
    #[clap(flatten)]
    pub train_config: TrainConfig,
    #[clap(flatten)]
    pub init_config: ModelConfig,
    #[clap(flatten)]
    pub load_config: LoadDataseConfig,
    #[clap(flatten)]
    pub process_config: ProcessConfig,
}
