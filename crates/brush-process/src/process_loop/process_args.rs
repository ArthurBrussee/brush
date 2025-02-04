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
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    #[config(default = false)]
    pub eval_save_to_disk: bool,

    /// Export every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "5000")]
    #[config(default = 5000)]
    pub export_every: u32,

    /// Location to put exported files. By default uses the cwd.
    ///
    /// This path can be set to be relative to the CWD.
    #[arg(long, help_heading = "Process options")]
    pub export_path: Option<String>,

    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./export_{iter}.ply"
    )]
    #[config(default = "String::from(\"./export_{iter}.ply\")")]
    pub export_name: String,

    /// Iterationto resume from
    #[config(default = 0)]
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,
}

#[derive(Config, Args)]
pub struct RerunConfig {
    /// Whether to enable rerun.io logging for this run.
    #[arg(long, help_heading = "Rerun options", default_value = "false")]
    #[config(default = false)]
    pub rerun_enabled: bool,
    /// How often to log basic training statistics.
    #[arg(long, help_heading = "Rerun options", default_value = "50")]
    #[config(default = 50)]
    pub rerun_log_train_stats_every: u32,
    /// How often to log out the full splat point cloud to rerun (warning: heavy).
    #[arg(long, help_heading = "Rerun options")]
    pub rerun_log_splats_every: Option<u32>,
    /// The maximum size of images from the dataset logged to rerun.
    #[arg(long, help_heading = "Rerun options", default_value = "512")]
    #[config(default = 512)]
    pub rerun_max_img_size: u32,
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
