use brush_dataset::config::{LoadDataseConfig, ModelConfig};
use brush_train::config::TrainConfig;
use clap::{Args, Parser};

#[derive(Clone, Args)]
pub struct ProcessConfig {
    /// Random seed.
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,

    /// Iteration to resume from
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,

    /// Eval every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "1000")]
    pub eval_every: u32,
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    pub eval_save_to_disk: bool,

    /// Export every this many steps.
    #[arg(long, help_heading = "Process options", default_value = "5000")]
    pub export_every: u32,
    /// Location to put exported files. By default uses the cwd.
    ///
    /// This path can be set to be relative to the CWD.
    #[arg(long, help_heading = "Process options", default_value = ".")]
    pub export_path: String,
    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "export_{iter}.ply"
    )]
    pub export_name: String,
}

#[derive(Parser, Clone)]
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
        Self::parse_from([""])
    }
}

#[derive(Clone, Args)]
pub struct RerunConfig {
    /// Whether to enable rerun.io logging for this run.
    #[arg(long, help_heading = "Rerun options", default_value = "false")]
    pub rerun_enabled: bool,
    /// How often to log basic training statistics.
    #[arg(long, help_heading = "Rerun options", default_value = "50")]
    pub rerun_log_train_stats_every: u32,
    /// How often to log out the full splat point cloud to rerun (warning: heavy).
    #[arg(long, help_heading = "Rerun options")]
    pub rerun_log_splats_every: Option<u32>,
    /// The maximum size of images from the dataset logged to rerun.
    #[arg(long, help_heading = "Rerun options", default_value = "512")]
    pub rerun_max_img_size: u32,
}
