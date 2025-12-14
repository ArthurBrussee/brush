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
    /// Location to put exported files. By default uses the data directory if available,
    /// or the CWD otherwise.
    ///
    /// This path can be set as a relative path.
    #[arg(long, help_heading = "Process options", default_value = "./")]
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
#[cfg(feature = "training")]
pub struct TrainStreamConfig {
    #[clap(flatten)]
    pub train_config: brush_train::config::TrainConfig,
    #[clap(flatten)]
    pub model_config: brush_dataset::config::ModelConfig,
    #[clap(flatten)]
    pub load_config: brush_dataset::config::LoadDataseConfig,
    #[clap(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    pub rerun_config: brush_rerun::RerunConfig,
}

#[cfg(feature = "training")]
impl Default for TrainStreamConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

#[cfg(not(feature = "training"))]
#[derive(Parser, Default, Clone)]
pub struct TrainStreamConfig {}
