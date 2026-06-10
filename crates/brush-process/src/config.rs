use clap::{Args, Parser};
use serde::{Deserialize, Serialize};

#[derive(Clone, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ProcessConfig {
    /// Random seed.
    #[arg(long, help_heading = "Process options", default_value = "42")]
    pub seed: u64,
    /// Iteration to resume from
    #[arg(long, help_heading = "Process options", default_value = "0")]
    pub start_iter: u32,
    /// Eval every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "1000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub eval_every: u32,
    /// Save the rendered eval images to disk. Uses export-path for the file location.
    #[arg(long, help_heading = "Process options", default_value = "false")]
    pub eval_save_to_disk: bool,
    /// Export every this many steps.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "5000",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub export_every: u32,
    /// Extract and export a mesh (GOF-style) every this many steps, and always
    /// on the last step. Unset = no mesh export. With --total-train-iters 0
    /// this gives mesh-only extraction from the initial splats.
    #[arg(
        long,
        help_heading = "Process options",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub export_mesh_every: Option<u32>,
    /// Mesh-export region: the union of all camera frustums truncated at
    /// this distance (scene units; metres on metric scenes).
    #[arg(long, help_heading = "Process options", default_value = "4.0")]
    pub export_mesh_dist: f32,
    /// Location to put exported files. Supports {dataset} interpolation for the dataset
    /// folder name. Path is relative to the dataset's parent directory (or CWD if unavailable).
    /// Use "./{dataset}/" to export inside the dataset folder.
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "./{dataset}_exports/"
    )]
    pub export_path: String,
    /// Filename of exported ply file
    #[arg(
        long,
        help_heading = "Process options",
        default_value = "export_{iter}.ply"
    )]
    pub export_name: String,
}

#[derive(Parser, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainStreamConfig {
    #[clap(flatten)]
    #[serde(flatten)]
    pub train_config: brush_train::config::TrainConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub model_config: brush_dataset::config::ModelConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub load_config: brush_dataset::config::LoadDataseConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub process_config: ProcessConfig,
    #[clap(flatten)]
    #[serde(flatten)]
    pub rerun_config: brush_rerun::RerunConfig,
}

impl Default for TrainStreamConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
