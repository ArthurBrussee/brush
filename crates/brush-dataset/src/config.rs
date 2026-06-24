use brush_render::AlphaMode;
use clap::Args;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(
        long,
        help_heading = "Model Options",
        default_value = "3",
        value_parser = clap::value_parser!(u32).range(0..=4)
    )]
    pub sh_degree: u32,
}

#[derive(Clone, Debug, Args, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LoadDataseConfig {
    /// Max nr. of frames of dataset to load
    #[arg(long, help_heading = "Dataset Options")]
    pub max_frames: Option<usize>,
    /// Max resolution of images to load.
    #[arg(long, help_heading = "Dataset Options", default_value = "1920")]
    pub max_resolution: u32,
    /// Create an eval dataset by selecting every nth image
    #[arg(long, help_heading = "Dataset Options")]
    pub eval_split_every: Option<usize>,
    /// Load only every nth frame
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_frames: Option<u32>,
    /// Load only every nth point from the initial sfm data
    #[arg(long, help_heading = "Dataset Options")]
    pub subsample_points: Option<u32>,
    /// Whether to interpret an alpha channel (or masks) as transparency or masking.
    #[arg(long, help_heading = "Dataset Options")]
    pub alpha_mode: Option<AlphaMode>,
    /// Whether to rescale the dataset to suite the depth map scale
    #[arg(long, help_heading = "Dataset Options", default_value = "false")]
    pub estimate_metric_scale: bool,
}
