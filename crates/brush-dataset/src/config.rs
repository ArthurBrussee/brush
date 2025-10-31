use clap::{Args, ValueEnum};

#[derive(Clone, Debug, Args)]
pub struct ModelConfig {
    /// SH degree of splats.
    #[arg(long, help_heading = "Model Options", default_value = "3")]
    pub sh_degree: u32,
}

#[derive(Default, ValueEnum, Clone, Copy, Eq, PartialEq, Debug)]
pub enum AlphaMode {
    #[default]
    Masked,
    Transparent,
}

#[derive(Clone, Debug, Args)]
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
}
