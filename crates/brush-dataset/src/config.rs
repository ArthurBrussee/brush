use brush_render::AlphaMode;
use clap::Args;
use serde::{Deserialize, Serialize};

/// Default Cache budget for packed scene batches. 6 GB on native; less on
/// wasm since the whole heap is bounded by browser limits.
#[cfg(not(target_family = "wasm"))]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "6GiB";
#[cfg(target_family = "wasm")]
const DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE: &str = "2GiB";

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
pub struct LoadDatasetConfig {
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
    /// Max size of the cache for frames of the dataset, larger values usually improve performance for large datasets at the cost of more memory usage, can be e.g. 6G, 6000M, 6000MiB, 6000MB
    #[arg(long, help_heading = "Dataset Options", default_value = DEFAULT_MAX_SCENE_BATCH_CACHE_SIZE, value_parser = parse_size)]
    pub max_scene_batch_cache_size: u64,
}

fn parse_size(s: &str) -> Result<u64, parse_size::Error> {
    parse_size::parse_size(s)
}
