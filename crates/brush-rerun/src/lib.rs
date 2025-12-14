use clap::Args;

// Exclude entirely on wasm.
#[cfg(not(target_family = "wasm"))]
pub mod burn_to_rerun;
#[cfg(not(target_family = "wasm"))]
pub mod visualize_tools;

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
