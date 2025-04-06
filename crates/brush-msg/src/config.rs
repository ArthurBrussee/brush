use brush_dataset::{LoadDataseConfig, ModelConfig};
use burn::config::Config;
use clap::{Args, arg};

#[derive(Config, Args)]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[config(default = 30000)]
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_steps: u32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[config(default = 0.2)]
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// SSIM window size
    #[config(default = 11)]
    #[clap(long, help_heading = "Training options", default_value = "11")]
    pub ssim_window_size: usize,

    /// Start learning rate for the mean parameters.
    #[config(default = 4e-5)]
    #[arg(long, help_heading = "Training options", default_value = "4e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[config(default = 4e-7)]
    #[arg(long, help_heading = "Training options", default_value = "4e-7")]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[config(default = 1e4)]
    #[arg(long, help_heading = "Training options", default_value = "1e4")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[config(default = 3e-3)]
    #[arg(long, help_heading = "Training options", default_value = "3e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[config(default = 20.0)]
    #[arg(long, help_heading = "Training options", default_value = "20.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[config(default = 3e-2)]
    #[arg(long, help_heading = "Training options", default_value = "3e-2")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[config(default = 1e-2)]
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    pub lr_scale: f64,

    /// Learning rate for the scale parameters.
    #[config(default = 6e-3)]
    #[arg(long, help_heading = "Training options", default_value = "6e-3")]
    pub lr_scale_end: f64,

    /// Learning rate for the rotation parameters.
    #[config(default = 1e-3)]
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    pub lr_rotation: f64,

    /// Weight of the opacity loss.
    #[config(default = 1e-8)]
    #[arg(long, help_heading = "Training options", default_value = "1e-8")]
    pub opac_loss_weight: f32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[config(default = 150)]
    #[arg(long, help_heading = "Refine options", default_value = "150")]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[config(default = 0.00085)]
    #[arg(long, help_heading = "Refine options", default_value = "0.00085")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[config(default = 12500)]
    #[arg(long, help_heading = "Refine options", default_value = "12500")]
    pub growth_stop_iter: u32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    /// Max nr. of splats. This is an upper bound, but the actual final number of splats might be lower than this.
    #[config(default = 10000000)]
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,
}

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

    /// Iteration to resume from
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
