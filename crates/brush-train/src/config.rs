use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_steps: u32,

    #[arg(long, help_heading = "Training options")]
    pub render_mode: Option<SplatRenderMode>,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-7")]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[arg(long, help_heading = "Training options", default_value = "50.0")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "20.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "7e-3")]
    pub lr_scale: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale_end: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Max nr. of splats. This is an upper bound, but the actual final number of splats might be lower than this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Frequency of 'refinement' where gaussians densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(long, help_heading = "Refine options", default_value = "200")]
    pub refine_every: u32,

    // Refine final frequency
    #[arg(long, help_heading = "Refine options", default_value = "3000")]
    pub refine_every_final: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.003")]
    pub growth_grad_threshold: f32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Factor of the scaling decay.
    #[arg(long, help_heading = "Training options", default_value = "0.002")]
    pub scale_decay: f32,

    /// How long to apply aux losses and augementations for (1 being the full training duration).
    #[arg(long, help_heading = "Training options", default_value = "0.9")]
    pub aux_loss_time: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    // N views used in multi view densification
    #[arg(long, help_heading = "Refine options", default_value = "10")]
    pub n_views: i32,

    // High error threshold used in multi view densification
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub high_error_threshold: f32,

    // Min importance score used in multi view densification
    #[arg(long, help_heading = "Refine options", default_value = "5.0")]
    pub min_importance_score: f32,

    // Final min opacity
    #[arg(long, help_heading = "Refine options", default_value = "0.005")]
    pub final_min_opacity: f32,

    // Final max pruning score
    #[arg(long, help_heading = "Refine options", default_value = "0.95")]
    pub final_max_pruning_score: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
