use clap::Parser;
use serde::Deserialize;

#[derive(Clone, Parser, Deserialize)]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_steps: u32,

    /// Max nr. of splats. This is an upper bound, but the actual final number of splats might be lower than this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-5")]
    pub lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[arg(long, help_heading = "Training options", default_value = "1e-6")]
    pub lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[arg(long, help_heading = "Training options", default_value = "50")]
    pub mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[arg(long, help_heading = "Training options", default_value = "20.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    pub lr_scale: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "6e-3")]
    pub lr_scale_end: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    pub lr_rotation: f64,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(long, help_heading = "Refine options", default_value = "200")]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.00009")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[arg(long, help_heading = "Refine options", default_value = "0.6")]
    pub growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Weight of the opacity loss.
    #[arg(long, help_heading = "Training options", default_value = "1e-9")]
    pub opac_loss_weight: f32,

    /// Weight of the opacity loss.
    #[arg(long, help_heading = "Training options", default_value = "1e-8")]
    pub scale_loss_weight: f32,

    /// How long to apply aux losses and augementations for (1 being the full training duration).
    #[arg(long, help_heading = "Training options", default_value = "0.85")]
    pub aux_loss_time: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}
