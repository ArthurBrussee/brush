use brush_render::gaussian_splats::SplatRenderMode;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Clone, Parser, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_train_iters: u32,

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
    #[arg(long, help_heading = "Training options", default_value = "10.0")]
    pub lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[arg(long, help_heading = "Training options", default_value = "0.012")]
    pub lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    pub lr_scale: f64,

    /// Learning rate for the rotation parameters.
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    pub lr_rotation: f64,

    /// Max nr. of splats. This is only an upper bound, the actual final number of splats is NOT determined by this.
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    pub max_splats: u32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[arg(
        long,
        help_heading = "Refine options",
        default_value = "200",
        value_parser = clap::value_parser!(u32).range(1..)
    )]
    pub refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[arg(long, help_heading = "Refine options", default_value = "0.0025")]
    pub growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[arg(long, help_heading = "Refine options", default_value = "0.25")]
    pub growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    pub growth_stop_iter: u32,

    /// Prune-and-resample any splat whose max screen-space extent exceeds this
    /// fraction of the image dimension.
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
    pub kill_at_screen_size: f32,

    /// Mip-Splatting 3D-filter strength: each splat gets a frozen world-space
    /// scale floor of `sqrt(factor)` pixels at its nearest observing camera,
    /// folded into derived scales/opacity (and baked at export). 0 disables.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    #[serde(default = "default_min_scale_factor")]
    pub min_scale_factor: f32,

    /// Weight of the per-splat screen-area loss. Works as a nudge toward small on-screen
    /// footprints.
    #[arg(long, help_heading = "Training options", default_value = "0.05")]
    pub screen_area_penalty: f32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    pub ssim_weight: f32,

    /// Factor of the opacity decay.
    #[arg(long, help_heading = "Training options", default_value = "0.004")]
    pub opac_decay: f32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    pub match_alpha_weight: f32,

    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    pub lpips_loss_weight: f32,

    /// Base background color (R,G,B) used during training.
    #[arg(
        long,
        help_heading = "Training options",
        default_value = "0,0,0",
        value_delimiter = ',',
        num_args = 3
    )]
    pub background_color: Vec<f32>,

    /// Strength of random noise added to the background color each step.
    /// Noise is uniform in [-strength, +strength], clamped to [0, 1].
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    pub background_noise_strength: f32,

    /// Number of LOD levels to generate after initial training (0 = disabled).
    #[arg(long, help_heading = "LOD options", default_value = "0")]
    pub lod_levels: u32,

    /// Number of refinement training steps per LOD level.
    #[arg(long, help_heading = "LOD options", default_value = "5000")]
    pub lod_refine_steps: u32,

    /// Percentage of gaussians to keep at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_decimation_keep: u32,

    /// Percentage to scale source images at each LOD level (1-100).
    #[arg(long, help_heading = "LOD options", default_value = "50")]
    pub lod_image_scale: u32,

    /// Scene scale used for random splat initialization.
    /// When no init is provided, splats are randomly placed
    /// inside camera frustums up to this depth. By default this is
    /// estimated from the camera spacing (with a 1m minimum).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,

    /// Enable hybrid appearance compensation (recommended): per-camera
    /// PPISP vignetting plus a per-view bilateral grid of PPISP exposure +
    /// color parameters, applied to the render before the loss so exposure /
    /// white-balance variation isn't baked into the splats.
    #[arg(long, help_heading = "Appearance options", default_value = "false")]
    #[serde(default)]
    pub ppisp_grid: bool,

    /// Hybrid grid holds one log2-exposure scalar per cell instead of
    /// exposure + color.
    #[arg(long, help_heading = "Appearance options", default_value = "false")]
    #[serde(default)]
    pub ppisp_grid_expose_only: bool,

    /// Hybrid grid additionally holds per-cell CRF (tone curve) offsets.
    /// On by default: without tone-curve freedom the grid can only apply
    /// display-space gain, which under-fits tone-mapped exposure changes
    /// and leaves floaters. `--ppisp-grid-crf=false` disables.
    #[arg(
        long,
        help_heading = "Appearance options",
        default_value = "true",
        default_missing_value = "true",
        num_args = 0..=1,
        action = clap::ArgAction::Set
    )]
    #[serde(default = "default_ppisp_grid_crf")]
    pub ppisp_grid_crf: bool,

    /// Learn a per-camera CRF (tone curve) applied after the hybrid grid.
    #[arg(long, help_heading = "Appearance options", default_value = "false")]
    #[serde(default)]
    pub ppisp_crf_per_camera: bool,

    /// Comparison mode: per-view affine bilateral grids (BilaRF-style).
    #[arg(long, help_heading = "Appearance options", default_value = "false")]
    #[serde(default)]
    pub bilateral_grid: bool,

    /// Bilateral grid dimensions as `x,y,guidance`.
    #[arg(
        long,
        help_heading = "Appearance options",
        default_value = "16,16,8",
        value_delimiter = ','
    )]
    #[serde(default = "default_bilagrid_dims")]
    pub bilagrid_dims: Vec<u32>,

    /// Weight of the bilateral grid's total-variation regularizer.
    #[arg(long, help_heading = "Appearance options", default_value = "10.0")]
    #[serde(default = "default_bilagrid_tv_weight")]
    pub bilagrid_tv_weight: f32,

    /// Weight of the grid's EMA-anchored mean-to-identity regularizer
    /// (hybrid mode; keeps the dataset-mean correction at identity).
    /// Channel-mean normalized, directly comparable to spirulae's
    /// `bilagrid_mean_reg_weight`. Too high caps how much exposure the
    /// grid can absorb, pushing it into floaters instead.
    #[arg(long, help_heading = "Appearance options", default_value = "10.0")]
    #[serde(default = "default_bilagrid_mean_reg")]
    pub bilagrid_mean_reg: f32,

    /// Learning rate for the bilateral grids.
    #[arg(long, help_heading = "Appearance options", default_value = "2e-3")]
    #[serde(default = "default_bilagrid_lr")]
    pub bilagrid_lr: f64,

    /// Adam betas for the per-view grid updates as `b1,b2`. The sparse
    /// updates are dense-Adam equivalent (moments decay over the gap
    /// between a view's visits), so the horizons are in global steps and
    /// the defaults match the reference implementations.
    #[arg(
        long,
        help_heading = "Appearance options",
        default_value = "0.9,0.999",
        value_delimiter = ','
    )]
    #[serde(default = "default_bilagrid_betas")]
    pub bilagrid_betas: Vec<f64>,

    /// Grid-gradient subsampling: only every Nth subgroup scatters
    /// gradients into the grid (scaled to stay unbiased; offset rotates per
    /// step). 1 disables. Cuts the dominant atomic cost of the grid
    /// backward; measured quality-neutral at the default for the heavily
    /// regularized grids.
    #[arg(long, help_heading = "Appearance options", default_value = "4")]
    #[serde(default = "default_bilagrid_grad_subsample")]
    pub bilagrid_grad_subsample: u32,

    /// Enable PPISP appearance compensation: per-frame exposure + color
    /// homography and per-camera vignetting + tone curve (physically
    /// plausible ISP model), applied to the render before the loss.
    #[arg(long, help_heading = "Appearance options", default_value = "false")]
    #[serde(default)]
    pub ppisp: bool,

    /// Learning rate for the PPISP parameters.
    #[arg(long, help_heading = "Appearance options", default_value = "2e-3")]
    #[serde(default = "default_ppisp_lr")]
    pub ppisp_lr: f64,

    /// Scale on all PPISP parameter-regularization terms.
    #[arg(long, help_heading = "Appearance options", default_value = "1.0")]
    #[serde(default = "default_ppisp_reg_scale")]
    pub ppisp_reg_scale: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::parse_from([""])
    }
}

impl TrainConfig {
    pub fn total_iters(&self) -> u32 {
        self.total_train_iters + self.lod_levels * self.lod_refine_steps
    }
}

fn default_min_scale_factor() -> f32 {
    0.1
}

fn default_bilagrid_dims() -> Vec<u32> {
    vec![16, 16, 8]
}

fn default_ppisp_grid_crf() -> bool {
    true
}

fn default_bilagrid_tv_weight() -> f32 {
    10.0
}

fn default_bilagrid_lr() -> f64 {
    2e-3
}

fn default_bilagrid_mean_reg() -> f32 {
    10.0
}

fn default_bilagrid_betas() -> Vec<f64> {
    vec![0.9, 0.999]
}

fn default_bilagrid_grad_subsample() -> u32 {
    4
}

fn default_ppisp_lr() -> f64 {
    2e-3
}

fn default_ppisp_reg_scale() -> f32 {
    1.0
}
