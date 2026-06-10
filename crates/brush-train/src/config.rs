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

    /// Weight of the flattening regularizer (PGSR `L_s`): penalizes each splat's
    /// smallest scale axis so Gaussians commit to planes. Scene-scale dependent.
    #[arg(long, help_heading = "Geometry options", default_value = "10.0")]
    pub flatten_weight: f32,

    /// Weight of the single-view depth↔normal consistency loss (PGSR `L_svgeo`).
    /// Unitless (normal agreement), so scene-independent.
    #[arg(long, help_heading = "Geometry options", default_value = "0.05")]
    pub depth_normal_weight: f32,

    /// Weight of the metric depth supervision loss: L1 between the rendered
    /// depth and the per-view `LiDAR` depth (when the dataset provides it),
    /// confidence-weighted and sparse (at the `LiDAR` grid). Ground-truth
    /// supervision, so it runs from iteration 0 (not gated by
    /// `--geo-from-iter`). No-op without depth data.
    #[arg(long, help_heading = "Geometry options", default_value = "0.2")]
    pub depth_loss_weight: f32,

    /// Weight of the depth-distortion loss (GOF `L_d`: squared pairwise error
    /// over NDC-mapped depths, normalized per pixel): pulls each ray's splats
    /// onto a single depth so the surface stops being a fuzzy shell.
    #[arg(long, help_heading = "Geometry options", default_value = "100.0")]
    pub distortion_weight: f32,

    /// Master switch for the self-consistency geometry regularizers: the
    /// iteration to turn them on at. Unset = off (their weights above are
    /// ignored). Set it to enable flatten + depth-normal + depth-distortion
    /// from that iteration onward.
    #[arg(long, help_heading = "Geometry options")]
    pub geo_from_iter: Option<u32>,

    /// `LiDAR` init voxel grid: cells along the cloud's longest axis. Higher =
    /// finer/denser init. Density is scene-scale invariant (metric voxel).
    #[arg(long, help_heading = "Geometry options", default_value = "256")]
    pub lidar_grid: f32,

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

    /// Scene radius used for random splat initialization (the half-extent of
    /// the scene around its center). When no point-cloud init is provided,
    /// splats are sampled in camera frustums at depths bracketing the scene
    /// center (min standoff `0.2·scale`, out to `d_center + scale`). By
    /// default this is the mean camera-to-centroid distance (floored by the
    /// camera-spacing estimate).
    #[arg(long, help_heading = "Training options")]
    pub random_init_scene_scale: Option<f32>,
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

    /// Whether the self-consistency geometry regularizers (depth-normal,
    /// distortion, flatten) are active at `iter` (gated by `geo_from_iter`).
    pub fn geo_regs_on(&self, iter: u32) -> bool {
        self.geo_from_iter.is_some_and(|from| iter >= from)
    }

    /// Whether any active loss needs the geometry render pass at `iter`:
    /// the metric depth loss runs from iteration 0, the regularizers from
    /// `geo_from_iter`. The depth loss additionally needs per-batch depth,
    /// which the trainer checks per step.
    pub fn needs_geometry(&self, iter: u32) -> bool {
        self.depth_loss_weight > 0.0
            || (self.geo_regs_on(iter)
                && (self.depth_normal_weight > 0.0 || self.distortion_weight > 0.0))
    }
}
