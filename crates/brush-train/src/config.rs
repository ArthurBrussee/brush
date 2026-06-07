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

    /// Weight of the metric depth supervision loss: L1 between the RaDe-GS
    /// rendered depth and the per-view `LiDAR` depth (when the dataset provides
    /// it), per-pixel weighted by `ARKit` confidence. No-op without depth data.
    #[arg(long, help_heading = "Geometry options", default_value = "0.2")]
    pub depth_loss_weight: f32,

    /// Weight of the depth-distortion loss (2DGS `L_d`, squared form): penalizes
    /// each ray's weighted depth variance, concentrating its weight onto a single
    /// depth so the surface stops being a fuzzy shell. Scene-scale dependent.
    #[arg(long, help_heading = "Geometry options", default_value = "0.1")]
    pub distortion_weight: f32,

    /// Master switch for the geometry losses: the iteration to turn them on at.
    /// Unset = geometry off (the weights above are ignored). Set it to enable
    /// flatten + depth-normal + depth-distortion + metric depth from that
    /// iteration onward (they want a settled-enough reconstruction first; PGSR
    /// uses ~7000). An individual loss can still be disabled with its weight = 0.
    #[arg(long, help_heading = "Geometry options")]
    pub geo_from_iter: Option<u32>,

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
    #[arg(long, help_heading = "Training options", default_value = "1.0")]
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

    /// Voxel size (metres) for `LiDAR`-init downsampling. `0` (default)
    /// auto-derives it from the cloud extent (scene-relative density); a
    /// positive value forces a fixed metric size.
    #[arg(long, help_heading = "Init options", default_value = "0.0")]
    pub lidar_voxel_size: f32,

    /// Minimum `ARKit` confidence (0=low,1=med,2=high) for a `LiDAR` point to be
    /// used at init.
    #[arg(long, help_heading = "Init options", default_value = "2")]
    pub lidar_min_confidence: u32,
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

    /// Whether any active loss needs the PGSR geometry render pass (blended
    /// normal + plane distance). Enabled on demand so the geometry cost is
    /// only paid when something consumes it.
    pub fn needs_geometry(&self) -> bool {
        self.geo_from_iter.is_some()
            && (self.depth_normal_weight > 0.0
                || self.depth_loss_weight > 0.0
                || self.distortion_weight > 0.0)
    }
}
