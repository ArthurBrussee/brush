use anyhow::Result;
use brush_render::gaussian_splats::{Splats, inverse_sigmoid};
use brush_render::render::sh_coeffs_for_degree;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::lr_scheduler::LrScheduler;
use burn::lr_scheduler::exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig};
use burn::module::ParamId;
use burn::optim::Optimizer;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::record::AdaptorRecord;
use burn::prelude::Backend;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, Distribution, Int, TensorPrimitive};
use burn::{config::Config, optim::GradientsParams, tensor::Tensor};
use hashbrown::HashMap;
use tracing::trace_span;

use crate::adam_scaled::{AdamScaled, AdamScaledConfig, AdamState};
use crate::burn_glue::SplatForwardDiff;
use crate::scene::{SceneView, ViewImageType};
use crate::ssim::Ssim;
use crate::stats::RefineRecord;
use clap::Args;

const MIN_OPACITY: f32 = 0.99 / 255.0;

#[derive(Config, Args)]
pub struct TrainConfig {
    /// Total number of steps to train for.
    #[config(default = 30000)]
    #[arg(long, help_heading = "Training options", default_value = "30000")]
    pub total_steps: u32,

    /// Weight of SSIM loss (compared to l1 loss)
    #[config(default = 0.2)]
    #[clap(long, help_heading = "Training options", default_value = "0.2")]
    ssim_weight: f32,

    /// SSIM window size
    #[config(default = 11)]
    #[clap(long, help_heading = "Training options", default_value = "11")]
    ssim_window_size: usize,

    /// Start learning rate for the mean.
    #[config(default = 5e-5)]
    #[arg(long, help_heading = "Training options", default_value = "5e-5")]
    lr_mean: f64,

    /// Start learning rate for the mean.
    #[config(default = 1e-6)]
    #[arg(long, help_heading = "Training options", default_value = "1e-6")]
    lr_mean_end: f64,

    /// Learning rate for the basic coefficients.
    #[config(default = 2e-3)]
    #[arg(long, help_heading = "Training options", default_value = "2e-3")]
    lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[config(default = 20.0)]
    #[arg(long, help_heading = "Training options", default_value = "20.0")]
    lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity.
    #[config(default = 3e-2)]
    #[arg(long, help_heading = "Training options", default_value = "3e-2")]
    lr_opac: f64,

    /// Learning rate for the scale.
    #[config(default = 5e-3)]
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    lr_scale: f64,

    /// Learning rate for the rotation.
    #[config(default = 1e-3)]
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    lr_rotation: f64,

    /// Weight of mean-opacity loss.
    #[config(default = 0.0)]
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    opac_loss_weight: f32,

    /// How much opacity to subtrat every refine step.
    #[config(default = 0.002)]
    #[arg(long, help_heading = "Training options", default_value = "0.002")]
    opac_refine_subtract: f32,

    /// Threshold for positional gradient norm
    #[config(default = 0.0008)]
    #[arg(long, help_heading = "Refine options", default_value = "0.0008")]
    densify_grad_thresh: f32,

    /// Gaussians bigger than this size in screenspace radius are split
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    densify_radius_threshold: f32,

    /// Below this size, gaussians are cloned, otherwise split
    #[config(default = 0.01)]
    #[arg(long, help_heading = "Refine options", default_value = "0.01")]
    densify_size_threshold: f32,

    /// Gaussians bigger than this size in percent of the scene extent are culled
    #[config(default = 0.8)]
    #[arg(long, help_heading = "Refine options", default_value = "0.8")]
    cull_scale3d_percentage_threshold: f32,

    /// Period before refinement starts.
    #[config(default = 500)]
    #[arg(long, help_heading = "Refine options", default_value = "500")]
    refine_start_iter: u32,

    /// Period after which refinement stops.
    #[config(default = 15000)]
    #[arg(long, help_heading = "Refine options", default_value = "15000")]
    refine_stop_iter: u32,

    /// Every this many refinement steps, reset the alpha
    #[config(default = 30)]
    #[arg(long, help_heading = "Refine options", default_value = "30")]
    reset_alpha_every_refine: u32,

    /// Period of steps where gaussians are culled and densified
    #[config(default = 100)]
    #[arg(long, help_heading = "Refine options", default_value = "100")]
    refine_every: u32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    match_alpha_weight: f32,
}

pub type TrainBack = Autodiff<Wgpu>;
// pub type TrainBack = Autodiff<Vulkan>;

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_image: Tensor<B, 3>,
    pub gt_view: SceneView,
}

#[derive(Clone)]
pub struct RefineStats {
    pub num_split: u32,
    pub num_cloned: u32,
    pub num_transparent_pruned: u32,
    pub num_scale_pruned: u32,
}

#[derive(Clone)]
pub struct TrainStepStats<B: Backend> {
    pub pred_image: Tensor<B, 3>,

    pub gt_views: SceneView,

    pub num_intersections: Tensor<B, 1, Int>,
    pub num_visible: Tensor<B, 1, Int>,
    pub loss: Tensor<B, 1>,

    pub lr_mean: f64,
    pub lr_rotation: f64,
    pub lr_scale: f64,
    pub lr_coeffs: f64,
    pub lr_opac: f64,
}

type OptimizerType = OptimizerAdaptor<AdamScaled, Splats<TrainBack>, TrainBack>;

pub struct SplatTrainer {
    config: TrainConfig,
    sched_mean: ExponentialLrScheduler,
    ssim: Ssim<TrainBack>,

    optim: Option<OptimizerType>,
    refine_record: Option<RefineRecord<<TrainBack as AutodiffBackend>::InnerBackend>>,
}

fn quaternion_vec_multiply<B: Backend>(
    quaternions: Tensor<B, 2>,
    vectors: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let num_points = quaternions.dims()[0];

    // Extract components
    let qw = quaternions.clone().slice([0..num_points, 0..1]);
    let qx = quaternions.clone().slice([0..num_points, 1..2]);
    let qy = quaternions.clone().slice([0..num_points, 2..3]);
    let qz = quaternions.slice([0..num_points, 3..4]);

    let vx = vectors.clone().slice([0..num_points, 0..1]);
    let vy = vectors.clone().slice([0..num_points, 1..2]);
    let vz = vectors.slice([0..num_points, 2..3]);

    // Common terms
    let qw2 = qw.clone().powf_scalar(2.0);
    let qx2 = qx.clone().powf_scalar(2.0);
    let qy2 = qy.clone().powf_scalar(2.0);
    let qz2 = qz.clone().powf_scalar(2.0);

    // Cross products (multiplied by 2.0 later)
    let xy = qx.clone() * qy.clone();
    let xz = qx.clone() * qz.clone();
    let yz = qy.clone() * qz.clone();
    let wx = qw.clone() * qx;
    let wy = qw.clone() * qy;
    let wz = qw * qz;

    // Final components with reused terms
    let x = (qw2.clone() + qx2.clone() - qy2.clone() - qz2.clone()) * vx.clone()
        + (xy.clone() * vy.clone() + xz.clone() * vz.clone() + wy.clone() * vz.clone()
            - wz.clone() * vy.clone())
            * 2.0;

    let y = (qw2.clone() - qx2.clone() + qy2.clone() - qz2.clone()) * vy.clone()
        + (xy * vx.clone() + yz.clone() * vz.clone() + wz * vx.clone() - wx.clone() * vz.clone())
            * 2.0;

    let z = (qw2 - qx2 - qy2 + qz2) * vz
        + (xz * vx.clone() + yz * vy.clone() + wx * vy - wy * vx) * 2.0;

    Tensor::cat(vec![x, y, z], 1)
}

pub fn inv_sigmoid<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    (x.clone() / (-x + 1.0)).log()
}

fn create_default_optimizer() -> OptimizerType {
    AdamScaledConfig::new().with_epsilon(1e-15).init()
}

impl SplatTrainer {
    pub fn new(config: &TrainConfig, device: &WgpuDevice) -> Self {
        let ssim = Ssim::new(config.ssim_window_size, 3, device);

        let decay = (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_steps as f64);
        let lr_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay);

        Self {
            config: config.clone(),
            sched_mean: lr_mean.init().expect("Lr schedule must be valid."),
            optim: None,
            refine_record: None,
            ssim,
        }
    }

    pub fn step(
        &mut self,
        scene_extent: f32,
        iter: u32,
        batch: SceneBatch<TrainBack>,
        splats: Splats<TrainBack>,
    ) -> (Splats<TrainBack>, TrainStepStats<TrainBack>) {
        let mut splats = splats;

        let [img_h, img_w, _] = batch.gt_image.dims();

        let camera = &batch.gt_view.camera;

        let (pred_image, aux, refine_weight_holder) = {
            let diff_out = <TrainBack as SplatForwardDiff<TrainBack>>::render_splats(
                camera,
                glam::uvec2(img_w as u32, img_h as u32),
                splats.means.val().into_primitive().tensor(),
                splats.log_scales.val().into_primitive().tensor(),
                splats.rotation.val().into_primitive().tensor(),
                splats.sh_coeffs.val().into_primitive().tensor(),
                splats.raw_opacity.val().into_primitive().tensor(),
            );
            let img = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
            let wrapped_aux = diff_out.aux.into_wrapped();
            (img, wrapped_aux, diff_out.refine_weight_holder)
        };

        let _span = trace_span!("Calculate losses", sync_burn = true).entered();

        let pred_rgb = pred_image.clone().slice([0..img_h, 0..img_w, 0..3]);
        let gt_rgb = batch.gt_image.clone().slice([0..img_h, 0..img_w, 0..3]);

        let l1_rgb = (pred_rgb.clone() - gt_rgb).abs();

        let total_err = if self.config.ssim_weight > 0.0 {
            let gt_rgb = batch.gt_image.clone().slice([0..img_h, 0..img_w, 0..3]);

            let ssim_err = -self.ssim.ssim(pred_rgb, gt_rgb);
            l1_rgb * (1.0 - self.config.ssim_weight) + ssim_err * self.config.ssim_weight
        } else {
            l1_rgb
        };

        let mut loss = if batch.gt_view.image.color().has_alpha() {
            let alpha_input = batch.gt_image.clone().slice([0..img_h, 0..img_w, 3..4]);

            match batch.gt_view.img_type {
                // In masked mode, weigh the errors by the alpha channel.
                ViewImageType::Masked => (total_err * alpha_input).mean(),
                // In alpha mode, add the l1 error of the alpha channel to the total error.
                ViewImageType::Alpha => {
                    let pred_alpha = pred_image.clone().slice([0..img_h, 0..img_w, 3..4]);
                    total_err.mean()
                        + (alpha_input - pred_alpha).abs().mean() * self.config.match_alpha_weight
                }
            }
        } else {
            total_err.mean()
        };

        // Add in opacity loss if enabled.
        if self.config.opac_loss_weight > 0.0 {
            let opac_loss = splats.opacity().mean();
            loss = loss + opac_loss * self.config.opac_loss_weight;
        }

        let mut grads = trace_span!("Backward pass", sync_burn = true).in_scope(|| loss.backward());

        let (lr_mean, lr_rotation, lr_scale, lr_coeffs, lr_opac) = (
            self.sched_mean.step() * scene_extent as f64,
            self.config.lr_rotation,
            // Scale is relative to the scene scale, but the exp() activation function
            // means "offsetting" all values also solves the learning rate scaling.
            self.config.lr_scale,
            self.config.lr_coeffs_dc,
            self.config.lr_opac,
        );

        let optimizer = self.optim.get_or_insert_with(|| {
            let sh_degree = splats.sh_degree();
            let device = splats.device();

            let coeff_count = sh_coeffs_for_degree(sh_degree) as i32;
            let sh_size = coeff_count;
            let mut sh_lr_scales = vec![1.0];
            for _ in 1..sh_size {
                sh_lr_scales.push(1.0 / self.config.lr_coeffs_sh_scale);
            }
            let sh_lr_scales = Tensor::<_, 1>::from_floats(sh_lr_scales.as_slice(), &device)
                .reshape([1, coeff_count, 1]);

            create_default_optimizer().load_record(HashMap::from([(
                splats.sh_coeffs.id,
                AdaptorRecord::from_state(AdamState {
                    momentum: None,
                    scaling: Some(sh_lr_scales),
                }),
            )]))
        });

        splats = trace_span!("Optimizer step", sync_burn = true).in_scope(|| {
            splats = trace_span!("SH Coeffs step", sync_burn = true).in_scope(|| {
                let grad_coeff =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);
                optimizer.step(lr_coeffs, splats, grad_coeff)
            });

            splats = trace_span!("Rotation step", sync_burn = true).in_scope(|| {
                let grad_rot =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.rotation.id]);
                optimizer.step(lr_rotation, splats, grad_rot)
            });

            splats = trace_span!("Scale step", sync_burn = true).in_scope(|| {
                let grad_scale =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.log_scales.id]);
                optimizer.step(lr_scale, splats, grad_scale)
            });

            splats = trace_span!("Mean step", sync_burn = true).in_scope(|| {
                let grad_means =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.means.id]);
                optimizer.step(lr_mean, splats, grad_means)
            });

            splats = trace_span!("Opacity step", sync_burn = true).in_scope(|| {
                let grad_opac =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.raw_opacity.id]);
                optimizer.step(lr_opac, splats, grad_opac)
            });

            // Make sure rotations are still valid after optimization step.
            splats
        });

        let num_visible = aux.num_visible.clone();
        let num_intersections = aux.num_intersections.clone();

        trace_span!("Housekeeping", sync_burn = true).in_scope(|| {
            let start_collect_iter = self
                .config
                .refine_start_iter
                .saturating_sub(self.config.refine_every);

            if iter > start_collect_iter {
                // Get the xy gradient norm from the dummy tensor.
                let refine_weight = refine_weight_holder
                    .grad_remove(&mut grads)
                    .expect("XY gradients need to be calculated.");
                let aux = aux.clone();

                let device = splats.device();
                let num_splats = splats.num_splats();
                let record = self
                    .refine_record
                    .get_or_insert_with(|| RefineRecord::new(num_splats, &device));
                record.gather_stats(refine_weight, aux);
            }
        });

        let stats = TrainStepStats {
            pred_image,
            gt_views: batch.gt_view,
            num_visible,
            num_intersections,
            loss,
            lr_mean,
            lr_rotation,
            lr_scale,
            lr_coeffs,
            lr_opac,
        };

        (splats, stats)
    }

    pub async fn refine_if_needed(
        &mut self,
        iter: u32,
        splats: Splats<TrainBack>,
        scene_extent: f32,
    ) -> (Splats<TrainBack>, Option<RefineStats>) {
        if iter > 0 && iter % self.config.refine_every == 0 {
            // Normalize rotations to prevent them from slowly drifting towards 0. When they
            // get to 0 they are effectively killed off.
            // This is slightly wrong wrt to adam gradients, but that's fine.
            let splats = splats.with_normed_rotations();

            // If not refining, update splat to step with gradients applied.
            if iter >= self.config.refine_start_iter && iter < self.config.refine_stop_iter {
                let (splats, refine) = self.refine_splats(iter, splats, scene_extent).await;
                (splats, Some(refine))
            } else {
                (splats, None)
            }
        } else {
            (splats, None)
        }
    }

    async fn refine_splats(
        &mut self,
        iter: u32,
        splats: Splats<TrainBack>,
        scene_extent: f32,
    ) -> (Splats<TrainBack>, RefineStats) {
        let mut record = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized")
            .to_record();

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refin if refin stats are initialized");

        // Otherwise, do refinement, but do the split/clone on gaussians with no grads applied.
        let avg_grad = refiner.refine_weight_norm / refiner.visible_counts.clamp_min(1).float();

        let mut splats = splats;

        let device = splats.means.device();

        let is_grad_high = avg_grad.greater_equal_elem(self.config.densify_grad_thresh);
        let split_clone_size_mask = splats
            .scales()
            .inner()
            .max_dim(1)
            .squeeze(1)
            .lower_elem(self.config.densify_size_threshold * scene_extent);

        let mut append_means = vec![];
        let mut append_rots = vec![];
        let mut append_coeffs = vec![];
        let mut append_opac = vec![];
        let mut append_scales = vec![];

        let clone_mask =
            Tensor::stack::<2>(vec![is_grad_high.clone(), split_clone_size_mask.clone()], 1)
                .all_dim(1)
                .squeeze::<1>(1);

        let clone_inds = clone_mask.clone().argwhere_async().await;

        // Clone splats
        let clone_count = clone_inds.dims()[0] as u32;
        if clone_count > 0 {
            let clone_inds = clone_inds.squeeze(1);
            let cur_means = splats.means.val().inner().select(0, clone_inds.clone());
            let cur_rots = splats.rotation.val().inner().select(0, clone_inds.clone());
            let cur_scale = splats
                .log_scales
                .val()
                .inner()
                .select(0, clone_inds.clone());
            let cur_coeff = splats.sh_coeffs.val().inner().select(0, clone_inds.clone());
            let cur_raw_opac = splats.raw_opacity.val().inner().select(0, clone_inds);

            let samples = quaternion_vec_multiply(
                cur_rots.clone(),
                Tensor::random(
                    [clone_count as usize, 3],
                    Distribution::Normal(0.0, 1.0),
                    &device,
                ) * cur_scale.clone().exp(),
            );

            append_means.push(cur_means + samples);
            append_rots.push(cur_rots);
            append_scales.push(cur_scale);
            append_coeffs.push(cur_coeff);
            append_opac.push(cur_raw_opac);
        }

        // Split splats.
        let split_mask = Tensor::stack::<2>(
            vec![is_grad_high.clone(), split_clone_size_mask.bool_not()],
            1,
        )
        .all_dim(1)
        .squeeze::<1>(1);

        let radii_grow = refiner
            .max_radii
            .greater_elem(self.config.densify_radius_threshold);

        let split_mask = Tensor::stack::<2>(vec![split_mask, radii_grow], 1)
            .any_dim(1)
            .squeeze::<1>(1);

        let split_inds = split_mask.clone().argwhere_async().await;

        let split_count = split_inds.dims()[0] as u32;
        if split_count > 0 {
            let split_inds = split_inds.squeeze(1);

            // Some parts can be straightforwardly copied to the new splats.
            let cur_means = splats.means.val().inner().select(0, split_inds.clone());
            let cur_coeff = splats.sh_coeffs.val().inner().select(0, split_inds.clone());
            let cur_raw_opac = splats
                .raw_opacity
                .val()
                .inner()
                .select(0, split_inds.clone());
            let cur_rots = splats.rotation.val().inner().select(0, split_inds.clone());
            let cur_scale = splats.log_scales.val().inner().select(0, split_inds);

            let samples = quaternion_vec_multiply(
                cur_rots.clone(),
                Tensor::random(
                    [split_count as usize, 3],
                    Distribution::Normal(0.0, 1.0),
                    &device,
                ) * cur_scale.clone().exp(),
            );

            let scale_div: f32 = 1.6;

            append_means.push(cur_means.clone() + samples.clone());
            append_rots.push(cur_rots.clone());
            append_scales.push(cur_scale.clone() - scale_div.ln());
            append_coeffs.push(cur_coeff.clone());
            append_opac.push(cur_raw_opac.clone());

            append_means.push(cur_means - samples);
            append_rots.push(cur_rots);
            append_scales.push(cur_scale - scale_div.ln());
            append_coeffs.push(cur_coeff);
            append_opac.push(cur_raw_opac);
        }

        (splats, _) = prune_points(splats, &mut record, split_mask.clone()).await;

        // Do some more processing. Important to do this last as otherwise you might mess up the correspondence
        // of gradient <-> splat.

        // Remove barely visible gaussians.
        let alpha_mask = splats
            .raw_opacity
            .val()
            .inner()
            .lower_elem(inverse_sigmoid(MIN_OPACITY));
        let (splats, alpha_pruned) = prune_points(splats, &mut record, alpha_mask).await;

        // Delete Gaussians with too large of a radius in world-units.
        let scale_big = splats
            .log_scales
            .val()
            .inner()
            .greater_elem((self.config.cull_scale3d_percentage_threshold * scene_extent).ln());

        let scale_mask = Tensor::any_dim(scale_big, 1).squeeze(1);
        let (mut splats, scale_pruned) = prune_points(splats, &mut record, scale_mask).await;

        if !append_means.is_empty() {
            let append_means = Tensor::cat(append_means, 0);
            let append_rots = Tensor::cat(append_rots, 0);
            let append_coeffs = Tensor::cat(append_coeffs, 0);
            let append_opac = Tensor::cat(append_opac, 0);
            let append_scales = Tensor::cat(append_scales, 0);

            splats = concat_splats(
                splats,
                &mut record,
                append_means,
                append_rots,
                append_scales,
                append_coeffs,
                append_opac,
            );
        }

        let refine_step = iter / self.config.refine_every;
        if refine_step % self.config.reset_alpha_every_refine == 0 {
            splats.raw_opacity = splats
                .raw_opacity
                .map(|op| op.clamp_max(inverse_sigmoid(0.01)));
            map_opt::<_, 1>(splats.raw_opacity.id, &mut record, &|s| {
                Tensor::zeros_like(&s)
            });
        } else {
            // Skip a refine after every reset.
            let time_per_reset = self.config.reset_alpha_every_refine * self.config.refine_every;
            let time_since_reset = iter % time_per_reset;

            // Slowly lower opacity.
            if self.config.opac_refine_subtract > 0.0 && time_since_reset > self.config.refine_every
            {
                splats.raw_opacity = splats.raw_opacity.map(|op| {
                    let op = op.inner();
                    Tensor::from_inner(inv_sigmoid(
                        (sigmoid(op) - self.config.opac_refine_subtract).clamp_min(1e-3),
                    ))
                    .require_grad()
                });
            }
        }

        // Stats don't line up anymore so have to reset them.
        self.optim = Some(create_default_optimizer().load_record(record));

        let stats = RefineStats {
            num_split: split_count,
            num_cloned: clone_count,
            num_transparent_pruned: alpha_pruned,
            num_scale_pruned: scale_pruned,
        };

        (splats, stats)
    }
}

fn map_splats_and_opt<B: AutodiffBackend>(
    splats: Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    map_mean: impl FnOnce(Tensor<B::InnerBackend, 2>) -> Tensor<B::InnerBackend, 2>,
    map_rotation: impl FnOnce(Tensor<B::InnerBackend, 2>) -> Tensor<B::InnerBackend, 2>,
    map_scale: impl FnOnce(Tensor<B::InnerBackend, 2>) -> Tensor<B::InnerBackend, 2>,
    map_coeffs: impl FnOnce(Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 3>,
    map_opac: impl FnOnce(Tensor<B::InnerBackend, 1>) -> Tensor<B::InnerBackend, 1>,

    map_opt_mean: impl Fn(Tensor<B::InnerBackend, 2>) -> Tensor<B::InnerBackend, 2>,
    map_opt_rotation: impl Fn(Tensor<B::InnerBackend, 2>) -> Tensor<B::InnerBackend, 2>,
    map_opt_scale: impl Fn(Tensor<B::InnerBackend, 2>) -> Tensor<B::InnerBackend, 2>,
    map_opt_coeffs: impl Fn(Tensor<B::InnerBackend, 3>) -> Tensor<B::InnerBackend, 3>,
    map_opt_opac: impl Fn(Tensor<B::InnerBackend, 1>) -> Tensor<B::InnerBackend, 1>,
) -> Splats<B> {
    map_opt(splats.means.id, record, &map_opt_mean);
    map_opt(splats.rotation.id, record, &map_opt_rotation);
    map_opt(splats.log_scales.id, record, &map_opt_scale);
    map_opt(splats.sh_coeffs.id, record, &map_opt_coeffs);
    map_opt(splats.raw_opacity.id, record, &map_opt_opac);

    Splats {
        means: splats
            .means
            .map(|m| Tensor::from_inner(map_mean(m.inner())).require_grad()),
        rotation: splats
            .rotation
            .map(|m| Tensor::from_inner(map_rotation(m.inner())).require_grad()),
        log_scales: splats
            .log_scales
            .map(|m| Tensor::from_inner(map_scale(m.inner())).require_grad()),
        sh_coeffs: splats
            .sh_coeffs
            .map(|m| Tensor::from_inner(map_coeffs(m.inner())).require_grad()),
        raw_opacity: splats
            .raw_opacity
            .map(|m| Tensor::from_inner(map_opac(m.inner())).require_grad()),
    }
}

fn map_opt<B: AutodiffBackend, const D: usize>(
    param_id: ParamId,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    map_opt: &impl Fn(Tensor<B::InnerBackend, D>) -> Tensor<B::InnerBackend, D>,
) {
    let mut state: AdamState<_, D> = record
        .remove(&param_id)
        .expect("failed to get optimizer record")
        .into_state();

    state.momentum = state.momentum.map(|mut moment| {
        moment.moment_1 = map_opt(moment.moment_1);
        moment.moment_2 = map_opt(moment.moment_2);
        moment
    });

    record.insert(param_id, AdaptorRecord::from_state(state));
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
pub async fn prune_points<B: AutodiffBackend>(
    mut splats: Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    prune: Tensor<B::InnerBackend, 1, Bool>,
) -> (Splats<B>, u32) {
    assert_eq!(
        prune.dims()[0] as u32,
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    // bool[n]. If True, delete these Gaussians.
    let prune_count = prune.dims()[0];

    if prune_count == 0 {
        return (splats, 0);
    }

    let valid_inds = prune.bool_not().argwhere_async().await;

    if valid_inds.dims()[0] == 0 {
        log::warn!("Trying to create empty splat!");
        return (splats, 0);
    }

    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0] as u32;

    if new_points < start_splats {
        let valid_inds = valid_inds.squeeze(1);
        splats = map_splats_and_opt(
            splats,
            record,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone()),
        );
    }

    (splats, start_splats - new_points)
}

pub fn concat_splats<B: AutodiffBackend>(
    splats: Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    means: Tensor<B::InnerBackend, 2>,
    rotations: Tensor<B::InnerBackend, 2>,
    log_scales: Tensor<B::InnerBackend, 2>,
    sh_coeffs: Tensor<B::InnerBackend, 3>,
    raw_opac: Tensor<B::InnerBackend, 1>,
) -> Splats<B> {
    let device = splats.means.device();

    let cur_count = splats.means.dims()[0];
    let append_count = means.dims()[0];
    let sh_dim = splats.sh_coeffs.dims()[1];

    map_splats_and_opt(
        splats,
        record,
        |x| Tensor::cat(vec![x, means], 0),
        |x| Tensor::cat(vec![x, rotations], 0),
        |x| Tensor::cat(vec![x, log_scales], 0),
        |x| Tensor::cat(vec![x, sh_coeffs], 0),
        |x| Tensor::cat(vec![x, raw_opac], 0),
        |x| {
            Tensor::zeros([cur_count + append_count, 3], &device)
                .slice_assign([0..cur_count, 0..3], x)
        },
        |x| {
            Tensor::zeros([cur_count + append_count, 4], &device)
                .slice_assign([0..cur_count, 0..4], x)
        },
        |x| {
            Tensor::zeros([cur_count + append_count, 3], &device)
                .slice_assign([0..cur_count, 0..3], x)
        },
        |x| {
            Tensor::zeros([cur_count + append_count, sh_dim, 3], &device)
                .slice_assign([0..cur_count, 0..sh_dim, 0..3], x)
        },
        |x| Tensor::zeros([cur_count + append_count], &device).slice_assign([0..cur_count], x),
    )
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{Wgpu, wgpu::WgpuDevice},
        tensor::Tensor,
    };
    use glam::Quat;

    use super::quaternion_vec_multiply;

    #[test]
    fn test_quat_multiply() {
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.2, 0.2, 0.3);
        let vec = glam::vec3(0.5, 0.7, 0.1);
        let result_ref = quat * vec;

        let device = WgpuDevice::DefaultDevice;
        let quaternions = Tensor::<Wgpu, 1>::from_floats([quat.w, quat.x, quat.y, quat.z], &device)
            .reshape([1, 4]);
        let vecs = Tensor::<Wgpu, 1>::from_floats([vec.x, vec.y, vec.z], &device).reshape([1, 3]);
        let result = quaternion_vec_multiply(quaternions, vecs);
        let result: Vec<f32> = result.into_data().to_vec().expect("Wrong type");
        let result = glam::vec3(result[0], result[1], result[2]);
        assert!((result_ref - result).length() < 1e-7);
    }
}
