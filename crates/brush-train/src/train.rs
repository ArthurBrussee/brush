use anyhow::Result;
use brush_render::gaussian_splats::{inverse_sigmoid, Splats};
use brush_render::render::sh_coeffs_for_degree;
use brush_render::{AutodiffBackend, Backend, RenderAux};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use burn::lr_scheduler::exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig};
use burn::lr_scheduler::LrScheduler;
use burn::module::{Param, ParamId};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::record::AdaptorRecord;
use burn::optim::Optimizer;
use burn::tensor::{Bool, Distribution};
use burn::{config::Config, optim::GradientsParams, tensor::Tensor};
use hashbrown::HashMap;
use tracing::trace_span;

use crate::adam_scaled::{AdamScaled, AdamScaledConfig, AdamState};
use crate::scene::SceneView;
use crate::ssim::Ssim;
use crate::stats::RefineRecord;
use clap::Args;

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
    #[config(default = 1.6e-4)]
    #[arg(long, help_heading = "Training options", default_value = "1.6e-4")]
    lr_mean: f64,
    /// Learning rate decay for the mean lr.
    #[config(default = 1e-2)]
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    lr_mean_decay: f64,

    /// Learning rate for the basic coefficients.
    #[config(default = 2.5e-3)]
    #[arg(long, help_heading = "Training options", default_value = "2.5e-3")]
    lr_coeffs_dc: f64,
    /// How much to divide the learning rate by for higher SH orders.
    #[config(default = 20.0)]
    #[arg(long, help_heading = "Training options", default_value = "20.0")]
    lr_coeffs_sh_scale: f32,
    /// Learning rate for the opacity.
    #[config(default = 5e-2)]
    #[arg(long, help_heading = "Training options", default_value = "5e-2")]
    lr_opac: f64,
    /// Learning rate for the scale.
    #[config(default = 5e-3)]
    #[arg(long, help_heading = "Training options", default_value = "5e-3")]
    lr_scale: f64,
    /// Learning rate for the rotation.
    #[config(default = 1e-3)]
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    lr_rotation: f64,

    /// GSs with opacity below this value will be pruned
    #[config(default = 0.005)]
    #[arg(long, help_heading = "Refine options", default_value = "0.005")]
    cull_opacity: f32,
    /// Threshold for positional gradient norm
    #[config(default = 0.0002)]
    #[arg(long, help_heading = "Refine options", default_value = "0.0002")]
    densify_grad_thresh: f32,
    /// Gaussians bigger than this size in screenspace radius are split
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    densify_radius_threshold: f32,
    /// Below this size, gaussians are *duplicated*, otherwise split
    #[config(default = 0.01)]
    #[arg(long, help_heading = "Refine options", default_value = "0.01")]
    densify_size_threshold: f32,
    /// Gaussians bigger than this size in percent of the scene extent are culled
    #[config(default = 0.5)]
    #[arg(long, help_heading = "Refine options", default_value = "0.5")]
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
}

type B = Autodiff<Wgpu>;

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_images: Tensor<B, 4>,
    pub gt_views: Vec<SceneView>,
    pub scene_extent: f32,
}

#[derive(Clone)]
pub struct RefineStats {
    pub num_split: usize,
    pub num_cloned: usize,
    pub num_transparent_pruned: usize,
    pub num_scale_pruned: usize,
}

#[derive(Clone)]
pub struct TrainStepStats<B: AutodiffBackend> {
    pub pred_images: Tensor<B, 4>,
    pub gt_images: Tensor<B, 4>,
    pub gt_views: Vec<SceneView>,
    pub auxes: Vec<RenderAux<B>>,
    pub loss: Tensor<B, 1>,
    pub lr_mean: f64,
    pub lr_rotation: f64,
    pub lr_scale: f64,
    pub lr_coeffs: f64,
    pub lr_opac: f64,
}

type OptimizerType = OptimizerAdaptor<AdamScaled, Splats<B>, B>;

pub struct SplatTrainer {
    config: TrainConfig,
    sched_mean: ExponentialLrScheduler,
    optim: OptimizerType,
    ssim: Ssim<B>,
    refine_record: RefineRecord,
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

impl SplatTrainer {
    pub fn new(splats: &Splats<B>, config: &TrainConfig, device: &WgpuDevice) -> Self {
        let optim = AdamScaledConfig::new().with_epsilon(1e-15).init();
        let ssim = Ssim::new(config.ssim_window_size, 3, device);

        let decay = config.lr_mean_decay.powf(1.0 / config.total_steps as f64);
        let lr_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay);

        Self {
            config: config.clone(),
            sched_mean: lr_mean.init().expect("Lr schedule must be valid."),
            optim,
            refine_record: RefineRecord::new(splats.num_splats(), device),
            ssim,
        }
    }

    pub(crate) fn reset_opacity(
        &self,
        splats: &mut Splats<B>,
        record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    ) {
        map_param(
            &mut splats.raw_opacity,
            record,
            |op| Tensor::zeros_like(&op) + inverse_sigmoid(self.config.cull_opacity * 2.0),
            |state| Tensor::zeros_like(&state),
        );
    }

    pub async fn step(
        &mut self,
        iter: u32,
        batch: SceneBatch<B>,
        splats: Splats<B>,
    ) -> (Splats<B>, TrainStepStats<B>) {
        assert!(
            batch.gt_views.len() == 1,
            "Bigger batches aren't yet supported"
        );

        let mut splats = splats;

        let [batch_size, img_h, img_w, _] = batch.gt_images.dims();

        let (pred_images, auxes, loss) = {
            let mut renders = vec![];
            let mut auxes = vec![];

            for i in 0..batch.gt_views.len() {
                let camera = &batch.gt_views[i].camera;

                let (pred_image, aux) =
                    splats.render(camera, glam::uvec2(img_w as u32, img_h as u32), false);

                renders.push(pred_image);
                auxes.push(aux);
            }

            for aux in &auxes {
                aux.resolve_bwd_data().await;
            }

            let pred_images = Tensor::stack(renders, 0);

            let _span = trace_span!("Calculate losses", sync_burn = true).entered();

            // Convert to srgb space.
            let pred_rgb = pred_images
                .clone()
                .slice([0..batch_size, 0..img_h, 0..img_w, 0..3])
                .clamp_min(0.0);

            // This is wrong if the batch has mixed transparent and non-transparent images,
            // but that's ok for now.
            let pred_compare = if batch.gt_views[0].image.color().has_alpha() {
                pred_images.clone()
            } else {
                pred_rgb.clone()
            };

            let loss = (pred_compare - batch.gt_images.clone()).abs().mean();

            let loss = if self.config.ssim_weight > 0.0 {
                let gt_rgb =
                    batch
                        .gt_images
                        .clone()
                        .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);

                let ssim_loss = -self.ssim.ssim(pred_rgb, gt_rgb) + 1.0;
                loss * (1.0 - self.config.ssim_weight) + ssim_loss * self.config.ssim_weight
            } else {
                loss
            };

            (pred_images, auxes, loss)
        };

        let mut grads = trace_span!("Backward pass", sync_burn = true).in_scope(|| loss.backward());

        // TODO: Should scale lr be scales by scene scale as well?
        let (lr_mean, lr_rotation, lr_scale, lr_coeffs, lr_opac) = (
            self.sched_mean.step() * batch.scene_extent as f64,
            self.config.lr_rotation,
            self.config.lr_scale,
            self.config.lr_coeffs_dc,
            self.config.lr_opac,
        );

        trace_span!("Housekeeping", sync_burn = true).in_scope(|| {
            // TODO: Burn really should implement +=
            if iter > self.config.refine_start_iter {
                // Get the xy gradient norm from the dummy tensor.
                let xys_grad = splats
                    .xys_dummy
                    .grad_remove(&mut grads)
                    .expect("XY gradients need to be calculated.");

                let aux = auxes[0].clone();
                self.refine_record.gather_stats(xys_grad, aux);
            }
        });

        splats = trace_span!("Optimizer step", sync_burn = true).in_scope(|| {
            splats = trace_span!("Mean step", sync_burn = true).in_scope(|| {
                let grad_means =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.means.id]);
                self.optim.step(lr_mean, splats, grad_means)
            });

            splats = trace_span!("Opacity step", sync_burn = true).in_scope(|| {
                let grad_opac =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.raw_opacity.id]);
                self.optim.step(lr_opac, splats, grad_opac)
            });

            splats = trace_span!("SH Coeffs step", sync_burn = true).in_scope(|| {
                let grad_coeff =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);

                let coeff_count = sh_coeffs_for_degree(splats.sh_degree()) as i32;
                let sh_size = coeff_count;
                let mut sh_lr_scales = vec![1.0];
                for _ in 1..sh_size {
                    sh_lr_scales.push(1.0 / self.config.lr_coeffs_sh_scale);
                }
                let sh_lr_scales =
                    Tensor::<_, 1>::from_floats(sh_lr_scales.as_slice(), &splats.means.device())
                        .reshape([1, coeff_count, 1]);

                let mut record = self.optim.to_record();
                let mut param_record = record.get_mut(&splats.sh_coeffs.id);
                if let Some(param) = param_record.as_mut() {
                    let mut state = param.clone().into_state();
                    state.scaling = Some(sh_lr_scales);
                    record.insert(splats.sh_coeffs.id, AdaptorRecord::from_state(state));
                    self.optim = self.optim.clone().load_record(record);
                }

                self.optim.step(lr_coeffs, splats, grad_coeff)
            });

            splats = trace_span!("Rotation step", sync_burn = true).in_scope(|| {
                let grad_rot =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.rotation.id]);
                self.optim.step(lr_rotation, splats, grad_rot)
            });

            splats = trace_span!("Scale step", sync_burn = true).in_scope(|| {
                let grad_scale =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.log_scales.id]);
                self.optim.step(lr_scale, splats, grad_scale)
            });

            // Make sure rotations are still valid after optimization step.
            splats
        });

        let stats = TrainStepStats {
            pred_images,
            gt_images: batch.gt_images,
            gt_views: batch.gt_views,
            auxes,
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
        splats: Splats<B>,
        scene_extent: f32,
    ) -> (Splats<B>, Option<RefineStats>) {
        let do_refine = iter < self.config.refine_stop_iter
            && iter >= self.config.refine_start_iter
            && iter % self.config.refine_every == 1;

        if do_refine {
            // If not refining, update splat to step with gradients applied.
            let (refined_splats, refine) = self.refine_splats(iter, splats, scene_extent).await;
            (refined_splats, Some(refine))
        } else {
            (splats, None)
        }
    }

    async fn refine_splats(
        &mut self,
        iter: u32,
        splats: Splats<B>,
        scene_extent: f32,
    ) -> (Splats<B>, RefineStats) {
        let mut record = self.optim.to_record();

        let mut splats = splats;

        let device = splats.means.device();

        // Otherwise, do refinement, but do the split/clone on gaussians with no grads applied.
        let avg_grad = self.refine_record.average_grad_2d();

        let is_grad_high = avg_grad.greater_equal_elem(self.config.densify_grad_thresh);
        let split_clone_size_mask = splats
            .scales()
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
        let clone_count = clone_inds.dims()[0];
        if clone_count > 0 {
            let clone_inds = clone_inds.squeeze(1);
            let cur_means = splats.means.val().select(0, clone_inds.clone());
            let cur_rots = splats.rotation.val().select(0, clone_inds.clone());
            let cur_scale = splats.log_scales.val().select(0, clone_inds.clone());

            let cur_coeff = splats.sh_coeffs.val().select(0, clone_inds.clone());
            let cur_raw_opac = splats.raw_opacity.val().select(0, clone_inds);

            // let alpha = sigmoid(cur_raw_opac);
            // let new_alpha = -(-alpha + 1.0).sqrt() + 1.0;
            // let new_raw_opacity = inverse_sigmoid_tensor(new_alpha);
            let samples = quaternion_vec_multiply(
                cur_rots.clone(),
                Tensor::random([clone_count, 3], Distribution::Normal(0.0, 1.0), &device),
            ) * cur_scale.clone().exp();

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

        let radii_grow = self
            .refine_record
            .max_radii()
            .greater_elem(self.config.densify_radius_threshold);
        let split_mask = Tensor::stack::<2>(vec![split_mask, radii_grow], 1)
            .any_dim(1)
            .squeeze::<1>(1);

        let split_inds = split_mask.clone().argwhere_async().await;

        let split_count = split_inds.dims()[0];
        if split_count > 0 {
            let split_inds = split_inds.squeeze(1);

            // Some parts can be straightforwardly copied to the new splats.
            let cur_means = splats.means.val().select(0, split_inds.clone());
            let cur_coeff = splats.sh_coeffs.val().select(0, split_inds.clone());
            let cur_raw_opac = splats.raw_opacity.val().select(0, split_inds.clone());
            let cur_rots = splats.rotation.val().select(0, split_inds.clone());
            let cur_scale = splats.log_scales.val().select(0, split_inds);

            let samples = quaternion_vec_multiply(
                cur_rots.clone(),
                Tensor::random([split_count, 3], Distribution::Normal(0.0, 1.0), &device),
            ) * cur_scale.clone().exp();

            append_means.push(cur_means.clone() + samples.clone());
            append_rots.push(cur_rots.clone());
            append_scales.push(cur_scale.clone() - 1.6f32.ln());
            append_coeffs.push(cur_coeff.clone());
            append_opac.push(cur_raw_opac.clone());

            append_means.push(cur_means - samples);
            append_rots.push(cur_rots);
            append_scales.push(cur_scale - 1.6f32.ln());
            append_coeffs.push(cur_coeff);
            append_opac.push(cur_raw_opac);
        }

        prune_points(&mut splats, &mut record, split_mask.clone()).await;

        // Do some more processing. Important to do this last as otherwise you might mess up the correspondence
        // of gradient <-> splat.
        let start_count = splats.num_splats();
        // Remove barely visible gaussians.
        let alpha_mask = splats.opacity().lower_elem(self.config.cull_opacity);
        prune_points(&mut splats, &mut record, alpha_mask).await;
        let alpha_pruned = start_count - splats.num_splats();

        // Delete Gaussians with too large of a radius in world-units.
        let scale_big = splats
            .log_scales
            .val()
            .greater_elem((self.config.cull_scale3d_percentage_threshold * scene_extent).ln());

        // less than e^-10, too small to care about.
        let scale_small = splats.log_scales.val().lower_elem(-10.0);

        let scale_mask =
            Tensor::any_dim(Tensor::cat(vec![scale_small, scale_big], 1), 1).squeeze(1);
        prune_points(&mut splats, &mut record, scale_mask).await;
        let scale_pruned = start_count - splats.num_splats();

        if !append_means.is_empty() {
            let append_means = Tensor::cat(append_means, 0);
            let append_rots = Tensor::cat(append_rots, 0);
            let append_coeffs = Tensor::cat(append_coeffs, 0);
            let append_opac = Tensor::cat(append_opac, 0);
            let append_scales = Tensor::cat(append_scales, 0);

            concat_splats(
                &mut splats,
                &mut record,
                append_means,
                append_rots,
                append_coeffs,
                append_opac,
                append_scales,
            );
        }

        let refine_step = iter / self.config.refine_every;
        if refine_step % self.config.reset_alpha_every_refine == 0 {
            self.reset_opacity(&mut splats, &mut record);
        }

        // Stats don't line up anymore so have to reset them.
        self.refine_record = RefineRecord::new(splats.num_splats(), &device);
        self.optim = self.optim.clone().load_record(record);

        let stats = RefineStats {
            num_split: split_count,
            num_cloned: clone_count,
            num_transparent_pruned: alpha_pruned,
            num_scale_pruned: scale_pruned,
        };

        (splats, stats)
    }
}

fn map_param<B: AutodiffBackend, const D: usize>(
    param: &mut Param<Tensor<B, D>>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    map_param: impl FnOnce(Tensor<B, D>) -> Tensor<B, D>,
    map_opt: impl Fn(Tensor<B::InnerBackend, D>) -> Tensor<B::InnerBackend, D>,
) {
    Splats::map_param(param, map_param);
    let mut state: AdamState<_, D> = record[&param.id].clone().into_state();
    state.momentum.moment_1 = map_opt(state.momentum.moment_1);
    state.momentum.moment_2 = map_opt(state.momentum.moment_2);
    record.insert(param.id, AdaptorRecord::from_state(state));
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
pub async fn prune_points<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    prune: Tensor<B, 1, Bool>,
) {
    assert_eq!(
        prune.dims()[0],
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    // bool[n]. If True, delete these Gaussians.
    let prune_count = prune.dims()[0];

    if prune_count == 0 {
        return;
    }

    let valid_inds = prune.bool_not().argwhere_async().await.squeeze(1);
    let start_splats = splats.num_splats();
    let new_points = valid_inds.dims()[0];

    if new_points < start_splats {
        map_param(
            &mut splats.means,
            record,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone().inner()),
        );
        map_param(
            &mut splats.sh_coeffs,
            record,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone().inner()),
        );
        map_param(
            &mut splats.rotation,
            record,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone().inner()),
        );
        map_param(
            &mut splats.raw_opacity,
            record,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone().inner()),
        );
        map_param(
            &mut splats.log_scales,
            record,
            |x| x.select(0, valid_inds.clone()),
            |x| x.select(0, valid_inds.clone().inner()),
        );
    }
}

pub fn concat_splats<B: AutodiffBackend>(
    splats: &mut Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    means: Tensor<B, 2>,
    rotations: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 3>,
    raw_opac: Tensor<B, 1>,
    log_scales: Tensor<B, 2>,
) {
    // Concat
    let means_shape = means.shape();
    let device = means.device();
    map_param(
        &mut splats.means,
        record,
        move |x| Tensor::cat(vec![x, means], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros(means_shape.clone(), &device)], 0),
    );

    let rotations_shape = rotations.shape();
    map_param(
        &mut splats.rotation,
        record,
        move |x| Tensor::cat(vec![x, rotations], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros(rotations_shape.clone(), &device)], 0),
    );

    let sh_coeffs_shape = sh_coeffs.shape();
    map_param(
        &mut splats.sh_coeffs,
        record,
        move |x| Tensor::cat(vec![x, sh_coeffs], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros(sh_coeffs_shape.clone(), &device)], 0),
    );
    let raw_opac_shape = raw_opac.shape();
    map_param(
        &mut splats.raw_opacity,
        record,
        move |x| Tensor::cat(vec![x, raw_opac], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros(raw_opac_shape.clone(), &device)], 0),
    );
    let log_scales_shape = log_scales.shape();
    map_param(
        &mut splats.log_scales,
        record,
        move |x| Tensor::cat(vec![x, log_scales], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros(log_scales_shape.clone(), &device)], 0),
    );
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::{wgpu::WgpuDevice, Wgpu},
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
