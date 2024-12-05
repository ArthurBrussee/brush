use anyhow::Result;
use brush_render::gaussian_splats::{inverse_sigmoid, Splats};
use brush_render::{AutodiffBackend, Backend, RenderAux};
use burn::lr_scheduler::exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig};
use burn::lr_scheduler::LrScheduler;
use burn::module::{Param, ParamId};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::record::AdaptorRecord;
use burn::optim::{Adam, AdamState};
use burn::tensor::{Bool, Distribution, Int, TensorPrimitive};
use burn::{
    config::Config,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::Tensor,
};
use hashbrown::HashMap;
use tracing::trace_span;

use crate::scene::SceneView;
use crate::ssim::Ssim;

type OptimizerType<B> = OptimizerAdaptor<Adam, Splats<B>, B>;

#[derive(Config)]
pub struct TrainConfig {
    // Weight for SSIM loss
    #[config(default = 0.2)]
    ssim_weight: f32,

    // GSs with opacity below this value will be pruned
    #[config(default = 0.005)]
    cull_opacity: f32,

    // threshold of positional gradient norm for densifying gaussians
    #[config(default = 0.0002)]
    densify_grad_thresh: f32,

    // Gaussians bigger than this size in screenspace radius are split.
    // Set to 1.0 to disable.
    #[config(default = 1.0)]
    densify_radius_threshold: f32,

    // below this size, gaussians are *duplicated*, otherwise split.
    #[config(default = 0.01)]
    densify_size_threshold: f32,

    // threshold of scale for culling huge gaussians
    #[config(default = 0.5)]
    cull_scale3d_percentage_threshold: f32,

    // period of steps where refinement is turned off
    #[config(default = 500)]
    refine_start_iter: u32,

    #[config(default = 15000)]
    refine_stop_iter: u32,

    // Every this many refinement steps, reset the alpha
    #[config(default = 30)]
    reset_alpha_every_refine: u32,
    // period of steps where gaussians are culled and densified
    #[config(default = 100)]
    refine_every: u32,

    #[config(default = 11)]
    ssim_window_size: usize,

    // Learning rates.
    lr_mean: ExponentialLrSchedulerConfig,

    // Learning rate for the basic coefficients.
    #[config(default = 2.5e-3)]
    lr_coeffs_dc: f64,

    // How much to divide the learning rate by for higher SH orders.
    #[config(default = 20.0)]
    lr_coeffs_sh_scale: f64,

    #[config(default = 5e-2)]
    lr_opac: f64,

    #[config(default = 5e-3)]
    lr_scale: f64,

    #[config(default = 1e-3)]
    lr_rotation: f64,

    #[config(default = 42)]
    seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        let decay_steps = 30000;
        let lr_max = 1.6e-4;
        let decay = 1e-2f64.powf(1.0 / decay_steps as f64);
        TrainConfig::new(ExponentialLrSchedulerConfig::new(lr_max, decay))
    }
}

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

    pub refine: Option<RefineStats>,
}

pub struct SplatTrainer<B: AutodiffBackend> {
    pub iter: u32,

    config: TrainConfig,

    sched_mean: ExponentialLrScheduler,
    optim: OptimizerType<B>,

    // Helper tensors for accumulating the viewspace_xy gradients and the number
    // of observations per gaussian. Used in pruning and densification.
    grad_2d_accum: Tensor<B, 1>,
    xy_grad_counts: Tensor<B, 1, Int>,
    max_radii: Tensor<B, 1>,

    ssim: Ssim<B>,
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
    let qz = quaternions.clone().slice([0..num_points, 3..4]);

    let vx = vectors.clone().slice([0..num_points, 0..1]);
    let vy = vectors.clone().slice([0..num_points, 1..2]);
    let vz = vectors.clone().slice([0..num_points, 2..3]);

    // Common terms
    let qw2 = qw.clone().powf_scalar(2.0);
    let qx2 = qx.clone().powf_scalar(2.0);
    let qy2 = qy.clone().powf_scalar(2.0);
    let qz2 = qz.clone().powf_scalar(2.0);

    // Cross products (multiplied by 2.0 later)
    let xy = qx.clone() * qy.clone();
    let xz = qx.clone() * qz.clone();
    let yz = qy.clone() * qz.clone();
    let wx = qw.clone() * qx.clone();
    let wy = qw.clone() * qy.clone();
    let wz = qw * qz;

    // Final components with reused terms
    let x = (qw2.clone() + qx2.clone() - qy2.clone() - qz2.clone()) * vx.clone()
        + (xy.clone() * vy.clone() + xz.clone() * vz.clone() + wy.clone() * vz.clone()
            - wz.clone() * vy.clone())
            * 2.0;

    let y = (qw2.clone() - qx2.clone() + qy2.clone() - qz2.clone()) * vy.clone()
        + (xy.clone() * vx.clone() + yz.clone() * vz.clone() + wz.clone() * vx.clone()
            - wx.clone() * vz.clone())
            * 2.0;

    let z = (qw2 - qx2 - qy2 + qz2) * vz.clone()
        + (xz * vx.clone() + yz * vy.clone() + wx * vy - wy * vx) * 2.0;

    Tensor::cat(vec![x, y, z], 1)
}

impl<B: AutodiffBackend> SplatTrainer<B> {
    pub fn new(num_points: usize, config: &TrainConfig, device: &B::Device) -> Self {
        let opt_config = AdamConfig::new().with_epsilon(1e-15);
        let ssim = Ssim::new(config.ssim_window_size, 3, device);

        Self {
            config: config.clone(),
            iter: 0,
            sched_mean: config.lr_mean.init().expect("Lr schedule must be valid."),
            optim: opt_config.init::<B, Splats<B>>(),
            grad_2d_accum: Tensor::zeros([num_points], device),
            xy_grad_counts: Tensor::zeros([num_points], device),
            max_radii: Tensor::zeros([num_points], device),
            ssim,
        }
    }

    fn reset_stats(&mut self, num_points: usize, device: &B::Device) {
        self.grad_2d_accum = Tensor::zeros([num_points], device);
        self.xy_grad_counts = Tensor::zeros([num_points], device);
        self.max_radii = Tensor::zeros([num_points], device);
    }

    pub(crate) fn reset_opacity(
        &self,
        splats: &mut Splats<B>,
        record: &mut HashMap<ParamId, AdaptorRecord<Adam, B>>,
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
        batch: SceneBatch<B>,
        splats: Splats<B>,
    ) -> Result<(Splats<B>, TrainStepStats<B>), anyhow::Error> {
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

            let pred_images = Tensor::stack(renders, 0);

            let _span = trace_span!("Calculate losses", sync_burn = true).entered();

            let pred_rgb = pred_images
                .clone()
                .slice([0..batch_size, 0..img_h, 0..img_w, 0..3]);

            // This is wrong if the batch has mixed transparent and non-transparent images,
            // but that's ok for now.
            let pred_compare = if batch.gt_views[0].image.color().has_alpha() {
                pred_images.clone()
            } else {
                pred_rgb.clone()
            };

            let loss = (pred_compare - batch.gt_images.clone()).abs().mean();

            // Disabled on WASM for now. On WebGPU + Metal this unfortunately has glitches.
            let loss = if self.config.ssim_weight > 0.0 && !cfg!(target_family = "wasm") {
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
            if self.iter > self.config.refine_start_iter {
                // Get the xy gradient norm from the dummy tensor.
                let xys_grad = Tensor::from_inner(
                    splats
                        .xys_dummy
                        .grad_remove(&mut grads)
                        .expect("XY gradients need to be calculated."),
                );

                let aux = auxes[0].clone();
                let gs_ids = Tensor::from_primitive(aux.global_from_compact_gid);
                let radii = Tensor::from_primitive(TensorPrimitive::Float(aux.radii));

                let [_, h, w, _] = pred_images.dims();
                let device = batch.gt_images.device();
                let xys_grad = xys_grad
                    * Tensor::<_, 1>::from_floats([w as f32 / 2.0, h as f32 / 2.0], &device)
                        .reshape([1, 2]);

                let xys_grad_norm = xys_grad.powi_scalar(2).sum_dim(1).squeeze(1).sqrt();

                let num_vis = Tensor::from_primitive(aux.num_visible.clone());
                let valid = Tensor::arange(0..splats.num_splats() as i64, &device).lower(num_vis);

                self.grad_2d_accum =
                    self.grad_2d_accum
                        .clone()
                        .select_assign(0, gs_ids.clone(), xys_grad_norm);

                self.xy_grad_counts =
                    self.xy_grad_counts
                        .clone()
                        .select_assign(0, gs_ids.clone(), valid.int());

                let radii_norm = radii / (w.max(h) as f32);
                self.max_radii = self.max_radii.clone().max_pair(radii_norm);
            }
        });

        splats = trace_span!("Optimizer step", sync_burn = true).in_scope(|| {
            let grad_means = GradientsParams::from_params(&mut grads, &splats, &[splats.means.id]);
            splats = self.optim.step(lr_mean, splats, grad_means);

            let grad_opac =
                GradientsParams::from_params(&mut grads, &splats, &[splats.raw_opacity.id]);
            splats = self.optim.step(lr_opac, splats, grad_opac);

            let old_coeffs = splats.sh_coeffs.val();
            let grad_coeff =
                GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);
            splats = self.optim.step(lr_coeffs, splats, grad_coeff);
            let num_splats = splats.num_splats();
            let sh_num = splats.sh_coeffs.dims()[1];

            // HACK: Want a lower learning rate for higher SH order.
            // This works as long as the update rule is linear.
            // (Adam Update rule is theta_{t + 1} = theta_{t} - lr * step)
            if sh_num > 1 {
                Splats::map_param(&mut splats.sh_coeffs, |coeffs| {
                    let lerp_alpha = 1.0 / self.config.lr_coeffs_sh_scale;
                    let scaled_coeffs =
                        old_coeffs.clone() * (1.0 - lerp_alpha) + coeffs.clone() * lerp_alpha;

                    let base = coeffs.slice([0..num_splats, 0..1]);
                    let scaled = scaled_coeffs.slice([0..num_splats, 1..sh_num]);

                    Tensor::cat(vec![base, scaled], 1)
                });
            }

            let grad_rot = GradientsParams::from_params(&mut grads, &splats, &[splats.rotation.id]);
            splats = self.optim.step(lr_rotation, splats, grad_rot);

            let grad_scale =
                GradientsParams::from_params(&mut grads, &splats, &[splats.log_scales.id]);
            splats = self.optim.step(lr_scale, splats, grad_scale);

            splats.norm_rotations();

            // Make sure rotations are still valid after optimization step.
            splats
        });

        let mut refine_stats = None;

        let do_refine = self.iter < self.config.refine_stop_iter
            && self.iter >= self.config.refine_start_iter
            && self.iter % self.config.refine_every == 1;

        if do_refine {
            // If not refining, update splat to step with gradients applied.
            let (refined_splats, refine) = self.refine_splats(splats, batch.scene_extent).await;
            refine_stats = Some(refine);
            splats = refined_splats;
        }

        self.iter += 1;

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
            refine: refine_stats,
        };

        Ok((splats, stats))
    }

    async fn refine_splats(
        &mut self,
        splats: Splats<B>,
        scene_extent: f32,
    ) -> (Splats<B>, RefineStats) {
        let mut record = self.optim.to_record();

        let mut splats = splats;

        let device = splats.means.device();

        // Otherwise, do refinement, but do the split/clone on gaussians with no grads applied.
        let grads = self.grad_2d_accum.clone() / self.xy_grad_counts.clone().clamp_min(1).float();

        let is_grad_high = grads.greater_equal_elem(self.config.densify_grad_thresh);
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

        let clone_inds =
            Tensor::stack::<2>(vec![is_grad_high.clone(), split_clone_size_mask.clone()], 1)
                .all_dim(1)
                .squeeze::<1>(1)
                .argwhere_async()
                .await;

        // Clone splats
        let clone_count = clone_inds.dims()[0];
        if clone_count > 0 {
            let clone_inds = clone_inds.squeeze(1);
            let cur_rots = splats.rotation.val().select(0, clone_inds.clone());
            let cur_scale = splats.log_scales.val().select(0, clone_inds.clone());
            append_means.push(splats.means.val().select(0, clone_inds.clone()));
            append_rots.push(cur_rots);
            append_coeffs.push(splats.sh_coeffs.val().select(0, clone_inds.clone()));
            append_opac.push(splats.raw_opacity.val().select(0, clone_inds.clone()));
            append_scales.push(cur_scale);
        }

        // Split splats.
        let split_mask =
            Tensor::stack::<2>(vec![is_grad_high, split_clone_size_mask.bool_not()], 1)
                .all_dim(1)
                .squeeze(1);

        let radii_grow = self
            .max_radii
            .clone()
            .greater_elem(self.config.densify_radius_threshold);
        let split_mask = Tensor::stack::<2>(vec![split_mask, radii_grow], 1)
            .any_dim(1)
            .squeeze(1);

        let split_inds = split_mask.clone().argwhere_async().await;
        let split_count = split_inds.dims()[0];

        if split_count > 0 {
            let split_inds = split_inds.squeeze(1);

            for _ in 0..2 {
                // Some parts can be straightforwardly copied to the new splats.
                let cur_means = splats.means.val().select(0, split_inds.clone());
                let cur_coeff = splats.sh_coeffs.val().select(0, split_inds.clone());
                let cur_raw_opac = splats.raw_opacity.val().select(0, split_inds.clone());
                let cur_rots = splats.rotation.val().select(0, split_inds.clone());
                let cur_scale = splats.log_scales.val().select(0, split_inds.clone());

                let samples = quaternion_vec_multiply(
                    cur_rots.clone(),
                    Tensor::random([split_count, 3], Distribution::Normal(0.0, 1.0), &device),
                ) * cur_scale.clone().exp();

                append_means.push(cur_means.clone() + samples);
                append_rots.push(cur_rots.clone());
                append_scales.push(cur_scale - 1.6f32.ln());
                append_coeffs.push(cur_coeff.clone());
                append_opac.push(cur_raw_opac.clone());
            }
        }

        prune_points(&mut splats, &mut record, split_mask).await;

        // Do some more processing. Important to do this last as otherwise you might mess up the correspondence
        // of gradient <-> splat.
        let start_count = splats.num_splats();
        // Remove barely visible gaussians.
        let alpha_mask = splats.opacity().lower_elem(self.config.cull_opacity);
        prune_points(&mut splats, &mut record, alpha_mask).await;
        let alpha_pruned = start_count - splats.num_splats();

        // Delete Gaussians with too large of a radius in world-units.
        let scale_mask = splats
            .scales()
            .max_dim(1)
            .squeeze(1)
            .greater_elem(self.config.cull_scale3d_percentage_threshold * scene_extent);
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

        let refine_step = self.iter / self.config.refine_every;
        if refine_step % self.config.reset_alpha_every_refine == 0 {
            self.reset_opacity(&mut splats, &mut record);
        }

        // Stats don't line up anymore so have to reset them.
        self.reset_stats(splats.num_splats(), &device);
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
    record: &mut HashMap<ParamId, AdaptorRecord<Adam, B>>,
    map_param: impl Fn(Tensor<B, D>) -> Tensor<B, D>,
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
    record: &mut HashMap<ParamId, AdaptorRecord<Adam, B>>,
    prune: Tensor<B, 1, Bool>,
) {
    assert!(prune.dims()[0] == splats.num_splats());

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
    record: &mut HashMap<ParamId, AdaptorRecord<Adam, B>>,
    means: Tensor<B, 2>,
    rotations: Tensor<B, 2>,
    sh_coeffs: Tensor<B, 3>,
    raw_opac: Tensor<B, 1>,
    log_scales: Tensor<B, 2>,
) {
    // Concat
    map_param(
        &mut splats.means,
        record,
        |x| Tensor::cat(vec![x, means.clone()], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros_like(&means.clone().inner())], 0),
    );
    map_param(
        &mut splats.rotation,
        record,
        |x| Tensor::cat(vec![x, rotations.clone()], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros_like(&rotations.clone().inner())], 0),
    );
    map_param(
        &mut splats.sh_coeffs,
        record,
        |x| Tensor::cat(vec![x, sh_coeffs.clone()], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros_like(&sh_coeffs.clone().inner())], 0),
    );
    map_param(
        &mut splats.raw_opacity,
        record,
        |x| Tensor::cat(vec![x, raw_opac.clone()], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros_like(&raw_opac.clone().inner())], 0),
    );
    map_param(
        &mut splats.log_scales,
        record,
        |x| Tensor::cat(vec![x, log_scales.clone()], 0),
        |x| Tensor::cat(vec![x, Tensor::zeros_like(&log_scales.clone().inner())], 0),
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
        let result: Vec<f32> = result.into_data().to_vec().unwrap();
        let result = glam::vec3(result[0], result[1], result[2]);
        assert!((result_ref - result).length() < 1e-7);
    }
}
