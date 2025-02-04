use anyhow::Result;
use brush_render::gaussian_splats::Splats;
use brush_render::render::sh_coeffs_for_degree;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Vulkan};
use burn::lr_scheduler::exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig};
use burn::lr_scheduler::LrScheduler;
use burn::module::ParamId;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::record::AdaptorRecord;
use burn::optim::Optimizer;
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, Distribution, Int, TensorPrimitive};
use burn::{config::Config, optim::GradientsParams, tensor::Tensor};
use hashbrown::HashMap;
use tracing::trace_span;

use crate::adam_scaled::{AdamScaled, AdamScaledConfig, AdamState};
use crate::burn_glue::SplatForwardDiff;
use crate::multinomial::multinomial_sample;
use crate::scene::{SceneView, ViewImageType};
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
    #[config(default = 1e-4)]
    #[arg(long, help_heading = "Training options", default_value = "1e-4")]
    lr_mean: f64,

    /// Start learning rate for the mean.
    #[config(default = 1e-2)]
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    lr_mean_decay: f64,

    #[config(default = 0.0)]
    #[arg(long, help_heading = "Training options", default_value = "0.0")]
    mean_noise_weight: f32,

    /// Learning rate for the basic coefficients.
    #[config(default = 3e-3)]
    #[arg(long, help_heading = "Training options", default_value = "3e-3")]
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

    /// GSs with opacity below this value will be pruned
    #[config(default = 0.002)]
    #[arg(long, help_heading = "Refine options", default_value = "0.002")]
    cull_opacity: f32,

    /// Threshold for positional gradient norm
    #[config(default = 0.0015)]
    #[arg(long, help_heading = "Refine options", default_value = "0.0015")]
    densify_grad_thresh: f32,

    /// Weight of l1 loss on alpha if input view has alpha transparency.
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Training options", default_value = "0.1")]
    alpha_match_weight: f32,

    /// Weight of mean-opacity loss.
    #[config(default = 0.15)]
    #[arg(long, help_heading = "Training options", default_value = "0.15")]
    max_growth_rate: f32,

    /// Period of steps where gaussians are culled and densified
    #[config(default = 100)]
    #[arg(long, help_heading = "Refine options", default_value = "100")]
    refine_every: u32,

    #[config(default = 10000000)]
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    max_splat_count: u32,

    #[config(default = 5e-7)]
    #[arg(long, help_heading = "Refine options", default_value = "5e-7")]
    opac_loss_weight: f32,

    #[config(default = 0.0)]
    #[arg(long, help_heading = "Refine options", default_value = "0.0")]
    scale_loss_weight: f32,
}

pub type TrainBack = Autodiff<Vulkan>;

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_image: Tensor<B, 3>,
    pub gt_view: SceneView,
}

#[derive(Clone)]
pub struct RefineStats {
    pub num_transparent_pruned: u32,
    pub num_scale_pruned: u32,
    pub splats_added: u32,
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
    optim: OptimizerType,
    ssim: Ssim<TrainBack>,
    refine_record: RefineRecord<<TrainBack as AutodiffBackend>::InnerBackend>,
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

impl SplatTrainer {
    pub fn new(init_count: u32, config: &TrainConfig, device: &WgpuDevice) -> Self {
        // I've tried some other momentum settings, but without much luck.
        let optim = AdamScaledConfig::new().with_epsilon(1e-15).init();

        let ssim = Ssim::new(config.ssim_window_size, 3, device);
        let decay = config.lr_mean_decay.powf(1.0 / config.total_steps as f64);
        let lr_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay);

        Self {
            config: config.clone(),
            sched_mean: lr_mean.init().expect("Lr schedule must be valid."),
            optim,
            refine_record: RefineRecord::new(init_count, device),
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

        let (pred_image, aux, refine_weight_hold) = {
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
                        + (alpha_input - pred_alpha).abs().mean() * self.config.alpha_match_weight
                }
            }
        } else {
            total_err.mean()
        };

        let train_t = iter as f64 / self.config.total_steps as f64;

        // Only regularize opacity if we're still growing splat count.
        if self.config.opac_loss_weight > 0.0 {
            loss = loss + splats.opacity().sum() * (self.config.opac_loss_weight);
        }

        // // Only regularize opacity if we're still growing splat count.
        // if self.config.scale_loss_weight > 0.0 {
        //     loss = loss
        //         + splats.scales().max_dim(1).sum()
        //             * (self.config.scale_loss_weight * (1.0 - train_t) as f32);
        // }

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

        splats = trace_span!("Optimizer step", sync_burn = true).in_scope(|| {
            splats = trace_span!("SH Coeffs step", sync_burn = true).in_scope(|| {
                let grad_coeff =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);

                let mut record = self.optim.to_record();

                let mut param_record = record.get_mut(&splats.sh_coeffs.id);

                if let Some(param) = param_record.as_mut() {
                    let mut state = param.clone().into_state();

                    if state.scaling.is_none() {
                        let coeff_count = sh_coeffs_for_degree(splats.sh_degree()) as i32;
                        let sh_size = coeff_count;
                        let mut sh_lr_scales = vec![1.0];
                        for _ in 1..sh_size {
                            sh_lr_scales.push(1.0 / self.config.lr_coeffs_sh_scale);
                        }
                        let sh_lr_scales = Tensor::<_, 1>::from_floats(
                            sh_lr_scales.as_slice(),
                            &splats.means.device(),
                        )
                        .reshape([1, coeff_count, 1]);

                        state.scaling = Some(sh_lr_scales);
                        record.insert(splats.sh_coeffs.id, AdaptorRecord::from_state(state));
                        self.optim = self.optim.clone().load_record(record);
                    }
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

            // Make sure rotations are still valid after optimization step.
            splats
        });

        let device = splats.means.device();
        let num_visible = aux.num_visible.clone();
        let num_intersections = aux.num_intersections.clone();

        trace_span!("Housekeeping", sync_burn = true).in_scope(|| {
            if self.is_growing_splats(iter) {
                // Get the xy gradient norm from the dummy tensor.
                let refine_weight = refine_weight_hold
                    .grad_remove(&mut grads)
                    .expect("XY gradients need to be calculated.");
                let aux = aux.clone();
                self.refine_record.gather_stats(refine_weight, aux);
            }
        });

        // Add random noise. Only do this in the growth phase, otherwise
        // let the splats settle in without noise, not much point in exploring regions anymore.
        // trace_span!("Noise means").in_scope(|| {
        let one = Tensor::ones([1], &device);
        let noise_weight = (one - splats.opacity().inner()).powf_scalar(100.0);
        let noise_weight = noise_weight.unsqueeze_dim(1);

        // How much lr_mean has decayed.
        let lr_mean_decay = lr_mean / self.config.lr_mean;

        let samples = quaternion_vec_multiply(
            splats.rotations_normed().inner(),
            Tensor::random(
                [splats.num_splats() as usize, 3],
                Distribution::Normal(0.0, 1.0),
                &device,
            ) * splats.scales().inner(),
        );

        let mean_noise = samples
            * (noise_weight
                * scene_extent
                * lr_mean_decay as f32
                * (1.0 - train_t as f32)
                * self.config.mean_noise_weight);

        splats.means = splats
            .means
            .map(|m| Tensor::from_inner(m.inner() + mean_noise).require_grad());

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

    fn is_growing_splats(&self, iter: u32) -> bool {
        iter >= self.config.refine_every * 5
            && iter <= self.config.total_steps - self.config.refine_every * 100
    }

    pub async fn refine_if_needed(
        &mut self,
        iter: u32,
        splats: Splats<TrainBack>,
    ) -> (Splats<TrainBack>, Option<RefineStats>) {
        if iter % self.config.refine_every != 0 || !self.is_growing_splats(iter) || iter == 0 {
            return (splats, None);
        }

        let mut record = self.optim.to_record();
        // Create new optimizer, so that we don't hang on to the old state.
        // TODO: Mayne just need to add a into_record() to Burn.
        self.optim = AdamScaledConfig::new().with_epsilon(1e-15).init();

        let mut splats = splats;
        let device = splats.means.device();
        // let train_t = iter as f32 / self.config.total_steps as f32;

        // Calculate growth.
        let refine_weight = self.refine_record.refine_weight();
        let growth_percent = refine_weight
            .clone()
            .greater_elem(self.config.densify_grad_thresh)
            .float()
            .sum()
            / splats.num_splats();
        let growth_percent = growth_percent
            .into_scalar_async()
            .await
            .min(self.config.max_growth_rate);
        // let growth_percent = growth_percent * (1.0 - train_t);

        // Prune based on alpha.
        let start_count = splats.num_splats();
        let alpha_mask = splats
            .opacity()
            .inner()
            .lower_elem(self.config.cull_opacity);
        splats = prune_points(splats, &mut record, alpha_mask).await;
        let alpha_pruned = start_count - splats.num_splats();

        // Delete Gaussians with too large of a radius in world-units.
        let scale_big = splats.log_scales.val().inner().greater_elem(100f32.ln());

        // less than e^-10, too small to care about.
        let scale_small = splats.log_scales.val().inner().lower_elem(-10.0);

        let start_count = splats.num_splats();
        let scale_mask =
            Tensor::any_dim(Tensor::cat(vec![scale_small, scale_big], 1), 1).squeeze(1);
        splats = prune_points(splats, &mut record, scale_mask).await;
        let scale_pruned = start_count - splats.num_splats();

        // Add new splats as needed.
        let target_splats = (splats.num_splats() as f32 * (1.0 + growth_percent)).round() as u32;
        let target_splats = target_splats.min(self.config.max_splat_count);

        let add_count = target_splats.saturating_sub(splats.num_splats());

        if add_count > 0 {
            // Pick gaussians to clone, weighted by their opacity.
            // let weight = splats.opacity().inner() / splats.opacity().inner().sum()
            //     * (1.0 - train_t)
            //     + refine_weight.clone() / refine_weight.sum() * train_t;

            let split_inds = multinomial_sample(splats.opacity().inner(), add_count).await;

            // Add up how many times this splat has been split.
            let counts: Tensor<_, 1> = Tensor::ones([splats.num_splats() as usize], &device);
            let counts = counts.clone().select_assign(0, split_inds.clone(), counts);

            // splats.raw_opacity.val().select_assign(dim, indices, values)
            splats = concat_splats(splats, &mut record, &split_inds);

            let counts_concat = counts.clone().select(0, split_inds);
            let counts = Tensor::cat(vec![counts, counts_concat], 0);

            let cur_opacity = splats.opacity().inner();

            // Adjust the opacity of the gaussians about to be split.
            //
            // o' = 1 - (1 - o)^(1/N)
            splats.raw_opacity = splats.raw_opacity.map(|_| {
                // TODO: Burn doesn't do 1.0/x so errr, just x^-1 will do I guess!
                let inv_counts = counts.clone().powf_scalar(-1.0);

                let one = Tensor::ones([1], &device);
                let opac_p = one.clone() - (one - cur_opacity.clone()).powf(inv_counts);
                Tensor::from_inner(inv_sigmoid(opac_p)).require_grad()
            });

            // s' = s * (1 - o * 0.5) ^ ((N-1)/N)
            // splats.log_scales = splats.log_scales.map(|s| {
            //     // let scale_scaling =
            //     //     (one - cur_opacity * 0.01).powf((counts.clone() - 1.0) / counts);
            //     // let new_scale = s.inner().exp() * scale_scaling.unsqueeze_dim(1);
            //     Tensor::from_inner(s.inner() - ((counts - 1.0) * 0.1 + 1.0).unsqueeze_dim(1).log())
            // });
        }

        self.optim = self.optim.clone().load_record(record);
        self.refine_record = RefineRecord::new(splats.num_splats(), &device);

        let stats = RefineStats {
            num_transparent_pruned: alpha_pruned,
            num_scale_pruned: scale_pruned,
            splats_added: add_count,
        };
        (splats, Some(stats))
    }
}

fn map_splats_and_opt<B: AutodiffBackend>(
    mut splats: Splats<B>,
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
    splats.means = splats
        .means
        .map(|m| Tensor::from_inner(map_mean(m.inner())).require_grad());
    map_opt(splats.means.id, record, &map_opt_mean);

    splats.rotation = splats
        .rotation
        .map(|m| Tensor::from_inner(map_rotation(m.inner())).require_grad());
    map_opt(splats.rotation.id, record, &map_opt_rotation);

    splats.log_scales = splats
        .log_scales
        .map(|m| Tensor::from_inner(map_scale(m.inner())).require_grad());
    map_opt(splats.log_scales.id, record, &map_opt_scale);

    splats.sh_coeffs = splats
        .sh_coeffs
        .map(|m| Tensor::from_inner(map_coeffs(m.inner())).require_grad());
    map_opt(splats.sh_coeffs.id, record, &map_opt_coeffs);

    splats.raw_opacity = splats
        .raw_opacity
        .map(|m| Tensor::from_inner(map_opac(m.inner())).require_grad());
    map_opt(splats.raw_opacity.id, record, &map_opt_opac);

    splats
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
    state.momentum.moment_1 = map_opt(state.momentum.moment_1);
    state.momentum.moment_2 = map_opt(state.momentum.moment_2);
    record.insert(param_id, AdaptorRecord::from_state(state));
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
async fn prune_points<B: AutodiffBackend>(
    mut splats: Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    prune: Tensor<B::InnerBackend, 1, Bool>,
) -> Splats<B> {
    assert_eq!(
        prune.dims()[0] as u32,
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    // bool[n]. If True, delete these Gaussians.
    let prune_count = prune.dims()[0];

    if prune_count == 0 {
        return splats;
    }

    let valid_inds = prune.bool_not().argwhere_async().await;

    if valid_inds.dims()[0] == 0 {
        log::warn!("Trying to create empty splat!");
        return splats;
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

    splats
}

pub fn concat_splats<B: AutodiffBackend>(
    splats: Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    copy_inds: &Tensor<B::InnerBackend, 1, Int>,
) -> Splats<B> {
    // Concat
    let device = splats.means.device();

    let cur_count = splats.means.dims()[0];
    let append_count = copy_inds.dims()[0];
    let sh_dim = splats.sh_coeffs.dims()[1];

    map_splats_and_opt(
        splats,
        record,
        |x| Tensor::cat(vec![x.clone(), x.select(0, copy_inds.clone())], 0),
        |x| Tensor::cat(vec![x.clone(), x.select(0, copy_inds.clone())], 0),
        |x| Tensor::cat(vec![x.clone(), x.select(0, copy_inds.clone())], 0),
        |x| Tensor::cat(vec![x.clone(), x.select(0, copy_inds.clone())], 0),
        |x| Tensor::cat(vec![x.clone(), x.select(0, copy_inds.clone())], 0),
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
