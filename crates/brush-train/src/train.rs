use std::f64::consts::SQRT_2;

use anyhow::Result;
use brush_render::gaussian_splats::{Splats, inverse_sigmoid};

use brush_render::sh::sh_coeffs_for_degree;
use burn::backend::wgpu::{WgpuDevice, WgpuRuntime};
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
use burn::tensor::{Bool, Distribution, Int, TensorData, TensorPrimitive};
use burn::{config::Config, optim::GradientsParams, tensor::Tensor};
use burn_cubecl::cubecl::Runtime;
use hashbrown::{HashMap, HashSet};
use tracing::trace_span;

use crate::adam_scaled::{AdamScaled, AdamScaledConfig, AdamState};
use crate::burn_glue::SplatForwardDiff;
use crate::multinomial::multinomial_sample;
use crate::scene::{SceneView, ViewImageType};
use crate::ssim::Ssim;
use crate::stats::RefineRecord;
use clap::Args;

const MIN_OPACITY: f32 = 0.9 / 255.0;

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

    /// Start learning rate for the mean parameters.
    #[config(default = 4e-5)]
    #[arg(long, help_heading = "Training options", default_value = "4e-5")]
    lr_mean: f64,

    /// Start learning rate for the mean parameters.
    #[config(default = 4e-7)]
    #[arg(long, help_heading = "Training options", default_value = "4e-7")]
    lr_mean_end: f64,

    /// How much noise to add to the mean parameters of low opacity gaussians.
    #[config(default = 1e4)]
    #[arg(long, help_heading = "Training options", default_value = "1e4")]
    mean_noise_weight: f32,

    /// Learning rate for the base SH (RGB) coefficients.
    #[config(default = 3e-3)]
    #[arg(long, help_heading = "Training options", default_value = "3e-3")]
    lr_coeffs_dc: f64,

    /// How much to divide the learning rate by for higher SH orders.
    #[config(default = 20.0)]
    #[arg(long, help_heading = "Training options", default_value = "20.0")]
    lr_coeffs_sh_scale: f32,

    /// Learning rate for the opacity parameter.
    #[config(default = 3e-2)]
    #[arg(long, help_heading = "Training options", default_value = "3e-2")]
    lr_opac: f64,

    /// Learning rate for the scale parameters.
    #[config(default = 1e-2)]
    #[arg(long, help_heading = "Training options", default_value = "1e-2")]
    lr_scale: f64,

    /// Learning rate for the scale parameters.
    #[config(default = 6e-3)]
    #[arg(long, help_heading = "Training options", default_value = "6e-3")]
    lr_scale_end: f64,

    /// Learning rate for the rotation parameters.
    #[config(default = 1e-3)]
    #[arg(long, help_heading = "Training options", default_value = "1e-3")]
    lr_rotation: f64,

    /// Weight of the opacity loss.
    #[config(default = 1e-8)]
    #[arg(long, help_heading = "Training options", default_value = "1e-8")]
    opac_loss_weight: f32,

    /// Frequency of 'refinement' where gaussians are replaced and densified. This should
    /// roughly be the number of images it takes to properly "cover" your scene.
    #[config(default = 150)]
    #[arg(long, help_heading = "Refine options", default_value = "150")]
    refine_every: u32,

    /// Threshold to control splat growth. Lower means faster growth.
    #[config(default = 0.00085)]
    #[arg(long, help_heading = "Refine options", default_value = "0.00085")]
    growth_grad_threshold: f32,

    /// What fraction of splats that are deemed as needing to grow do actually grow.
    /// Increase this to make splats grow more aggressively.
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    growth_select_fraction: f32,

    /// Period after which splat growth stops.
    #[config(default = 12500)]
    #[arg(long, help_heading = "Refine options", default_value = "12500")]
    growth_stop_iter: u32,

    /// Weight of l1 loss on alpha if input view has transparency.
    #[config(default = 0.1)]
    #[arg(long, help_heading = "Refine options", default_value = "0.1")]
    match_alpha_weight: f32,

    /// Max nr. of splats. This is an upper bound, but the actual final number of splats might be lower than this.
    #[config(default = 10000000)]
    #[arg(long, help_heading = "Refine options", default_value = "10000000")]
    max_splats: u32,
}

pub type TrainBack = Autodiff<Wgpu>;

#[derive(Clone, Debug)]
pub struct SceneBatch<B: Backend> {
    pub gt_image: Tensor<B, 3>,
    pub gt_view: SceneView,
}

#[derive(Clone)]
pub struct RefineStats {
    pub num_added: u32,
    pub num_pruned: u32,
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
    sched_scale: ExponentialLrScheduler,
    ssim: Ssim<TrainBack>,

    refine_record: Option<RefineRecord<<TrainBack as AutodiffBackend>::InnerBackend>>,
    optim: Option<OptimizerType>,
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

        let decay = (config.lr_scale_end / config.lr_scale).powf(1.0 / config.total_steps as f64);
        let lr_scale = ExponentialLrSchedulerConfig::new(config.lr_scale, decay);

        Self {
            config: config.clone(),
            sched_mean: lr_mean.init().expect("Mean lr schedule must be valid."),
            sched_scale: lr_scale.init().expect("Scale lr schedule must be valid."),
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

        let current_opacity = splats.opacities();

        let (
            pred_image,
            visible,
            global_from_compact_gid,
            num_visible,
            num_intersections,
            refine_weight_holder,
        ) = {
            let diff_out = <TrainBack as SplatForwardDiff<TrainBack>>::render_splats(
                camera,
                glam::uvec2(img_w as u32, img_h as u32),
                splats.means.val().into_primitive().tensor(),
                splats.log_scales.val().into_primitive().tensor(),
                splats.rotation.val().into_primitive().tensor(),
                splats.sh_coeffs.val().into_primitive().tensor(),
                current_opacity.clone().into_primitive().tensor(),
            );
            let img = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
            (
                img,
                diff_out.aux.visible,
                diff_out.aux.global_from_compact_gid,
                diff_out.aux.num_visible,
                diff_out.aux.num_intersections,
                diff_out.refine_weight_holder,
            )
        };

        let train_t = (iter as f32 / self.config.total_steps as f32).clamp(0.0, 1.0);

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

        let loss = if batch.gt_view.image.color().has_alpha() {
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

        let opac_loss_weight = self.config.opac_loss_weight;
        let visible: Tensor<_, 1> = Tensor::from_primitive(TensorPrimitive::Float(visible));

        let loss = if opac_loss_weight > 0.0 {
            // let visible_count = visible.clone().sum();
            // Invisible splats still have a tiny bit of loss. Otherwise,
            // they would never die off.
            let visible = visible.clone() + 1e-3;
            loss + (current_opacity * visible).sum() * (opac_loss_weight * (1.0 - train_t))
        } else {
            loss
        };

        let mut grads = trace_span!("Backward pass", sync_burn = true).in_scope(|| loss.backward());

        let (lr_mean, lr_rotation, lr_scale, lr_coeffs, lr_opac) = (
            self.sched_mean.step() * scene_extent as f64,
            self.config.lr_rotation,
            // Scale is relative to the scene scale, but the exp() activation function
            // means "offsetting" all values also solves the learning rate scaling.
            self.sched_scale.step(),
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

        trace_span!("Housekeeping", sync_burn = true).in_scope(|| {
            // Get the xy gradient norm from the dummy tensor.
            let refine_weight = refine_weight_holder
                .grad_remove(&mut grads)
                .expect("XY gradients need to be calculated.");

            let device = splats.device();
            let num_splats = splats.num_splats();
            let record = self
                .refine_record
                .get_or_insert_with(|| RefineRecord::new(num_splats, &device));

            record.gather_stats(
                refine_weight,
                glam::uvec2(img_w as u32, img_h as u32),
                global_from_compact_gid,
                num_visible.clone(),
            );
        });

        let mean_noise_weight_scale = self.config.mean_noise_weight * (1.0 - train_t);

        if mean_noise_weight_scale > 0.0 {
            let device = splats.device();
            // Add random noise. Only do this in the growth phase, otherwise
            // let the splats settle in without noise, not much point in exploring regions anymore.
            // trace_span!("Noise means").in_scope(|| {
            let one = Tensor::ones([1], &device);
            let noise_weight = (one - splats.opacities().inner())
                .powf_scalar(100.0)
                .clamp(0.0, 1.0);
            let noise_weight = noise_weight * visible.inner(); // Only noise visible gaussians.
            let noise_weight = noise_weight.unsqueeze_dim(1);

            let samples = quaternion_vec_multiply(
                splats.rotations_normed().inner(),
                Tensor::random(
                    [splats.num_splats() as usize, 3],
                    Distribution::Normal(0.0, 1.0),
                    &device,
                ) * splats.scales().inner(),
            );

            let noise_weight = noise_weight * (lr_mean as f32 * mean_noise_weight_scale);
            splats.means = splats
                .means
                .map(|m| Tensor::from_inner(m.inner() + samples * noise_weight).require_grad());
        }

        let stats = TrainStepStats {
            pred_image,
            gt_views: batch.gt_view,
            num_visible: Tensor::from_primitive(num_visible),
            num_intersections: Tensor::from_primitive(num_intersections),
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
    ) -> (Splats<TrainBack>, Option<RefineStats>) {
        if iter == 0 || iter % self.config.refine_every != 0 {
            return (splats, None);
        }

        let device = splats.means.device();
        let client = WgpuRuntime::client(&device);
        client.memory_cleanup();

        // If not refining, update splat to step with gradients applied.
        // Prune dead splats. This ALWAYS happen even if we're not "refining" anymore.
        let mut record = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized")
            .to_record();
        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");
        let alpha_mask = splats
            .raw_opacity
            .val()
            .inner()
            .lower_elem(inverse_sigmoid(MIN_OPACITY));

        let (mut splats, refiner, pruned_count) =
            prune_points(splats, &mut record, refiner, alpha_mask).await;

        let mut add_indices = HashSet::new();

        // Replace dead gaussians if we're still refining.
        if pruned_count > 0 {
            // Sample from random opacities.
            let resampled_weights = splats.opacities().inner();
            let resampled_weights = resampled_weights
                .into_data_async()
                .await
                .to_vec::<f32>()
                .expect("Failed to read weights");
            let resampled_inds = multinomial_sample(&resampled_weights, pruned_count);
            add_indices.extend(resampled_inds);
        }

        if iter < self.config.growth_stop_iter {
            let above_threshold = refiner
                .refine_weight_norm
                .clone()
                .greater_elem(self.config.growth_grad_threshold)
                .int();
            let threshold_count = above_threshold.clone().sum().into_scalar_async().await as u32;

            let grow_count =
                (threshold_count as f32 * self.config.growth_select_fraction).round() as u32;

            let sample_high_grad = grow_count.saturating_sub(pruned_count);

            // Only grow to the max nr. of splats.
            let cur_splats = splats.num_splats() + add_indices.len() as u32;
            let grow_count = sample_high_grad.min(self.config.max_splats - cur_splats);

            // If still growing, sample from indices which are over the threshold.
            if grow_count > 0 {
                let weights = above_threshold.float() * refiner.refine_weight_norm;
                let weights = weights
                    .into_data_async()
                    .await
                    .to_vec::<f32>()
                    .expect("Failed to read weights");
                let growth_inds = multinomial_sample(&weights, grow_count);
                add_indices.extend(growth_inds);
            }
        }

        let refine_count = add_indices.len();

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(add_indices.into_iter().collect(), [refine_count]),
                &device,
            );

            let cur_means = splats.means.val().inner().select(0, refine_inds.clone());
            let cur_rots = splats
                .rotations_normed()
                .inner()
                .select(0, refine_inds.clone());
            let cur_log_scale = splats
                .log_scales
                .val()
                .inner()
                .select(0, refine_inds.clone());
            let cur_coeff = splats
                .sh_coeffs
                .val()
                .inner()
                .select(0, refine_inds.clone());
            let cur_raw_opac = splats
                .raw_opacity
                .val()
                .inner()
                .select(0, refine_inds.clone());

            // The amount to offset the scale and opacity should maybe depend on how far away we have sampled these gaussians,
            // but a fixed amount seems to work ok. The only note is that divide by _less_ than SQRT(2) seems to exponentially
            // blow up, as more 'mass' is added each refine.
            let scale_div = Tensor::ones_like(&cur_log_scale) * SQRT_2.ln();

            let one = Tensor::ones([1], &device);
            let cur_opac = sigmoid(cur_raw_opac.clone());
            let new_opac = one.clone() - (one - cur_opac).sqrt();
            let new_raw_opac = inv_sigmoid(new_opac.clamp(1e-24, 1.0 - 1e-24));

            // Scatter needs [N, 3] indices for means and scales.
            let refine_inds_2d = refine_inds.clone().unsqueeze_dim(1).repeat_dim(1, 3);

            let samples = quaternion_vec_multiply(
                cur_rots.clone(),
                Tensor::random([refine_count, 3], Distribution::Normal(0.0, 0.5), &device)
                    * cur_log_scale.clone().exp(),
            );

            // Shrink & offset existing splats.
            splats.means = splats.means.map(|m| {
                let new_means = m
                    .inner()
                    .scatter(0, refine_inds_2d.clone(), -samples.clone());
                Tensor::from_inner(new_means).require_grad()
            });
            splats.log_scales = splats.log_scales.map(|s| {
                let new_scales = s
                    .inner()
                    .scatter(0, refine_inds_2d.clone(), -scale_div.clone());
                Tensor::from_inner(new_scales).require_grad()
            });
            splats.raw_opacity = splats.raw_opacity.map(|m| {
                let difference = new_raw_opac.clone() - cur_raw_opac.clone();
                let new_opacities = m.inner().scatter(0, refine_inds.clone(), difference);
                Tensor::from_inner(new_opacities).require_grad()
            });

            // Concatenate new splats.
            let sh_dim = splats.sh_coeffs.dims()[1];
            splats = map_splats_and_opt(
                splats,
                &mut record,
                |x| Tensor::cat(vec![x, cur_means + samples], 0),
                |x| Tensor::cat(vec![x, cur_rots], 0),
                |x| Tensor::cat(vec![x, cur_log_scale - scale_div], 0),
                |x| Tensor::cat(vec![x, cur_coeff], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 3], &device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 4], &device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 3], &device)], 0),
                |x| {
                    Tensor::cat(
                        vec![x, Tensor::zeros([refine_count, sh_dim, 3], &device)],
                        0,
                    )
                },
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count], &device)], 0),
            );
        }

        self.optim = Some(create_default_optimizer().load_record(record));

        client.memory_cleanup();

        (
            splats,
            Some(RefineStats {
                num_added: refine_count as u32,
                num_pruned: pruned_count,
            }),
        )
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
        .map(|x| Tensor::from_inner(map_mean(x.inner())).require_grad());
    map_opt(splats.means.id, record, &map_opt_mean);

    splats.rotation = splats
        .rotation
        .map(|x| Tensor::from_inner(map_rotation(x.inner())).require_grad());
    map_opt(splats.rotation.id, record, &map_opt_rotation);

    splats.log_scales = splats
        .log_scales
        .map(|x| Tensor::from_inner(map_scale(x.inner())).require_grad());
    map_opt(splats.log_scales.id, record, &map_opt_scale);

    splats.sh_coeffs = splats
        .sh_coeffs
        .map(|x| Tensor::from_inner(map_coeffs(x.inner())).require_grad());
    map_opt(splats.sh_coeffs.id, record, &map_opt_coeffs);

    splats.raw_opacity = splats
        .raw_opacity
        .map(|x| Tensor::from_inner(map_opac(x.inner())).require_grad());
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
async fn prune_points<B: AutodiffBackend>(
    mut splats: Splats<B>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, B>>,
    mut refiner: RefineRecord<B::InnerBackend>,
    prune: Tensor<B::InnerBackend, 1, Bool>,
) -> (Splats<B>, RefineRecord<B::InnerBackend>, u32) {
    assert_eq!(
        prune.dims()[0] as u32,
        splats.num_splats(),
        "Prune mask must have same number of elements as splats"
    );

    let prune_count = prune.dims()[0];
    if prune_count == 0 {
        return (splats, refiner, 0);
    }

    let valid_inds = prune.bool_not().argwhere_async().await;

    if valid_inds.dims()[0] == 0 {
        log::warn!("Trying to create empty splat!");
        return (splats, refiner, 0);
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
        refiner = refiner.keep(valid_inds);
    }

    (splats, refiner, start_splats - new_points)
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
