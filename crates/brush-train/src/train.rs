use crate::{
    adam_scaled::{AdamScaled, AdamScaledConfig, AdamState},
    config::TrainConfig,
    msg::{RefineStats, TrainStepStats},
    multinomial::multinomial_sample,
    quat_vec::quaternion_vec_multiply,
    splat_init::bounds_from_pos,
    ssim::Ssim,
    stats::RefineRecord,
};

use brush_dataset::scene::SceneBatch;
use brush_render::{AlphaMode, MainBackend, gaussian_splats::Splats};
use brush_render::{bounding_box::BoundingBox, sh::sh_coeffs_for_degree};
use brush_render_bwd::render_splats;
use burn::{
    backend::{
        Autodiff,
        wgpu::{WgpuDevice, WgpuRuntime},
    },
    lr_scheduler::{
        LrScheduler,
        exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    },
    module::ParamId,
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor, record::AdaptorRecord},
    prelude::Backend,
    tensor::{
        Bool, Distribution, IndexingUpdateOp, Tensor, TensorData, TensorPrimitive,
        activation::sigmoid, backend::AutodiffBackend, s,
    },
};

use burn_cubecl::cubecl::Runtime;
use hashbrown::{HashMap, HashSet};
use tracing::{Instrument, trace_span};

pub const BOUND_PERCENTILE: f32 = 0.8;

const MIN_OPACITY: f32 = 1.0 / 255.0;

type DiffBackend = Autodiff<MainBackend>;
type OptimizerType = OptimizerAdaptor<AdamScaled, Splats<DiffBackend>, DiffBackend>;

pub struct SplatTrainer {
    config: TrainConfig,
    sched_mean: ExponentialLrScheduler,
    sched_scale: ExponentialLrScheduler,
    refine_record: Option<RefineRecord<MainBackend>>,
    optim: Option<OptimizerType>,
    ssim: Option<Ssim<DiffBackend>>,
    bounds: BoundingBox,
    #[cfg(not(target_family = "wasm"))]
    lpips: Option<lpips::LpipsModel<DiffBackend>>,
}

fn inv_sigmoid<B: Backend>(x: Tensor<B, 1>) -> Tensor<B, 1> {
    (x.clone() / (1.0f32 - x)).log()
}

fn create_default_optimizer() -> OptimizerType {
    AdamScaledConfig::new().with_epsilon(1e-15).init()
}

pub async fn get_splat_bounds<B: Backend>(splats: Splats<B>, percentile: f32) -> BoundingBox {
    let means: Vec<f32> = splats
        .means()
        .into_data_async()
        .await
        .expect("Failed to fetch splat data")
        .to_vec()
        .expect("Failed to get means");
    bounds_from_pos(percentile, &means)
}

impl SplatTrainer {
    pub fn new(config: &TrainConfig, device: &WgpuDevice, bounds: BoundingBox) -> Self {
        let decay =
            (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_train_iters as f64);
        let lr_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay);

        let decay =
            (config.lr_scale_end / config.lr_scale).powf(1.0 / config.total_train_iters as f64);
        let lr_scale = ExponentialLrSchedulerConfig::new(config.lr_scale, decay);

        const SSIM_WINDOW_SIZE: usize = 11; // Could be configurable but meh, rather keep consistent.
        let ssim = (config.ssim_weight > 0.0).then(|| Ssim::new(SSIM_WINDOW_SIZE, 3, device));

        Self {
            config: config.clone(),
            sched_mean: lr_mean.init().expect("Mean lr schedule must be valid."),
            sched_scale: lr_scale.init().expect("Scale lr schedule must be valid."),
            optim: None,
            refine_record: None,
            ssim,
            bounds,
            #[cfg(not(target_family = "wasm"))]
            lpips: (config.lpips_loss_weight > 0.0).then(|| lpips::load_vgg_lpips(device)),
        }
    }

    pub async fn step(
        &mut self,
        batch: SceneBatch,
        splats: Splats<DiffBackend>,
    ) -> (Splats<DiffBackend>, TrainStepStats<MainBackend>) {
        let mut splats = splats;

        let [img_h, img_w, _] = batch.img_tensor.shape.dims();
        let camera = batch.camera.clone();

        // Upload tensor early.
        let device = splats.device();
        let has_alpha = batch.has_alpha();
        let gt_tensor = Tensor::from_data(batch.img_tensor, &device);

        // Forward pass - render splats asynchronously.
        // Clone splats to avoid holding references across the await.
        let img_size = glam::uvec2(img_w as u32, img_h as u32);

        let base = &self.config.background_color;
        let base_bg = glam::Vec3::new(base[0], base[1], base[2]);
        let background = sample_background_color(base_bg, self.config.background_noise_strength);

        let diff_out = render_splats(splats.clone(), &camera, img_size, background)
            .instrument(trace_span!("Forward"))
            .await;

        let pred_image = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));
        let render_aux = diff_out.render_aux;
        let refine_weight_holder = diff_out.refine_weight_holder;

        let median_scale = self.bounds.median_size();
        let num_visible = render_aux.get_num_visible();

        let pred_rgb = pred_image.clone().slice(s![.., .., 0..3]);

        // For images with alpha, composite GT on the same background.
        let gt_rgb = if has_alpha && background != glam::Vec3::ZERO {
            let gt_rgb = gt_tensor.clone().slice(s![.., .., 0..3]);
            let gt_alpha = gt_tensor.clone().slice(s![.., .., 3..4]);
            let bg_3d: Tensor<DiffBackend, 3> = Tensor::<DiffBackend, 1>::from_floats(
                [background.x, background.y, background.z].as_slice(),
                &device,
            )
            .reshape([1, 1, 3]);
            gt_rgb + (1.0 - gt_alpha) * bg_3d
        } else {
            gt_tensor.clone().slice(s![.., .., 0..3])
        };

        let visible: Tensor<Autodiff<MainBackend>, 1> =
            Tensor::from_primitive(TensorPrimitive::Float(render_aux.visible));

        let loss = trace_span!("Calculate losses").in_scope(|| {
            let l1_rgb = (pred_rgb.clone() - gt_rgb.clone()).abs();

            let total_err = if let Some(ssim) = &self.ssim {
                let ssim_err = ssim.ssim(pred_rgb.clone(), gt_rgb.clone());
                l1_rgb * (1.0 - self.config.ssim_weight) - (ssim_err * self.config.ssim_weight)
            } else {
                l1_rgb
            };

            let total_err = if has_alpha {
                let alpha_input = gt_tensor.clone().slice(s![.., .., 3..4]);

                if batch.alpha_mode == AlphaMode::Masked {
                    total_err * alpha_input
                } else {
                    let pred_alpha = pred_image.clone().slice(s![.., .., 3..4]);
                    total_err + (alpha_input - pred_alpha).abs() * self.config.match_alpha_weight
                }
            } else {
                total_err
            };

            let loss = total_err.mean();

            // TODO: Support masked lpips.
            #[cfg(not(target_family = "wasm"))]
            let loss = if let Some(lpips) = &self.lpips {
                loss + lpips.lpips(pred_rgb.unsqueeze_dim(0), gt_rgb.unsqueeze_dim(0))
                    * self.config.lpips_loss_weight
            } else {
                loss
            };

            loss
        });

        let mut grads = trace_span!("Backward pass").in_scope(|| loss.backward());

        #[cfg(any(feature = "debug-validation", test))]
        {
            use brush_render::validation::validate_gradient;
            if let Some(g) = splats.transforms.grad(&grads) {
                validate_gradient(g, "transforms").await;
            }
            if let Some(g) = splats.sh_coeffs.grad(&grads) {
                validate_gradient(g, "sh_coeffs").await;
            }
            if let Some(g) = splats.raw_opacities.grad(&grads) {
                validate_gradient(g, "raw_opacity").await;
            }
        }

        let (lr_mean, lr_rotation, lr_scale, lr_coeffs, lr_opac) = (
            self.sched_mean.step() * median_scale as f64,
            self.config.lr_rotation,
            // Scale is relative to the scene scale, but the exp() activation function
            // means "offsetting" all values also solves the learning rate scaling.
            self.sched_scale.step(),
            self.config.lr_coeffs_dc,
            self.config.lr_opac,
        );

        let optimizer = self.optim.get_or_insert_with(|| {
            let sh_degree = splats.sh_degree();

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

        // Update per-component LR scaling for the transforms param.
        // transforms layout: means(3) + rotations(4) + log_scales(3)
        // We use base_lr=1.0 and encode actual LRs in the scaling tensor.
        {
            let lr_values: [f32; 10] = [
                lr_mean as f32,
                lr_mean as f32,
                lr_mean as f32,
                lr_rotation as f32,
                lr_rotation as f32,
                lr_rotation as f32,
                lr_rotation as f32,
                lr_scale as f32,
                lr_scale as f32,
                lr_scale as f32,
            ];
            let transform_scaling =
                Tensor::<MainBackend, 1>::from_floats(lr_values.as_slice(), &device)
                    .reshape([1, 10]);

            let mut record = optimizer.to_record();
            let existing = record.remove(&splats.transforms.id);
            let momentum = existing.and_then(|r| r.into_state::<2>().momentum);
            record.insert(
                splats.transforms.id,
                AdaptorRecord::from_state(AdamState {
                    momentum,
                    scaling: Some(transform_scaling),
                }),
            );
            *optimizer = create_default_optimizer().load_record(record);
        }

        splats = trace_span!("Optimizer step").in_scope(|| {
            splats = trace_span!("Transforms step").in_scope(|| {
                let grad_transforms =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.transforms.id]);
                optimizer.step(1.0, splats, grad_transforms)
            });
            splats = trace_span!("SH Coeffs step").in_scope(|| {
                let grad_coeff =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);
                optimizer.step(lr_coeffs, splats, grad_coeff)
            });
            splats = trace_span!("Opacity step").in_scope(|| {
                let grad_opac =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.raw_opacities.id]);
                optimizer.step(lr_opac, splats, grad_opac)
            });
            splats
        });

        trace_span!("Housekeeping").in_scope(|| {
            // Get the xy gradient norm from the dummy tensor.
            let refine_weight = refine_weight_holder
                .grad_remove(&mut grads)
                .expect("XY gradients need to be calculated.");
            let device = splats.device();
            let num_splats = splats.num_splats();
            let record = self
                .refine_record
                .get_or_insert_with(|| RefineRecord::new(num_splats, &device));

            record.gather_stats(refine_weight, visible.clone().inner());
        });

        let device = splats.device();
        // Add random noise. Only do this in the growth phase, otherwise
        // let the splats settle in without noise, not much point in exploring regions anymore.
        let inv_opac: Tensor<_, 1> = 1.0 - splats.opacities();
        let noise_weight = inv_opac.inner().powi_scalar(150.0).clamp(0.0, 1.0) * visible.inner();
        let noise_weight = noise_weight.unsqueeze_dim(1);
        let samples = Tensor::random(
            [splats.num_splats() as usize, 3],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Only allow noised gaussians to travel at most the entire extent of the current bounds.
        let max_noise = median_scale;
        // Could scale by train time, but, the mean_lr already heavily decays.
        let noise_weight = noise_weight * (lr_mean as f32 * self.config.mean_noise_weight);
        // Add noise to the means portion (cols 0..3) of transforms.
        splats.transforms = splats.transforms.map(|t| {
            let noise = (samples * noise_weight).clamp(-max_noise, max_noise);
            let inner = t.inner();
            let noised_means = inner.clone().slice(s![.., 0..3]) + noise;
            Tensor::from_inner(inner.slice_assign(s![.., 0..3], noised_means)).require_grad()
        });

        let stats = TrainStepStats {
            pred_image: pred_image.inner(),
            num_visible,
            loss: loss.inner(),
            lr_mean,
            lr_rotation,
            lr_scale,
            lr_coeffs,
            lr_opac,
        };

        (splats, stats)
    }

    pub async fn refine(
        &mut self,
        iter: u32,
        splats: Splats<MainBackend>,
    ) -> (Splats<MainBackend>, RefineStats) {
        let device = splats.device();
        let client = WgpuRuntime::client(&device);

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");

        let max_allowed_bounds = self.bounds.extent.max_element() * 100.0;

        // If not refining, update splat to step with gradients applied.
        // Prune dead splats. This ALWAYS happen even if we're not "refining" anymore.
        let mut record = self
            .optim
            .take()
            .expect("Can only refine after optimizer is initialized")
            .to_record();
        let alpha_mask = splats.opacities().lower_elem(MIN_OPACITY);
        let scales = splats.scales();

        let scale_small = scales.clone().lower_elem(1e-10).any_dim(1).squeeze_dim(1);
        let scale_big = scales
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<_, 1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let splat_dists = (splats.means() - bound_center).abs();
        let bound_mask = splat_dists
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);
        let prune_mask = alpha_mask
            .bool_or(scale_small)
            .bool_or(scale_big)
            .bool_or(bound_mask);

        let (mut splats, refiner, pruned_count) =
            prune_points(splats, &mut record, refiner, prune_mask).await;
        let mut split_inds = HashSet::new();

        // Always replace dead gaussians, so that the pruned budget is reused.
        if pruned_count > 0 {
            // Sample weighted by opacity from splat visible during optimization.
            let resampled_weights = splats.opacities() * refiner.vis_mask().float();

            let resampled_weights = resampled_weights
                .into_data_async()
                .await
                .expect("Failed to get weights")
                .into_vec::<f32>()
                .expect("Failed to read weights");
            let resampled_inds = multinomial_sample(&resampled_weights, pruned_count);
            split_inds.extend(resampled_inds);
        }

        if iter < self.config.growth_stop_iter {
            let above_threshold = refiner.above_threshold(self.config.growth_grad_threshold);

            let threshold_count = above_threshold
                .clone()
                .int()
                .sum()
                .into_scalar_async()
                .await
                .expect("Failed to get threshold") as u32;

            let grow_count =
                (threshold_count as f32 * self.config.growth_select_fraction).round() as u32;

            let sample_high_grad = grow_count.saturating_sub(pruned_count);

            // Only grow to the max nr. of splats.
            let cur_splats = splats.num_splats() + split_inds.len() as u32;
            let grow_count = sample_high_grad.min(self.config.max_splats - cur_splats);

            // If still growing, sample from indices which are over the threshold.
            if grow_count > 0 {
                let weights = above_threshold.float() * refiner.refine_weight_norm;
                let weights = weights
                    .into_data_async()
                    .await
                    .expect("Failed to get weights")
                    .into_vec::<f32>()
                    .expect("Failed to read weights");
                let growth_inds = multinomial_sample(&weights, grow_count);
                split_inds.extend(growth_inds);
            }
        }

        let refine_count = split_inds.len();
        splats = self.refine_splats(&device, record, splats, split_inds, iter);

        // Update current bounds based on the splats.
        self.bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        client.memory_cleanup();

        let splat_count = splats.num_splats();

        (
            splats,
            RefineStats {
                num_added: refine_count as u32,
                num_pruned: pruned_count,
                total_splats: splat_count,
            },
        )
    }

    fn refine_splats(
        &mut self,
        device: &WgpuDevice,
        mut record: HashMap<ParamId, AdaptorRecord<AdamScaled, DiffBackend>>,
        mut splats: Splats<MainBackend>,
        split_inds: HashSet<i32>,
        iter: u32,
    ) -> Splats<MainBackend> {
        let refine_count = split_inds.len();

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(split_inds.into_iter().collect(), [refine_count]),
                device,
            );

            let cur_transforms = splats.transforms.val().select(0, refine_inds.clone());
            let cur_means = cur_transforms.clone().slice(s![.., 0..3]);
            let cur_rots_raw = cur_transforms.clone().slice(s![.., 3..7]);
            let magnitudes = Tensor::clamp_min(
                Tensor::sum_dim(cur_rots_raw.clone().powi_scalar(2), 1).sqrt(),
                1e-32,
            );
            let cur_rots = cur_rots_raw / magnitudes;
            let cur_log_scale = cur_transforms.slice(s![.., 7..10]);
            let cur_coeff = splats.sh_coeffs.val().select(0, refine_inds.clone());
            let cur_raw_opac = splats.raw_opacities.val().select(0, refine_inds.clone());

            // The amount to offset the scale and opacity should maybe depend on how far away we have sampled these gaussians,
            // but a fixed amount seems to work ok. The only note is that divide by _less_ than SQRT(2) seems to exponentially
            // blow up, as more 'mass' is added each refine.
            // let scale_div = Tensor::ones_like(&cur_log_scale) * SQRT_2.ln();
            //
            let cur_scales = cur_log_scale.clone().exp();

            let cur_opac = sigmoid(cur_raw_opac.clone());
            let inv_opac: Tensor<_, 1> = 1.0 - cur_opac;
            let new_opac: Tensor<_, 1> = 1.0 - inv_opac.sqrt();
            let new_raw_opac = inv_sigmoid(new_opac.clamp(MIN_OPACITY, 1.0 - MIN_OPACITY));
            let new_scales = scale_down_largest_dim(cur_scales.clone(), 0.5);
            let new_log_scales = new_scales.log();

            // Move in direction of scaling axis.
            let samples = quaternion_vec_multiply(
                cur_rots.clone(),
                Tensor::random([refine_count, 1], Distribution::Normal(0.0, 1.0), device)
                    * cur_scales,
            );

            // Shrink & offset existing splats.

            // Scatter into transforms: build a [refine_count, 10] update tensor
            // with means offset in cols 0..3 and log_scales difference in cols 7..10
            let refine_inds_10 = refine_inds.clone().unsqueeze_dim(1).repeat_dim(1, 10);
            let scale_difference = new_log_scales.clone() - cur_log_scale;

            splats.transforms = splats.transforms.map(|t| {
                let dev = t.device();
                let mut update = Tensor::zeros([refine_count, 10], &dev);
                // Place -samples in means columns (0..3)
                update = update.slice_assign(s![.., 0..3], -samples.clone());
                // Place scale difference in log_scales columns (7..10)
                update = update.slice_assign(s![.., 7..10], scale_difference.clone());
                t.scatter(0, refine_inds_10.clone(), update, IndexingUpdateOp::Add)
            });
            splats.raw_opacities = splats.raw_opacities.map(|m| {
                let difference = new_raw_opac.clone() - cur_raw_opac.clone();
                m.scatter(0, refine_inds.clone(), difference, IndexingUpdateOp::Add)
            });

            // Concatenate new splats.
            let sh_dim = splats.sh_coeffs.dims()[1];
            // Build new transforms row: means(3) + rotations(4) + log_scales(3)
            let new_transforms =
                Tensor::cat(vec![cur_means + samples, cur_rots, new_log_scales], 1);
            splats = map_splats_and_opt(
                splats,
                &mut record,
                |x| Tensor::cat(vec![x, new_transforms], 0),
                |x| Tensor::cat(vec![x, cur_coeff], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 10], device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, sh_dim, 3], device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count], device)], 0),
            );
        }

        let train_t = (iter as f32 / self.config.total_train_iters as f32).clamp(0.0, 1.0);
        let t_shrink_strength = 1.0 - train_t;
        let minus_opac = self.config.opac_decay * t_shrink_strength;
        let scale_scaling = 1.0 - self.config.scale_decay * t_shrink_strength;

        // Lower opacity slowly over time.
        splats.raw_opacities = splats.raw_opacities.map(|f| {
            let new_opac = sigmoid(f) - minus_opac;
            inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))
        });

        // Decay log_scales (cols 7..10) within transforms
        splats.transforms = splats.transforms.map(|t| {
            let new_log_scales = (t.clone().slice(s![.., 7..10]).exp() * scale_scaling).log();
            t.slice_assign(s![.., 7..10], new_log_scales)
        });

        self.optim = Some(create_default_optimizer().load_record(record));

        splats
    }
}

fn map_splats_and_opt(
    mut splats: Splats<MainBackend>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, DiffBackend>>,
    map_transforms: impl FnOnce(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_coeffs: impl FnOnce(Tensor<MainBackend, 3>) -> Tensor<MainBackend, 3>,
    map_opac: impl FnOnce(Tensor<MainBackend, 1>) -> Tensor<MainBackend, 1>,

    map_opt_transforms: impl Fn(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_opt_coeffs: impl Fn(Tensor<MainBackend, 3>) -> Tensor<MainBackend, 3>,
    map_opt_opac: impl Fn(Tensor<MainBackend, 1>) -> Tensor<MainBackend, 1>,
) -> Splats<MainBackend> {
    splats.transforms = splats.transforms.map(map_transforms);
    map_opt(splats.transforms.id, record, &map_opt_transforms);
    splats.sh_coeffs = splats.sh_coeffs.map(map_coeffs);
    map_opt(splats.sh_coeffs.id, record, &map_opt_coeffs);
    splats.raw_opacities = splats.raw_opacities.map(map_opac);
    map_opt(splats.raw_opacities.id, record, &map_opt_opac);
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
async fn prune_points(
    mut splats: Splats<MainBackend>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, DiffBackend>>,
    mut refiner: RefineRecord<MainBackend>,
    prune: Tensor<MainBackend, 1, Bool>,
) -> (Splats<MainBackend>, RefineRecord<MainBackend>, u32) {
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
        let valid_inds = valid_inds.squeeze_dim(1);
        splats = map_splats_and_opt(
            splats,
            record,
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

fn scale_down_largest_dim<B: Backend>(scales: Tensor<B, 2>, factor: f32) -> Tensor<B, 2> {
    // Find the maximum values along dimension 1 (keeping dimensions for broadcasting)
    let max_mask = scales.clone().equal(scales.clone().max_dim(1));
    let scale = Tensor::ones_like(&scales).mask_fill(max_mask, factor);
    scales.mul(scale)
}

/// Sample a background color: base + uniform noise in [-strength, +strength], clamped to [0, 1].
fn sample_background_color(base: glam::Vec3, strength: f32) -> glam::Vec3 {
    use rand::RngExt as _;
    let mut rng = rand::rng();
    let noise = glam::Vec3::new(
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
    );
    (base + noise).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}
