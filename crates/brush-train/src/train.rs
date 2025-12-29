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
use glam::Vec3;
use hashbrown::{HashMap, HashSet};
use tracing::trace_span;

const MIN_OPACITY: f32 = 1.0 / 255.0;
const BOUND_PERCENTILE: f32 = 0.8;

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
        .means
        .val()
        .into_data_async()
        .await
        .expect("Failed to fetch splat data")
        .to_vec()
        .expect("Failed to get means");
    bounds_from_pos(percentile, &means)
}

impl SplatTrainer {
    pub async fn new<B: Backend>(
        config: &TrainConfig,
        device: &WgpuDevice,
        init_splats: Splats<B>,
    ) -> Self {
        let decay = (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_steps as f64);
        let lr_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay);

        let decay = (config.lr_scale_end / config.lr_scale).powf(1.0 / config.total_steps as f64);
        let lr_scale = ExponentialLrSchedulerConfig::new(config.lr_scale, decay);

        const SSIM_WINDOW_SIZE: usize = 11; // Could be configurable but meh, rather keep consistent.
        let ssim = (config.ssim_weight > 0.0).then(|| Ssim::new(SSIM_WINDOW_SIZE, 3, device));

        let bounds = get_splat_bounds(init_splats.clone(), BOUND_PERCENTILE).await;

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

    pub fn step(
        &mut self,
        batch: SceneBatch,
        splats: Splats<DiffBackend>,
    ) -> (Splats<DiffBackend>, TrainStepStats<MainBackend>) {
        let _span = trace_span!("Train step").entered();

        let mut splats = splats;

        let [img_h, img_w, _] = batch.img_tensor.shape.clone().try_into().unwrap();
        let camera = &batch.camera;

        // Upload tensor early.
        let device = splats.device();
        let has_alpha = batch.has_alpha();
        let gt_tensor = Tensor::from_data(batch.img_tensor, &device);

        let (pred_image, aux, refine_weight_holder) = trace_span!("Forward").in_scope(|| {
            // Could generate a random background color, but so far
            // results just seem worse.
            let background = Vec3::ZERO;

            let diff_out = render_splats(
                &splats,
                camera,
                glam::uvec2(img_w as u32, img_h as u32),
                background,
            );

            let img = Tensor::from_primitive(TensorPrimitive::Float(diff_out.img));

            #[cfg(any(feature = "debug-validation", test))]
            {
                splats.validate_values();
                diff_out.aux.validate_values();
            }

            (img, diff_out.aux, diff_out.refine_weight_holder)
        });

        let median_scale = self.bounds.median_size();
        let num_visible = aux.num_visible().inner();
        let num_intersections = aux.num_intersections().inner();
        let pred_rgb = pred_image.clone().slice(s![.., .., 0..3]);
        let gt_rgb = gt_tensor.clone().slice(s![.., .., 0..3]);

        let visible: Tensor<Autodiff<MainBackend>, 1> =
            Tensor::from_primitive(TensorPrimitive::Float(aux.visible));

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
            brush_render::validation::validate_splat_gradients(&splats, &grads);
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

        splats = trace_span!("Optimizer step").in_scope(|| {
            splats = trace_span!("SH Coeffs step").in_scope(|| {
                let grad_coeff =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.sh_coeffs.id]);
                optimizer.step(lr_coeffs, splats, grad_coeff)
            });
            splats = trace_span!("Rotation step").in_scope(|| {
                let grad_rot =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.rotations.id]);
                optimizer.step(lr_rotation, splats, grad_rot)
            });
            splats = trace_span!("Scale step").in_scope(|| {
                let grad_scale =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.log_scales.id]);
                optimizer.step(lr_scale, splats, grad_scale)
            });
            splats = trace_span!("Mean step").in_scope(|| {
                let grad_means =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.means.id]);
                optimizer.step(lr_mean, splats, grad_means)
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
        splats.means = splats.means.map(|m| {
            Tensor::from_inner(m.inner() + (samples * noise_weight).clamp(-max_noise, max_noise))
                .require_grad()
        });

        let stats = TrainStepStats {
            pred_image: pred_image.inner(),
            num_visible,
            num_intersections,
            loss: loss.inner(),
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
        splats: Splats<DiffBackend>,
    ) -> (Splats<DiffBackend>, Option<RefineStats>) {
        let train_t = (iter as f32 / self.config.total_steps as f32).clamp(0.0, 1.0);

        if iter == 0 || !iter.is_multiple_of(self.config.refine_every) || train_t > 0.95 {
            return (splats, None);
        }

        let device = splats.means.device();
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
        let alpha_mask = splats.opacities().inner().lower_elem(MIN_OPACITY);
        let scales = splats.scales().inner();

        let scale_small = scales.clone().lower_elem(1e-10).any_dim(1).squeeze_dim(1);
        let scale_big = scales
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<_, 1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let splat_dists = (splats.means.val().inner() - bound_center).abs();
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

        // Replace dead gaussians.
        if pruned_count > 0 {
            // Sample weighted by opacity from splat visible during optimization.
            let resampled_weights = splats.opacities().inner() * refiner.vis_mask().float();
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
        splats = self.refine_splats(&device, record, splats, split_inds, train_t);

        // Update current bounds based on the splats.
        self.bounds = get_splat_bounds(splats.clone(), BOUND_PERCENTILE).await;
        client.memory_cleanup();

        (
            splats,
            Some(RefineStats {
                num_added: refine_count as u32,
                num_pruned: pruned_count,
            }),
        )
    }

    fn refine_splats(
        &mut self,
        device: &WgpuDevice,
        mut record: HashMap<ParamId, AdaptorRecord<AdamScaled, DiffBackend>>,
        mut splats: Splats<DiffBackend>,
        split_inds: HashSet<i32>,
        train_t: f32,
    ) -> Splats<DiffBackend> {
        let refine_count = split_inds.len();

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(split_inds.into_iter().collect(), [refine_count]),
                device,
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
                .raw_opacities
                .val()
                .inner()
                .select(0, refine_inds.clone());

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

            // Scatter needs [N, 3] indices for means and scales.
            let refine_inds_3 = refine_inds.clone().unsqueeze_dim(1).repeat_dim(1, 3);

            splats.means = splats.means.map(|m| {
                let new_means = m.inner().scatter(
                    0,
                    refine_inds_3.clone(),
                    -samples.clone(),
                    IndexingUpdateOp::Add,
                );
                Tensor::from_inner(new_means).require_grad()
            });
            splats.log_scales = splats.log_scales.map(|s| {
                let difference = new_log_scales.clone() - cur_log_scale.clone();
                let new_scales =
                    s.inner()
                        .scatter(0, refine_inds_3.clone(), difference, IndexingUpdateOp::Add);
                Tensor::from_inner(new_scales).require_grad()
            });
            splats.raw_opacities = splats.raw_opacities.map(|m| {
                let difference = new_raw_opac.clone() - cur_raw_opac.clone();
                let new_opacities =
                    m.inner()
                        .scatter(0, refine_inds.clone(), difference, IndexingUpdateOp::Add);
                Tensor::from_inner(new_opacities).require_grad()
            });

            // Concatenate new splats.
            let sh_dim = splats.sh_coeffs.dims()[1];
            splats = map_splats_and_opt(
                splats,
                &mut record,
                |x| Tensor::cat(vec![x, cur_means + samples], 0),
                |x| Tensor::cat(vec![x, cur_rots], 0),
                |x| Tensor::cat(vec![x, new_log_scales], 0),
                |x| Tensor::cat(vec![x, cur_coeff], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 3], device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 4], device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, 3], device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count, sh_dim, 3], device)], 0),
                |x| Tensor::cat(vec![x, Tensor::zeros([refine_count], device)], 0),
            );
        }

        let t_shrink_strength = 1.0 - train_t;
        let minus_opac = self.config.opac_decay * t_shrink_strength;
        let scale_scaling = 1.0 - self.config.scale_decay * t_shrink_strength;

        // Lower opacity slowly over time.
        splats.raw_opacities = splats.raw_opacities.map(|f| {
            let new_opac = sigmoid(f.inner()) - minus_opac;
            Tensor::from_inner(inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))).require_grad()
        });

        splats.log_scales = splats.log_scales.map(|f| {
            let new_scale = f.inner().exp() * scale_scaling;
            Tensor::from_inner(new_scale.log()).require_grad()
        });

        self.optim = Some(create_default_optimizer().load_record(record));
        splats
    }
}

fn map_splats_and_opt(
    mut splats: Splats<DiffBackend>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, DiffBackend>>,
    map_mean: impl FnOnce(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_rotation: impl FnOnce(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_scale: impl FnOnce(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_coeffs: impl FnOnce(Tensor<MainBackend, 3>) -> Tensor<MainBackend, 3>,
    map_opac: impl FnOnce(Tensor<MainBackend, 1>) -> Tensor<MainBackend, 1>,

    map_opt_mean: impl Fn(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_opt_rotation: impl Fn(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_opt_scale: impl Fn(Tensor<MainBackend, 2>) -> Tensor<MainBackend, 2>,
    map_opt_coeffs: impl Fn(Tensor<MainBackend, 3>) -> Tensor<MainBackend, 3>,
    map_opt_opac: impl Fn(Tensor<MainBackend, 1>) -> Tensor<MainBackend, 1>,
) -> Splats<DiffBackend> {
    splats.means = splats
        .means
        .map(|x| Tensor::from_inner(map_mean(x.inner())).require_grad());
    map_opt(splats.means.id, record, &map_opt_mean);

    splats.rotations = splats
        .rotations
        .map(|x| Tensor::from_inner(map_rotation(x.inner())).require_grad());
    map_opt(splats.rotations.id, record, &map_opt_rotation);

    splats.log_scales = splats
        .log_scales
        .map(|x| Tensor::from_inner(map_scale(x.inner())).require_grad());
    map_opt(splats.log_scales.id, record, &map_opt_scale);

    splats.sh_coeffs = splats
        .sh_coeffs
        .map(|x| Tensor::from_inner(map_coeffs(x.inner())).require_grad());
    map_opt(splats.sh_coeffs.id, record, &map_opt_coeffs);

    splats.raw_opacities = splats
        .raw_opacities
        .map(|x| Tensor::from_inner(map_opac(x.inner())).require_grad());
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
    mut splats: Splats<DiffBackend>,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled, DiffBackend>>,
    mut refiner: RefineRecord<MainBackend>,
    prune: Tensor<MainBackend, 1, Bool>,
) -> (Splats<DiffBackend>, RefineRecord<MainBackend>, u32) {
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
