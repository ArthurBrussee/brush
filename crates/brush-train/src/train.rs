use crate::{
    adam_scaled::{AdamScaled, AdamScaledConfig, AdamState},
    config::TrainConfig,
    msg::{RefineStats, TrainStepStats},
    multinomial::multinomial_sample,
    quat_vec::quaternion_vec_multiply,
    splat_init::bounds_from_pos,
    stats::RefineRecord,
};

use brush_dataset::scene::SceneBatch;
use brush_loss::{ImageLossConfig, image_loss};
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, bounding_box::BoundingBox, sh::sh_coeffs_for_degree};
use brush_render_bwd::render_splats;
use burn::{
    backend::wgpu::{AutoCompiler, WgpuDevice, WgpuRuntime},
    lr_scheduler::{
        LrScheduler,
        exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    },
    module::{AutodiffModule, ParamId},
    optim::{GradientsParams, Optimizer, adaptor::OptimizerAdaptor, record::AdaptorRecord},
    tensor::{
        Bool, Device, Distribution, IndexingUpdateOp, Int, Tensor, TensorData, activation::sigmoid,
        s,
    },
};

use burn_cubecl::cubecl::Runtime;
use hashbrown::{HashMap, HashSet};
use tracing::{Instrument, trace_span};

pub const BOUND_PERCENTILE: f32 = 0.8;

const MIN_OPACITY: f32 = 1.0 / 255.0;

type OptimizerType = OptimizerAdaptor<AdamScaled, Splats>;

pub struct SplatTrainer {
    config: TrainConfig,
    sched_mean: ExponentialLrScheduler,
    refine_record: Option<RefineRecord>,
    optim: Option<OptimizerType>,
    ssim_enabled: bool,
    bounds: BoundingBox,
    step_count: u32,
    max_sh_degree: u32,
    #[cfg(not(target_family = "wasm"))]
    lpips: Option<lpips::LpipsModel>,
}

fn inv_sigmoid(x: Tensor<1>) -> Tensor<1> {
    (x.clone() / (1.0f32 - x)).log()
}

fn create_optimizer_from_config() -> OptimizerType {
    AdamScaledConfig::new().with_epsilon(1e-15).init()
}

pub async fn get_splat_bounds(splats: Splats, percentile: f32) -> BoundingBox {
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
    #[allow(unused_variables)]
    pub fn new(config: &TrainConfig, device: &Device, bounds: BoundingBox) -> Self {
        let decay =
            (config.lr_mean_end / config.lr_mean).powf(1.0 / config.total_train_iters as f64);
        let lr_mean = ExponentialLrSchedulerConfig::new(config.lr_mean, decay);

        let ssim_enabled = config.ssim_weight > 0.0;

        // Push the in-kernel screen-area regulariser config into the
        // brush-render global so the host code's ProjectUniforms
        // construction picks it up and the backward kernel applies the
        // gradient contribution to v_cov2d.
        brush_render::set_screen_area_penalty(
            config.screen_area_penalty,
            config.screen_area_threshold,
        );

        // Growth is gated on the global iter. LOD phases run past
        // total_train_iters but their refines should never grow — clamp
        // here so growth_stop is never effectively past end-of-training.
        let mut config = config.clone();
        config.growth_stop_iter = config.growth_stop_iter.min(config.total_train_iters);

        #[cfg(not(target_family = "wasm"))]
        let lpips = (config.lpips_loss_weight > 0.0).then(|| lpips::load_vgg_lpips(device));

        Self {
            config,
            sched_mean: lr_mean.init().expect("Mean lr schedule must be valid."),
            optim: None,
            refine_record: None,
            ssim_enabled,
            bounds,
            step_count: 0,
            max_sh_degree: 0,
            #[cfg(not(target_family = "wasm"))]
            lpips,
        }
    }

    pub async fn step(&mut self, batch: SceneBatch, splats: Splats) -> (Splats, TrainStepStats) {
        let mut splats = splats;

        // Track max SH degree from the first splats we see.
        if self.step_count == 0 {
            self.max_sh_degree = splats.sh_degree();
        }
        self.step_count += 1;

        let [img_h, img_w] = batch.img_size();
        let camera = batch.camera;

        let device = splats.device();
        let has_alpha = batch.has_alpha;
        // GT lives on the GPU as packed `[H, W]` u32 (RGBA u8). All mixing
        // (bg compositing, alpha matching, mask) is folded into the loss
        // kernels; no f32 GT image is ever materialised here.
        // GT is pure data — never differentiated. Build it on the inner
        // backend so it doesn't inherit the autodiff device's residual
        // checkpointing flag (the LPIPS `unpack_gt_rgb` path, via
        // `unwrap_wgpu_int`, expects a clean Wgpu tensor).
        let gt_packed: Tensor<2, Int> =
            Tensor::from_data(batch.img_packed, &device.clone().inner());
        let img_size = glam::uvec2(img_w as u32, img_h as u32);
        let base = &self.config.background_color;
        let base_bg = glam::Vec3::new(base[0], base[1], base[2]);
        let background = sample_background_color(base_bg, self.config.background_noise_strength);

        let median_scale = self.bounds.median_size();

        let (mut grads, visible, num_visible, loss_inner) = {
            let render_input = splats.clone();
            let diff_out = render_splats(render_input, &camera, img_size, background)
                .instrument(trace_span!("Forward"))
                .await;

            let pred_image = diff_out.img;
            let refine_weight_holder = diff_out.refine_weight_holder;
            let visible = diff_out.visible;
            let max_radius = diff_out.max_radius;

            // RGB loss is `(1 - w) * L1 + (-w) * SSIM` per pixel. Bg
            // compositing always runs in the kernel; for synthesised opaque
            // alpha or zero bg it's a no-op. Mask multiplies the loss-map
            // by `gt.a`; for synthesised opaque alpha that's a no-op too.
            // Alpha matching needs a real alpha source (synthesised
            // a = 1 would pull predicted alpha to fully opaque); we feed
            // `pred` with 4 channels and the kernel's `c == 3` workgroup
            // emits `|pred.a - gt.a|` into the alpha channel.
            let masked_alpha = batch.alpha_mode == AlphaMode::Masked;
            let (l1_w, ssim_w) = if self.ssim_enabled {
                (1.0 - self.config.ssim_weight, -self.config.ssim_weight)
            } else {
                (1.0, 0.0)
            };
            let do_alpha_match = has_alpha && !masked_alpha && self.config.match_alpha_weight > 0.0;
            // Only composite when there's a real alpha channel and a non-zero
            // bg to mix in; the kernel skips the per-pixel `(1-a)*bg` math
            // entirely when this is None.
            let composite_bg = (has_alpha && background != glam::Vec3::ZERO).then_some(background);
            let cfg = ImageLossConfig {
                l1_weight: l1_w,
                ssim_weight: ssim_w,
                composite_bg,
                mask: masked_alpha,
            };
            let pred_for_loss = if do_alpha_match {
                pred_image.clone()
            } else {
                pred_image.clone().slice(s![.., .., 0..3])
            };
            let loss_map = image_loss(pred_for_loss, gt_packed.clone(), cfg);

            // `loss` is only reassigned by the LPIPS path below, which is
            // compiled out on wasm — so `mut` is unused there.
            #[cfg_attr(target_family = "wasm", allow(unused_mut))]
            let mut loss = if do_alpha_match {
                let rgb = loss_map.clone().slice(s![.., .., 0..3]).mean();
                let alpha = loss_map.slice(s![.., .., 3..4]).mean();
                rgb + alpha * self.config.match_alpha_weight
            } else {
                loss_map.mean()
            };

            // LPIPS still needs an f32 RGB tensor for VGG. Materialising it
            // here costs ~99 MB at 4K, only when LPIPS is enabled.
            #[cfg(not(target_family = "wasm"))]
            if let Some(lpips) = &self.lpips {
                let gt_rgb = brush_loss::unpack_gt_rgb(gt_packed.clone(), composite_bg);
                let gt_rgb_diff: Tensor<3> = Tensor::from_inner(gt_rgb);
                loss = loss
                    + lpips.lpips(
                        pred_image.clone().slice(s![.., .., 0..3]).unsqueeze_dim(0),
                        gt_rgb_diff.unsqueeze_dim(0),
                    ) * self.config.lpips_loss_weight;
            }

            // Strip the autodiff graph off the loss so consumers can read the
            // scalar later without keeping the backward pass alive.
            let loss_inner = loss.clone().inner();
            let mut grads = splats.bwd_validate(loss).await;

            trace_span!("Housekeeping").in_scope(|| {
                // Refine state accumulates on the inner (non-autodiff) device
                // so we can mix it with `.inner()`-stripped gradients/aux
                // without crossing backends. `detach_autodiff` also clears
                // the residual `checkpointing` flag that bare `.inner()`
                // leaves behind (see `brush_render::burn_glue`).
                use brush_render::burn_glue::detach_autodiff;
                let refine_weight = refine_weight_holder
                    .grad_remove(&mut grads)
                    .expect("XY gradients need to be calculated.");
                let device = splats.device().inner();
                let record = self
                    .refine_record
                    .get_or_insert_with(|| RefineRecord::new(splats.num_splats(), &device));
                // `visible` / `max_radius` already arrive on the inner backend;
                // only the freshly-extracted `refine_weight` gradient needs the
                // autodiff stripped off.
                record.gather_stats(detach_autodiff(refine_weight), visible.clone(), max_radius);
            });

            (grads, visible, diff_out.num_visible, loss_inner)
        };

        // OptimizerAdaptor strips autodiff before calling SimpleOptimizer::step,
        // so optimizer state (scaling, momentum) lives on the inner device.
        let opt_device = device.clone().inner();
        let optimizer =
            self.optim.get_or_insert_with(|| {
                let sh_degree = splats.sh_degree();
                let num_coeffs = sh_coeffs_for_degree(sh_degree) as usize;

                // DC (band 0) uses full LR; bands 1+ are scaled down.
                let mut scales = vec![1.0f32; num_coeffs];
                let rest_scale = 1.0 / self.config.lr_coeffs_sh_scale;
                for s in &mut scales[1..] {
                    *s = rest_scale;
                }
                let sh_lr_scales = Tensor::<1>::from_floats(scales.as_slice(), &opt_device)
                    .reshape([1, num_coeffs as i32, 1]);

                create_optimizer_from_config().load_record(HashMap::from([(
                    splats.sh_coeffs.id,
                    AdaptorRecord::from_state(AdamState {
                        momentum: None,
                        scaling: Some(sh_lr_scales),
                        reduce_moment_2: true,
                    }),
                )]))
            });

        let lr_mean = self.sched_mean.step() * median_scale as f64;

        // Update per-component LR scaling for the transforms param.
        // transforms layout: means(3) + rotations(4) + log_scales(3)
        // We use base_lr=1.0 and encode actual LRs in the scaling tensor.
        //
        // TODO: Ideally we don't have to do this every step... but idk as long as mean is on a schedule not much to do!
        {
            let lr_values: [f32; 10] = [
                lr_mean as f32,
                lr_mean as f32,
                lr_mean as f32,
                self.config.lr_rotation as f32,
                self.config.lr_rotation as f32,
                self.config.lr_rotation as f32,
                self.config.lr_rotation as f32,
                self.config.lr_scale as f32,
                self.config.lr_scale as f32,
                self.config.lr_scale as f32,
            ];
            let transform_scaling =
                Tensor::<1>::from_floats(lr_values.as_slice(), &opt_device).reshape([1, 10]);
            let mut record = optimizer.to_record();
            let existing = record.remove(&splats.transforms.id);
            let momentum = existing.and_then(|r| r.into_state::<2>().momentum);
            record.insert(
                splats.transforms.id,
                AdaptorRecord::from_state(AdamState {
                    momentum,
                    scaling: Some(transform_scaling),
                    reduce_moment_2: false,
                }),
            );
            *optimizer = create_optimizer_from_config().load_record(record);
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
                optimizer.step(self.config.lr_coeffs_dc, splats, grad_coeff)
            });
            splats = trace_span!("Opacity step").in_scope(|| {
                let grad_opac =
                    GradientsParams::from_params(&mut grads, &splats, &[splats.raw_opacities.id]);
                optimizer.step(self.config.lr_opac, splats, grad_opac)
            });
            splats
        });

        // Add random noise. Only do this in the growth phase, otherwise
        // let the splats settle in without noise, not much point in exploring regions anymore.
        // The noise gate is non-differentiable bookkeeping. Read opacity from
        // the valid (inner) splats so the sigmoid never lands on the autodiff
        // graph, and `visible` is already inner — so nothing here builds a
        // node that won't get a backward pass.
        let inv_opac: Tensor<1> = 1.0 - splats.valid().opacities();
        let noise_weight = inv_opac.powi_scalar(150.0).clamp(0.0, 1.0) * visible;
        let noise_weight = noise_weight.unsqueeze_dim(1);
        // `samples` is pure data — keep it on the inner device so it can
        // multiply with the `.inner()`-stripped `noise_weight` without
        // crossing backends.
        let samples = Tensor::random(
            [splats.num_splats() as usize, 3],
            Distribution::Normal(0.0, 1.0),
            &splats.device().inner(),
        );

        // Could scale by train time, but, the mean_lr already decays over time.
        let noise_weight_means = noise_weight * (lr_mean as f32 * self.config.mean_noise_weight);

        // Add noise to the means portion (cols 0..3), and optionally scales
        // (cols 7..10) and rotations (cols 3..7).
        splats.transforms = splats.transforms.map(|t| {
            // Only allow noised gaussians to travel at most the entire extent of the current bounds.
            let noise_m = (samples * noise_weight_means).clamp(-median_scale, median_scale);
            let inner = t.inner();
            // slice + slice_assign with a clone of inner avoids holding two
            // refs across slice_assign — `inner` is consumed by slice_assign
            // and the resulting buffer is the only writer.
            let noised_means = inner.clone().slice(s![.., 0..3]) + noise_m;
            let out = inner.slice_assign(s![.., 0..3], noised_means);
            Tensor::from_inner(out).require_grad()
        });

        let stats = TrainStepStats {
            num_visible,
            lr_mean,
            lr_rotation: self.config.lr_rotation,
            lr_scale: self.config.lr_scale,
            lr_coeffs: self.config.lr_coeffs_dc,
            lr_opac: self.config.lr_opac,
            loss: loss_inner,
        };

        (splats, stats)
    }

    pub async fn refine(&mut self, iter: u32, splats: Splats) -> (Splats, RefineStats) {
        let device = splats.device();
        // `memory_cleanup` lives on the wgpu client, not on `Device`.
        let client = WgpuRuntime::<AutoCompiler>::client(&WgpuDevice::default());

        let refiner = self
            .refine_record
            .take()
            .expect("Can only refine if refine stats are initialized");

        // Log per-splat screen-size distribution so we can track how
        // many splats are visually large (and thus susceptible to the
        // "big-low-α" failure mode). Uses `max_screen_size` (fraction of
        // the smaller image dim covered by the larger 2D ellipse extent);
        // area is approximated by squaring it (upper bound — exact for
        // round splats, overestimates for needles). Cheap CPU pass on
        // ~1M floats; gated on refine cadence anyway.
        let ss_data = refiner
            .max_screen_size
            .clone()
            .into_data_async()
            .await
            .expect("Failed to read screen size")
            .into_vec::<f32>()
            .expect("Failed to read screen size vec");
        if !ss_data.is_empty() {
            let mut sorted: Vec<f32> = ss_data.iter().copied().filter(|v| v.is_finite()).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = sorted.len();
            let pct = |p: f32| sorted[((p * (n - 1) as f32) as usize).min(n - 1)];
            let n_total = n as f64;
            let n_gt_025 = ss_data.iter().filter(|v| **v > 0.25).count();
            let n_gt_010 = ss_data.iter().filter(|v| **v > 0.10).count();
            let n_gt_005 = ss_data.iter().filter(|v| **v > 0.05).count();
            // area fraction (upper bound) = max_dim^2
            let n_area_gt_005 = ss_data.iter().filter(|v| (*v * *v) > 0.05).count();
            let n_area_gt_010 = ss_data.iter().filter(|v| (*v * *v) > 0.10).count();
            log::info!(
                "screen_size iter={} n={} max_dim p50={:.4} p95={:.4} p99={:.4} max={:.4} frac>0.05={:.4} frac>0.10={:.4} frac>0.25={:.4} frac_area>0.05={:.4} frac_area>0.10={:.4}",
                iter,
                n,
                pct(0.5),
                pct(0.95),
                pct(0.99),
                pct(1.0),
                n_gt_005 as f64 / n_total,
                n_gt_010 as f64 / n_total,
                n_gt_025 as f64 / n_total,
                n_area_gt_005 as f64 / n_total,
                n_area_gt_010 as f64 / n_total,
            );
        }

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

        // Note: we do NOT cull on a minimum scale. A genuinely flat splat
        // (a thin "pancake" representing a surface) legitimately has a tiny
        // smallest axis, so there's no correct min-scale threshold — the
        // non-finite check below still removes actually-degenerate splats.
        let scale_big = scales
            .clone()
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Remove splats that are way out of bounds.
        let center = self.bounds.center;
        let bound_center =
            Tensor::<1>::from_floats([center.x, center.y, center.z], &device).reshape([1, 3]);
        let splat_dists = (splats.means() - bound_center).abs();
        let bound_mask = splat_dists
            .greater_elem(max_allowed_bounds)
            .any_dim(1)
            .squeeze_dim(1);

        // Prune parameter that's NaN.
        fn row_non_finite(t: &Tensor<2>) -> Tensor<1, Bool> {
            t.clone().is_finite().bool_not().any_dim(1).squeeze_dim(1)
        }
        let transforms_bad = row_non_finite(&splats.transforms.val());
        let sh_bad = row_non_finite(&splats.sh_coeffs.val().flatten(1, 2));
        let opac_bad = row_non_finite(&splats.raw_opacities.val().unsqueeze_dim(1));
        let non_finite_mask = transforms_bad.bool_or(sh_bad).bool_or(opac_bad);
        let num_pruned_non_finite = non_finite_mask
            .clone()
            .int()
            .sum()
            .into_scalar_async::<i32>()
            .await
            .expect("Failed to count non-finite splats") as u32;

        // "Poisoned" big-screen prune: kill splats whose max 2D screen extent
        // exceeds the threshold (every refine, including post-growth). The
        // freed budget is re-sampled below. This handles the hard outliers
        // that the smooth split + area regulariser don't reach on their own.
        let big_screen_mask = if self.config.kill_at_screen_size > 0.0 {
            Some(
                refiner
                    .max_screen_size
                    .clone()
                    .greater_elem(self.config.kill_at_screen_size),
            )
        } else {
            None
        };

        let prune_mask = alpha_mask
            .bool_or(scale_big)
            .bool_or(bound_mask)
            .bool_or(non_finite_mask);
        let prune_mask = if let Some(bsm) = big_screen_mask {
            prune_mask.bool_or(bsm)
        } else {
            prune_mask
        };

        let (mut splats, refiner, pruned_count) =
            prune_points(splats, &mut record, refiner, prune_mask).await;
        let mut split_inds = HashSet::new();

        // Always replace dead gaussians, so that the pruned budget is reused.
        if pruned_count > 0 {
            // Replacement weighting. By default opacity × visibility. With
            // `replace_by_gradient > 0`, interpolate toward the gradient-
            // weighted distribution (where error actually lives).
            let vis_f = refiner.vis_mask().float();
            let resampled_weights = splats.opacities() * vis_f.clone();
            let resampled_weights = resampled_weights
                .into_data_async()
                .await
                .expect("Failed to get weights")
                .into_vec::<f32>()
                .expect("Failed to read weights");
            let resampled_inds = multinomial_sample(&resampled_weights, pruned_count);
            split_inds.extend(resampled_inds);
        }

        // Big splats are no longer force-split in place — they're killed and
        // resampled via `kill_at_screen_size` (the prune path above), so the
        // "oversized split" count is always 0 now.
        let num_split_oversized = 0u32;

        let pre_high_grad = split_inds.len();
        if iter < self.config.growth_stop_iter {
            let above_threshold = refiner.above_threshold(self.config.growth_grad_threshold);

            let threshold_count = above_threshold
                .clone()
                .int()
                .sum()
                .into_scalar_async::<i32>()
                .await
                .expect("Failed to get threshold") as u32;

            let grow_count =
                (threshold_count as f32 * self.config.growth_select_fraction).round() as u32;

            let sample_high_grad = grow_count.saturating_sub(pruned_count);

            // Saturating — cur_splats can exceed max_splats if the scene
            // was loaded above cap, and the u32 underflow would request
            // ~4B new splats.
            let cur_splats = splats.num_splats() + split_inds.len() as u32;
            let headroom = self.config.max_splats.saturating_sub(cur_splats);
            let grow_count = sample_high_grad.min(headroom);

            // If still growing, sample from indices which are over the threshold.
            if grow_count > 0 {
                let weights = above_threshold.float() * refiner.refine_weight_norm.clone();
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

        let num_split_high_grad = (split_inds.len() - pre_high_grad) as u32;
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
                num_split_oversized,
                num_split_high_grad,
                num_pruned: pruned_count,
                num_pruned_non_finite,
                total_splats: splat_count,
            },
        )
    }

    fn refine_splats(
        &mut self,
        device: &Device,
        mut record: HashMap<ParamId, AdaptorRecord<AdamScaled>>,
        mut splats: Splats,
        split_inds: HashSet<i32>,
        iter: u32,
    ) -> Splats {
        let refine_count = split_inds.len();

        if refine_count > 0 {
            let refine_inds = Tensor::from_data(
                TensorData::new(split_inds.into_iter().collect::<Vec<_>>(), [refine_count]),
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
            let cur_sh_coeffs = splats.sh_coeffs.val().select(0, refine_inds.clone());
            let cur_raw_opac = splats.raw_opacities.val().select(0, refine_inds.clone());

            // The amount to offset the scale and opacity should maybe depend on how far away we have sampled these gaussians,
            // but a fixed amount seems to work ok. The only note is that divide by _less_ than SQRT(2) seems to exponentially
            // blow up, as more 'mass' is added each refine.
            // let scale_div = Tensor::ones_like(&cur_log_scale) * SQRT_2.ln();
            //
            let cur_scales = cur_log_scale.clone().exp();

            let cur_opac = sigmoid(cur_raw_opac.clone());
            let inv_opac: Tensor<1> = 1.0 - cur_opac.clone();
            // Post-split child opacity: power law in transmittance,
            //   a' = 1 - (1 - a)^p.
            // Alpha compositing multiplies transmittance (1-a), so p encodes
            // the effective overlap: p=1/2 reproduces the parent when the two
            // children fully overlap (classic 3DGS); p=1 is full inheritance
            // (no overlap). The covariance-aware split offsets the children
            // ~2 child-stds apart (= 2·sqrt(1-k²)/k at k=1/√2), i.e. between 1
            // and 2 effective overlapping copies — which is exactly where
            // p = 1/√2 lands. A p-sweep confirmed the PSNR optimum at ~0.7
            // with monotonically fewer low-opacity ambient splats. So p
            // reuses the split's own constant.
            let p = std::f32::consts::FRAC_1_SQRT_2;
            let new_opac: Tensor<1> = 1.0 - inv_opac.powf_scalar(p);
            let new_raw_opac = inv_sigmoid(new_opac.clamp(MIN_OPACITY, 1.0 - MIN_OPACITY));

            // Smooth covariance-aware split. Per-axis shrink factor:
            //   k_i = 1 - (1 - k) * (scale_i² / scale_max²)
            // is k on the principal axis, smoothly approaching 1 on much-
            // smaller axes. Mass-conserving on each axis independently:
            //   new_scale_i = k_i * scale_i
            //   offset_i    = sqrt(1 - k_i²) * scale_i (deterministic ±)
            // Limits: isotropic → uniform shrink, diagonal offset (≈ the old
            // iso mirror split). Needle → axial shrink + axial offset only.
            // Pancake → in-plane shrink + offset, thin axis untouched. One
            // function, no threshold.
            //   k_i = 1 - (1 - k) · (s_i² / s_max²)        ∈ [k, 1]
            //   new_scale_i = k_i · s_i
            //   offset_i    = sqrt(1 - k_i²) · s_i         (deterministic ±)
            // Mass-conserving on each axis: offset_i² + (k_i·s_i)² = s_i².
            // k is pinned near 1/√2 (the variance-preserving 2-component
            // mixture value); clamped to (0,1) for safety.
            let k: f32 = self.config.split_anisotropic_k.max(1e-3).min(0.99);
            let cur_scales_sq = cur_scales.clone().powi_scalar(2);
            let max_scale_sq = cur_scales_sq.clone().max_dim(1).clamp_min(1e-30);
            // ratio_i = s_i² / s_max²  (broadcast [N,3] / [N,1])
            let ratio = cur_scales_sq / max_scale_sq;
            // k_i = 1 - (1 - k) · ratio_i
            let one_minus_k = 1.0_f32 - k;
            let k_per_axis: Tensor<2> = -(ratio * one_minus_k) + 1.0;
            let offset_factor_per_axis = (-k_per_axis.clone().powi_scalar(2) + 1.0)
                .clamp_min(0.0)
                .sqrt();
            let offset_local = offset_factor_per_axis * cur_scales.clone();
            let samples = quaternion_vec_multiply(cur_rots.clone(), offset_local);
            // log_scale_diff_i = ln(k_i)  — applied additively to cur_log_scale below.
            let log_scale_diff: Tensor<2> = k_per_axis.log();

            let new_log_scales = cur_log_scale.clone() + log_scale_diff.clone();

            // Children inherit the parent's rotation; the split is expressed
            // entirely through the per-axis scale shrink + ±offset.
            let child_rots = cur_rots.clone();

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

            // Child sits at parent_mean + samples (parent moves to
            // parent_mean - samples) — anti-correlated, centroid-preserving.
            // Build new transforms row: means(3) + rotations(4) + log_scales(3)
            let new_transforms =
                Tensor::cat(vec![cur_means + samples, child_rots, new_log_scales], 1);
            // Optimizer state lives on the inner (non-autodiff) device.
            let opt_device = device.clone().inner();
            let refine_inds_opt = refine_inds.to_device(&opt_device);
            // Both halves of a split start with zero Adam moments; only the
            // ±sample position offset tells them apart. Burn's scatter bridge
            // only implements Add, so we add the negated parent value to zero
            // it out instead of using Assign.
            splats = map_splats_and_opt(
                splats,
                &mut record,
                |x| Tensor::cat(vec![x, new_transforms], 0),
                |x| Tensor::cat(vec![x, cur_sh_coeffs], 0),
                |x| Tensor::cat(vec![x, new_raw_opac], 0),
                |x: Tensor<2>| {
                    let d1 = x.dims()[1];
                    let neg_parent = -x.clone().select(0, refine_inds_opt.clone());
                    let inds: Tensor<2, Int> =
                        refine_inds_opt.clone().unsqueeze_dim(1).repeat_dim(1, d1);
                    let x = x.scatter(0, inds, neg_parent, IndexingUpdateOp::Add);
                    Tensor::cat(vec![x, Tensor::zeros([refine_count, d1], &opt_device)], 0)
                },
                |x: Tensor<3>| {
                    let [_, d1, d2] = x.dims();
                    let neg_parent = -x.clone().select(0, refine_inds_opt.clone());
                    let inds_2: Tensor<2, Int> =
                        refine_inds_opt.clone().unsqueeze_dim(1).repeat_dim(1, d1);
                    let inds: Tensor<3, Int> = inds_2.unsqueeze_dim(2).repeat_dim(2, d2);
                    let x = x.scatter(0, inds, neg_parent, IndexingUpdateOp::Add);
                    Tensor::cat(
                        vec![x, Tensor::zeros([refine_count, d1, d2], &opt_device)],
                        0,
                    )
                },
                |x: Tensor<1>| {
                    let neg_parent = -x.clone().select(0, refine_inds_opt.clone());
                    let x = x.scatter(
                        0,
                        refine_inds_opt.clone(),
                        neg_parent,
                        IndexingUpdateOp::Add,
                    );
                    Tensor::cat(vec![x, Tensor::zeros([refine_count], &opt_device)], 0)
                },
            );
        }

        let train_t = (iter as f32 / self.config.total_train_iters as f32).clamp(0.0, 1.0);
        let t_shrink_strength = 1.0 - train_t;
        let minus_opac = self.config.opac_decay * t_shrink_strength;

        // Lower opacity slowly over time.
        splats.raw_opacities = splats.raw_opacities.map(|f| {
            let new_opac = sigmoid(f) - minus_opac;
            inv_sigmoid(new_opac.clamp(1e-12, 1.0 - 1e-12))
        });

        self.optim = Some(create_optimizer_from_config().load_record(record));
        splats
    }
}

fn map_splats_and_opt(
    mut splats: Splats,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled>>,
    map_transforms: impl FnOnce(Tensor<2>) -> Tensor<2>,
    map_sh_coeffs: impl FnOnce(Tensor<3>) -> Tensor<3>,
    map_opac: impl FnOnce(Tensor<1>) -> Tensor<1>,

    map_opt_transforms: impl Fn(Tensor<2>) -> Tensor<2>,
    map_opt_sh_coeffs: impl Fn(Tensor<3>) -> Tensor<3>,
    map_opt_opac: impl Fn(Tensor<1>) -> Tensor<1>,
) -> Splats {
    splats.transforms = splats.transforms.map(map_transforms);
    map_opt(splats.transforms.id, record, &map_opt_transforms);
    splats.sh_coeffs = splats.sh_coeffs.map(map_sh_coeffs);
    map_opt(splats.sh_coeffs.id, record, &map_opt_sh_coeffs);
    splats.raw_opacities = splats.raw_opacities.map(map_opac);
    map_opt(splats.raw_opacities.id, record, &map_opt_opac);
    splats
}

/// Apply `map_fn` to `moment_1` and `moment_2`. `map_fn` must be shape-agnostic
/// along trailing dims since `moment_2` may have size-1 trailing dims under
/// `reduce_moment_2`.
fn map_opt<const D: usize>(
    param_id: ParamId,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled>>,
    map_fn: &impl Fn(Tensor<D>) -> Tensor<D>,
) {
    let mut state: AdamState<D> = record
        .remove(&param_id)
        .expect("failed to get optimizer record")
        .into_state();

    state.momentum = state.momentum.map(|mut moment| {
        moment.moment_1 = map_fn(moment.moment_1);
        moment.moment_2 = map_fn(moment.moment_2);
        moment
    });

    record.insert(param_id, AdaptorRecord::from_state(state));
}

// Prunes points based on the given mask.
//
// Args:
//   mask: bool[n]. If True, prune this Gaussian.
async fn prune_points(
    mut splats: Splats,
    record: &mut HashMap<ParamId, AdaptorRecord<AdamScaled>>,
    mut refiner: RefineRecord,
    prune: Tensor<1, Bool>,
) -> (Splats, RefineRecord, u32) {
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
        // Splat params + optimizer state share the autodiff device, but the
        // refiner runs on the inner device — give `keep()` an inner copy.
        use brush_render::burn_glue::detach_autodiff_int;
        let inner_valid_inds = detach_autodiff_int(valid_inds.clone().inner());
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
        refiner = refiner.keep(inner_valid_inds);
    }
    (splats, refiner, start_splats - new_points)
}

/// Sample a background color: base + uniform noise in [-strength, +strength], clamped to [0, 1].
fn sample_background_color(base: glam::Vec3, strength: f32) -> glam::Vec3 {
    if strength <= 0.0 {
        return base.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
    }
    use rand::RngExt as _;
    let mut rng = rand::rng();
    let noise = glam::Vec3::new(
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
        rng.random_range(-strength..strength),
    );
    (base + noise).clamp(glam::Vec3::ZERO, glam::Vec3::ONE)
}
