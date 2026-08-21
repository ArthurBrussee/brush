use burn::{
    Tensor,
    backend::Dispatch,
    module::{Module, Param, ParamId},
    tensor::{Device, Gradients, TensorData, activation::sigmoid, s},
};
use clap::ValueEnum;
use glam::Vec3;
use tracing::trace_span;

use crate::{
    RenderAux, SplatOps,
    camera::Camera,
    sh::{sh_coeffs_for_degree, sh_degree_from_coeffs},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SplatRenderMode {
    Default,
    Mip,
}

/// Forward/backward rasterizer mode. Replaces the old `bwd_info: bool` so the
/// test-only smooth-cutoff variant rides along on the same enum that already
/// switches in/out the backward bookkeeping.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum RasterPass {
    /// Forward only — inference / eval. No backward bookkeeping, hard
    /// `alpha >= 1/255` cutoff.
    #[default]
    Forward,
    /// Forward + backward bookkeeping (training). Hard cutoff.
    Backward,
    /// Backward + C^1 smoothstep around the alpha=1/255 cutoff. Test-only:
    /// makes the analytical backward agree with finite-diff at the cutoff,
    /// at the cost of a sub-1/255 forward shift on edge pixels.
    BackwardSmoothCutoff,
}

impl RasterPass {
    pub const fn bwd_info(self) -> bool {
        !matches!(self, Self::Forward)
    }
    pub const fn smooth_cutoff(self) -> bool {
        matches!(self, Self::BackwardSmoothCutoff)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TextureMode {
    Packed,
    #[default]
    Float,
}

/// Gaussian splat parameters.
///
/// `transforms` stores means(3) + rotations(4) + log scales(3) = 10 floats per splat
/// as a single contiguous [N, 10] tensor to minimize GPU shader bindings.
#[derive(Module, Debug)]
pub struct Splats {
    pub transforms: Param<Tensor<2>>,
    pub sh_coeffs: Param<Tensor<3>>,
    pub raw_opacities: Param<Tensor<1>>,
    #[module(skip)]
    pub render_mip: bool,
    /// Optional per-splat world-space scale floor (Mip-Splatting's 3D filter).
    /// Frozen, camera-derived, never optimized and never exported — a pure
    /// training-time pressure. When set, the render path inflates each splat's
    /// covariance to `sqrt(scale² + f²)` and energy-compensates opacity. `[N]`.
    #[module(skip)]
    pub min_scale: Option<Tensor<1>>,
}

pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

/// Mip-Splatting 3D smoothing filter: fold a per-splat world-space scale floor
/// `f` `[N]` into the packed `transforms` `[N,10]` and `raw_opac` `[N]`. Scales
/// become `sqrt(s² + f²)` and opacity is energy-compensated by `sqrt(det1/det2)`
/// over the three world axes. Differentiable w.r.t. the learned scale/opacity;
/// `f` is treated as a constant. This is the single source of truth for the
/// floor — used by both render paths and by [`Splats::bake_min_scale`].
pub fn fold_min_scale(
    transforms: Tensor<2>,
    raw_opac: Tensor<1>,
    f: Tensor<1>,
) -> (Tensor<2>, Tensor<1>) {
    // `f` is stored on the inner backend but the params may be lifted to
    // autodiff; align it so the elementwise mix below stays on one backend.
    let f = crate::burn_glue::match_backend(f, &transforms);
    let n = transforms.dims()[0] as i32;
    let log_scales = transforms.clone().slice(s![.., 7..10]); // [N,3]
    let s2 = log_scales.clone().mul_scalar(2.0).exp(); // s² = exp(2·log) [N,3]
    let f2 = f.clone().mul(f).reshape([n, 1]); // [N,1]
    let s2f = s2.add(f2); // s² + f² [N,3]

    let new_log = s2f.log().mul_scalar(0.5); // log(sqrt(s²+f²)) [N,3]
    let transforms = transforms.slice_assign(s![.., 7..10], new_log.clone());

    // Opacity energy compensation `sqrt(det(s²) / det(s²+f²))`, evaluated PER
    // AXIS in log space rather than as a ratio of three-axis determinants:
    //
    //     sqrt( Π_i s_i² / Π_i (s_i²+f²) ) = Π_i s_i / sqrt(s_i²+f²)
    //                                      = exp( Σ_i [ log s_i - log sqrt(s_i²+f²) ] )
    //
    // Algebraically identical; numerically bounded, which the determinant form
    // is not. Each summand is `log s_i - new_log_i <= 0`, so the sum is bounded
    // and `coef` lands in (0, 1] by construction. `log s_i` is the raw parameter
    // and `new_log_i` is already computed above, so neither logarithm is
    // evaluated: the round trip through `s2` (whose `log(exp(·))` would
    // reintroduce an underflow of its own for very small scales) is skipped.
    //
    // Why the determinant form had to go. `det(s²+f²)` CUBES a per-axis variance
    // that is legitimately tiny on a dense mesh/LiDAR seed — `s² ~ 1e-8` gives
    // `det ~ 1e-23`. The backward of `div` forms `rhs²`, which is then `~1e-46`:
    // subnormal. GPU shading languages flush subnormals to zero, so the backward
    // divides by exactly zero and every scale-lane gradient for that splat
    // becomes `±inf`, which the three-axis product rule then turns into NaN, and
    // which Adam writes straight into the parameters on the first step. Measured
    // on an ARKitScenes mesh seed: 11,525 of 1,129,403 splats poisoned at
    // iteration 1, the affected set matching `det(s²+f²) < sqrt(f32::MIN_POSITIVE)`
    // exactly, 11,525 for 11,525.
    //
    // The cliff is closer than it looks even on healthy scenes, because
    // `det² ∝ s¹²` when `s >> f`: a seed sitting a comfortable-sounding 85x above
    // the boundary is only `85^(1/12) = 1.44x` in scale away from it, i.e. 0.36
    // in log-scale. Any recipe that deliberately thins splats (a flattening
    // regularizer, say) walks the population over the edge during training. This
    // is why clamping the denominator would be the wrong fix: it would keep
    // producing distorted gradients for a population that is actively migrating
    // into the band.
    //
    // Gradient, for the record: `d coef / d log s_i = coef · f²/(s_i²+f²)`, which
    // is bounded by `coef <= 1` everywhere. The determinant form's gradient had
    // no such bound.
    let coef = log_scales.sub(new_log).sum_dim(1).exp().reshape([n]); // [N]
    let opac = sigmoid(raw_opac).mul(coef).clamp(1e-6, 1.0 - 1e-6);
    let raw_opac = opac.clone().div(opac.neg().add_scalar(1.0)).log(); // logit

    (transforms, raw_opac)
}

impl Splats {
    pub fn from_raw(
        pos_data: Vec<f32>,
        rot_data: Vec<f32>,
        scale_data: Vec<f32>,
        coeffs_data: Vec<f32>,
        opac_data: Vec<f32>,
        mode: SplatRenderMode,
        device: &Device,
    ) -> Self {
        let _ = trace_span!("Splats::from_raw").entered();
        let n_splats = pos_data.len() / 3;
        let log_scales = Tensor::from_data(TensorData::new(scale_data, [n_splats, 3]), device);
        let means_tensor = Tensor::from_data(TensorData::new(pos_data, [n_splats, 3]), device);
        let rotations = Tensor::from_data(TensorData::new(rot_data, [n_splats, 4]), device);
        let n_coeffs = coeffs_data.len() / n_splats;
        let sh_coeffs = Tensor::from_data(
            TensorData::new(coeffs_data, [n_splats, n_coeffs / 3, 3]),
            device,
        );
        let raw_opacities =
            Tensor::from_data(TensorData::new(opac_data, [n_splats]), device).require_grad();
        Self::from_tensor_data(
            means_tensor,
            rotations,
            log_scales,
            sh_coeffs,
            raw_opacities,
            mode,
        )
    }

    /// Set the SH degree of this splat to be equal to `sh_degree`
    pub fn with_sh_degree(mut self, sh_degree: u32) -> Self {
        let n_coeffs = sh_coeffs_for_degree(sh_degree) as usize;
        let n = self.num_splats() as usize;

        self.sh_coeffs = self.sh_coeffs.map(|coeffs| {
            let device = coeffs.device();
            let cur = coeffs.dims()[1];
            if cur < n_coeffs {
                let zeros = Tensor::<3>::zeros([n, n_coeffs - cur, 3], &device);
                Tensor::cat(vec![coeffs, zeros], 1)
            } else {
                coeffs.slice(s![.., 0..n_coeffs])
            }
            .detach()
            .require_grad()
        });
        self
    }

    pub fn from_tensor_data(
        means: Tensor<2>,
        rotation: Tensor<2>,
        log_scales: Tensor<2>,
        sh_coeffs: Tensor<3>,
        raw_opacity: Tensor<1>,
        mode: SplatRenderMode,
    ) -> Self {
        assert_eq!(means.dims()[1], 3, "Means must be 3D");
        assert_eq!(rotation.dims()[1], 4, "Rotation must be 4D");
        assert_eq!(log_scales.dims()[1], 3, "Scales must be 3D");

        let transforms = Tensor::cat(vec![means, rotation, log_scales], 1);

        Self {
            transforms: Param::initialized(ParamId::new(), transforms.detach().require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.detach().require_grad()),
            raw_opacities: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            render_mip: mode == SplatRenderMode::Mip,
            min_scale: None,
        }
    }

    /// Attach a per-splat world-space scale floor (see [`Splats::min_scale`]).
    /// `f` must be `[num_splats]`. Training-only; cleared by refine and never
    /// serialized.
    pub fn with_min_scale(mut self, f: Tensor<1>) -> Self {
        self.min_scale = Some(f);
        self
    }

    /// Get means (positions) — slice of transforms columns 0..3.
    pub fn means(&self) -> Tensor<2> {
        self.transforms.val().slice(s![.., 0..3])
    }

    /// Get rotation quaternions — slice of transforms columns 3..7.
    pub fn rotations(&self) -> Tensor<2> {
        self.transforms.val().slice(s![.., 3..7])
    }

    /// Get log-space scales — slice of transforms columns 7..10.
    pub fn log_scales(&self) -> Tensor<2> {
        self.transforms.val().slice(s![.., 7..10])
    }

    /// Post-activation opacity, with the 3D-filter energy compensation folded
    /// in when a `min_scale` floor is set (see [`fold_min_scale`]). This is the
    /// splat's *real* opacity — callers (export, refine decisions, viewer)
    /// should use it rather than reaching for `raw_opacities`.
    pub fn opacities(&self) -> Tensor<1> {
        match &self.min_scale {
            Some(f) => {
                let (_, raw_opac) =
                    fold_min_scale(self.transforms.val(), self.raw_opacities.val(), f.clone());
                sigmoid(raw_opac)
            }
            None => sigmoid(self.raw_opacities.val()),
        }
    }

    /// World-space scales, with the 3D-filter floor folded in when `min_scale`
    /// is set: `sqrt(scale² + f²)`. This is the splat's *real* size — the floor
    /// is part of the splat's definition, so renders/exports use this, not the
    /// raw `log_scales`.
    pub fn scales(&self) -> Tensor<2> {
        match &self.min_scale {
            Some(f) => {
                let (transforms, _) =
                    fold_min_scale(self.transforms.val(), self.raw_opacities.val(), f.clone());
                transforms.slice(s![.., 7..10]).exp()
            }
            None => self.log_scales().exp(),
        }
    }

    /// Permanently fold the `min_scale` floor into the raw scale/opacity params
    /// and clear it, yielding a plain canonical splat that renders identically.
    /// Used at ply export so the floor is written as ordinary derived scales —
    /// never as a separate field.
    pub fn bake_min_scale(mut self) -> Self {
        if let Some(f) = self.min_scale.take() {
            let (transforms, raw_opac) =
                fold_min_scale(self.transforms.val(), self.raw_opacities.val(), f);
            self.transforms =
                Param::initialized(self.transforms.id, transforms.detach().require_grad());
            self.raw_opacities =
                Param::initialized(self.raw_opacities.id, raw_opac.detach().require_grad());
        }
        self
    }

    pub fn num_splats(&self) -> u32 {
        self.transforms.dims()[0] as u32
    }

    pub fn sh_degree(&self) -> u32 {
        let [_, n_coeffs, _] = self.sh_coeffs.dims();
        sh_degree_from_coeffs(n_coeffs as u32)
    }

    pub fn device(&self) -> Device {
        self.transforms.device()
    }

    pub async fn validate_values(self) {
        #[cfg(any(test, feature = "debug-validation"))]
        {
            #[cfg(not(target_family = "wasm"))]
            if std::env::args().any(|a| a == "--bench") {
                return;
            }

            use crate::validation::validate_tensor_val;

            let num_splats = self.num_splats();

            // Validate means (positions)
            validate_tensor_val(self.means(), "means", None, None).await;
            // Validate rotations
            validate_tensor_val(self.rotations(), "rotations", None, None).await;
            // Validate pre-activation scales (log_scales) and post-activation scales
            validate_tensor_val(self.log_scales(), "log_scales", Some(-10.0), Some(10.0)).await;
            let scales = self.scales();
            validate_tensor_val(scales.clone(), "scales", Some(1e-20), Some(10000.0)).await;
            // Validate SH coefficients
            validate_tensor_val(self.sh_coeffs.val(), "sh_coeffs", Some(-5.0), Some(5.0)).await;
            // Validate pre-activation opacity (raw_opacity) and post-activation opacity
            validate_tensor_val(
                self.raw_opacities.val(),
                "raw_opacity",
                Some(-20.0),
                Some(20.0),
            )
            .await;
            let opacities = self.opacities();
            validate_tensor_val(opacities, "opacities", Some(0.0), Some(1.0)).await;
            // Range validation if requested
            // Scales should be positive and reasonable
            validate_tensor_val(scales, "scales", Some(1e-6), Some(100.0)).await;

            let [n_transforms, t_dims] = self.transforms.dims();
            assert_eq!(
                t_dims, 10,
                "Transforms must be 10D (means(3) + quats(4) + log_scales(3))"
            );
            assert_eq!(
                n_transforms, num_splats as usize,
                "Inconsistent number of splats in transforms"
            );
            let [n_opacity] = self.raw_opacities.dims();
            assert_eq!(
                n_opacity, num_splats as usize,
                "Inconsistent number of splats in opacity"
            );
            let [n_sh, _, sh_dims] = self.sh_coeffs.dims();
            assert_eq!(sh_dims, 3, "SH coeffs must have 3 color channels");
            assert_eq!(
                n_sh, num_splats as usize,
                "Inconsistent number of splats in SH coeffs"
            );
        }
    }

    /// Post-backward variant of `validate_values`, checks that no splat
    /// parameter gradient has a NaN or Inf. Debug-only.
    #[allow(unused_variables)]
    pub async fn bwd_validate(&self, loss: Tensor<1>) -> Gradients {
        let grads = loss.backward();
        #[cfg(any(test, feature = "debug-validation"))]
        let (t, sh, opac) = (
            self.transforms.grad(&grads),
            self.sh_coeffs.grad(&grads),
            self.raw_opacities.grad(&grads),
        );

        #[cfg(any(test, feature = "debug-validation"))]
        {
            use crate::validation::validate_gradient;

            #[cfg(not(target_family = "wasm"))]
            if std::env::args().any(|a| a == "--bench") {
                return grads;
            }
            if let Some(g) = t {
                validate_gradient(g, "transforms").await;
            }
            if let Some(g) = sh {
                validate_gradient(g, "sh_coeffs").await;
            }
            if let Some(g) = opac {
                validate_gradient(g, "raw_opacities").await;
            }
        }

        grads
    }
}

/// Render splats on a non-differentiable device.
pub async fn render_splats(
    splats: Splats,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    splat_scale: Option<f32>,
    texture_mode: TextureMode,
) -> (Tensor<3>, RenderAux) {
    splats.clone().validate_values().await;

    let sh_coeffs = splats.sh_coeffs.into_value();

    // Fold the 3D-filter floor into scales/opacity first (the floor is part of
    // the splat's definition, so eval/viewer render with it just like training).
    let (transforms, raw_opacities) = match &splats.min_scale {
        Some(f) => fold_min_scale(
            splats.transforms.val(),
            splats.raw_opacities.val(),
            f.clone(),
        ),
        None => (splats.transforms.val(), splats.raw_opacities.val()),
    };

    let transforms = if let Some(scale) = splat_scale {
        let adjusted = transforms.clone().slice(s![.., 7..10]) + scale.ln();
        transforms.slice_assign(s![.., 7..10], adjusted)
    } else {
        transforms
    };

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let use_float = matches!(texture_mode, TextureMode::Float);
    let render_device = transforms.device();

    // Float mode needs `Backward` (f32 image + per-splat bookkeeping); Packed
    // mode goes through the packed u8 path. Neither inference path uses the
    // smooth cutoff — that's reserved for the gradient-check tests.
    let pass = if use_float {
        RasterPass::Backward
    } else {
        RasterPass::Forward
    };
    // Route through the `#[backend_extension]`-generated `Dispatch` impl: it
    // unwraps these dispatch primitives to the Wgpu backend, runs the render,
    // and re-wraps the `RenderOutput` via its `ExtensionType` derive.
    let output = <Dispatch as SplatOps>::render(
        camera,
        img_size,
        transforms.into_dispatch(),
        sh_coeffs.into_dispatch(),
        raw_opacities.into_dispatch(),
        // Inference path: no gradients, so the refine-weight accumulator is a
        // throwaway scalar the concrete backends ignore.
        Tensor::<1>::zeros([1], &render_device).into_dispatch(),
        render_mode,
        background,
        pass,
    )
    .await;

    output.clone().validate().await;

    let img_size = output.aux.img_size;
    let num_visible = output.aux.num_visible;
    let num_intersections = output.aux.num_intersections;

    let aux = RenderAux {
        num_visible,
        num_intersections,
        visible: Tensor::from_dispatch(output.aux.visible),
        max_radius: Tensor::from_dispatch(output.aux.max_radius),
        tile_offsets: Tensor::from_dispatch(output.aux.tile_offsets),
        img_size,
    };

    (Tensor::from_dispatch(output.out_img), aux)
}

#[cfg(test)]
mod min_scale_fold_tests {
    use super::*;
    use burn::tensor::TensorData;

    /// Smallest positive NORMAL f32. Below this, f32 values are subnormal, and
    /// GPU shading languages are permitted to flush them to zero.
    const F32_MIN_POSITIVE: f32 = f32::MIN_POSITIVE;

    /// Build one splat per supplied isotropic log-scale, all at the origin with
    /// identity rotation and a fixed raw opacity.
    fn fold_inputs(
        log_scales: &[f32],
        floor: f32,
        device: &Device,
    ) -> (Tensor<2>, Tensor<1>, Tensor<1>) {
        let n = log_scales.len();
        let mut rows = Vec::with_capacity(n * 10);
        for &l in log_scales {
            rows.extend_from_slice(&[
                0.0, 0.0, 0.0, // mean
                1.0, 0.0, 0.0, 0.0, // quat (w, x, y, z)
                l, l, l, // isotropic log-scales
            ]);
        }
        let transforms =
            Tensor::<2>::from_data(TensorData::new(rows, [n, 10]), device).require_grad();
        let raw_opac =
            Tensor::<1>::from_data(TensorData::new(vec![-2.2f32; n], [n]), device).require_grad();
        let f = Tensor::<1>::from_data(TensorData::new(vec![floor; n], [n]), device);
        (transforms, raw_opac, f)
    }

    /// `det(s² + f²)` for an isotropic splat, evaluated the way the fold does.
    fn det_s2f(log_scale: f32, floor: f32) -> f32 {
        let s2f = (2.0 * log_scale).exp() + floor * floor;
        s2f * s2f * s2f
    }

    /// The opacity-compensation coefficient `sqrt(det(s²)/det(s²+f²))`,
    /// evaluated in f64 so it can serve as a reference for the f32 graph.
    fn reference_coef(log_scale: f32, floor: f32) -> f64 {
        let s2 = (2.0 * f64::from(log_scale)).exp();
        let f2 = f64::from(floor) * f64::from(floor);
        (s2 / (s2 + f2)).powf(1.5)
    }

    /// Host-f64 reference for `d/d(log_scale_i)` of
    /// `sum(new_log) + sum(fold_opacity)` on an isotropic splat.
    ///
    /// `new_log_i = ½·ln(s_i²+f²)`                       -> `s_i²/(s_i²+f²)`
    /// `coef      = exp(Σ_j [L_j - new_log_j])`          -> `coef·f²/(s_i²+f²)`
    /// `y         = logit(σ(raw)·coef)`, and `σ(raw)·coef = o`, so the second
    /// term collapses to `(f²/(s_i²+f²)) / (1-o)`.
    ///
    /// Valid only while the opacity clamp is inactive; the caller asserts that.
    fn reference_scale_grad(log_scale: f32, floor: f32, raw: f32) -> (f64, f64) {
        let s2 = (2.0 * f64::from(log_scale)).exp();
        let f2 = f64::from(floor) * f64::from(floor);
        let coef = reference_coef(log_scale, floor);
        let opac = 1.0 / (1.0 + f64::from(-raw).exp()) * coef;
        let grad = s2 / (s2 + f2) + (f2 / (s2 + f2)) / (1.0 - opac);
        (grad, opac)
    }

    /// The 3D-filter fold must produce finite scale-lane gradients for splats
    /// whose `det(s² + f²)` squares into the f32 subnormal range.
    ///
    /// `fold_min_scale` used to compute its opacity compensation as
    /// `sqrt(det(s²) / det(s²+f²))`, forming the three-axis product explicitly.
    /// On a dense mesh/LiDAR seed the per-axis variances are ~1e-8, so the
    /// product cubes to ~1e-23; `div`'s backward then squares that denominator,
    /// landing at ~1e-46. GPU backends flush subnormals to zero, so the backward
    /// divides by exactly zero and every scale-lane gradient for that splat goes
    /// non-finite. Adam turns the resulting inf/NaN into NaN parameters on the
    /// very first step.
    ///
    /// The scales below straddle `sqrt(f32::MIN_POSITIVE)` in `det(s²+f²)`, so
    /// the assertion is a real boundary test: rows above must have been finite
    /// before this fix too, and rows below must be finite after it.
    #[tokio::test]
    async fn fold_min_scale_gradient_is_finite_below_the_subnormal_cliff() {
        let device = Device::from(brush_cube::test_helpers::test_device().await).autodiff();

        // A floor in the range the Mip-Splatting filter actually produces on an
        // indoor capture (0.3162 * distance / focal).
        let floor = 1.35e-4f32;
        // Log-scales spanning the cliff. -9.21 is the smallest scale in the
        // ARKitScenes mesh seed that first exposed this.
        let log_scales = [-9.21f32, -8.5, -8.0, -7.5, -7.0, -6.0, -5.0, -4.0, -2.88];

        // Guard the fixture itself: the test is meaningless unless the sample
        // genuinely straddles the boundary.
        let sq = |l: f32| {
            let d = det_s2f(l, floor);
            d * d
        };
        assert!(
            sq(log_scales[0]) < F32_MIN_POSITIVE,
            "fixture must include a row whose det² is subnormal, got {}",
            sq(log_scales[0])
        );
        assert!(
            sq(log_scales[log_scales.len() - 1]) > F32_MIN_POSITIVE,
            "fixture must include a row whose det² is a normal f32, got {}",
            sq(log_scales[log_scales.len() - 1])
        );

        let (transforms, raw_opac, f) = fold_inputs(&log_scales, floor, &device);
        let (folded_transforms, folded_opac) =
            fold_min_scale(transforms.clone(), raw_opac.clone(), f);

        // A scalar loss that reaches BOTH fold outputs, so the scale lanes carry
        // the compensation branch's gradient as well as the scale-replacement
        // branch's.
        let loss = folded_transforms.clone().slice(s![.., 7..10]).sum() + folded_opac.sum();
        let grads = loss.backward();

        let grad = transforms
            .grad(&grads)
            .expect("the fold must reach the transforms");
        let scale_grad: Vec<f32> = grad
            .slice(s![.., 7..10])
            .into_data_async()
            .await
            .expect("scale-gradient readback")
            .to_vec()
            .expect("f32 scale gradients");

        let bad: Vec<(usize, f32, f32)> = scale_grad
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_finite())
            .map(|(i, g)| (i / 3, log_scales[i / 3], *g))
            .collect();
        assert!(
            bad.is_empty(),
            "non-finite scale gradients from the 3D-filter fold at \
             (row, log_scale, grad): {bad:?}"
        );

        // Finite is necessary but not sufficient: a fix that merely clamped the
        // denominator would also be finite, and WRONG. Pin the VALUE against a
        // host-f64 evaluation of the same expression — including, and especially,
        // on the rows that sat below the old cliff.
        for (row, &l) in log_scales.iter().enumerate() {
            let (expect, opac) = reference_scale_grad(l, floor, -2.2);
            assert!(
                (1e-6..=1.0 - 1e-6).contains(&opac),
                "row {row}: fixture must not engage the opacity clamp, opac = {opac}"
            );
            for axis in 0..3 {
                let got = f64::from(scale_grad[row * 3 + axis]);
                let rel = (got - expect).abs() / expect.abs();
                assert!(
                    rel < 1e-4,
                    "row {row} axis {axis} (log_scale {l}, det(s²+f²)² = {:e}): \
                     scale gradient {got} != host-f64 reference {expect} (rel {rel})",
                    {
                        let d = det_s2f(l, floor);
                        d * d
                    }
                );
            }
        }
    }

    /// The reformulated coefficient must agree with the original closed form
    /// wherever the original was numerically valid.
    #[tokio::test]
    async fn fold_min_scale_opacity_matches_the_closed_form() {
        let device = Device::from(brush_cube::test_helpers::test_device().await);

        let floor = 1.35e-4f32;
        // Normal-range rows only: this pins the VALUE, and the old expression
        // is only a valid reference where it did not underflow.
        let log_scales = [-6.0f32, -5.0, -4.0, -3.0, -2.0];
        let raw = -2.2f32;

        let (transforms, raw_opac, f) = fold_inputs(&log_scales, floor, &device);
        let (_, folded_opac) = fold_min_scale(transforms, raw_opac, f);
        let got: Vec<f32> = folded_opac
            .into_data_async()
            .await
            .expect("opacity readback")
            .to_vec()
            .expect("f32 opacities");

        for (i, &l) in log_scales.iter().enumerate() {
            let coef = reference_coef(l, floor);
            let opac = (1.0 / (1.0 + f64::from(-raw).exp())) * coef;
            let expect = (opac / (1.0 - opac)).ln() as f32;
            let rel = (got[i] - expect).abs() / expect.abs().max(1e-6);
            assert!(
                rel < 1e-4,
                "row {i} (log_scale {l}): folded logit {} != closed form {expect} (rel {rel})",
                got[i]
            );
        }
    }
}
