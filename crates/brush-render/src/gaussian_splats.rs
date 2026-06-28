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
    let s2 = log_scales.mul_scalar(2.0).exp(); // s² = exp(2·log) [N,3]
    let f2 = f.clone().mul(f).reshape([n, 1]); // [N,1]
    let s2f = s2.clone().add(f2); // s² + f² [N,3]

    let new_log = s2f.clone().log().mul_scalar(0.5); // log(sqrt(s²+f²)) [N,3]
    let transforms = transforms.slice_assign(s![.., 7..10], new_log);

    let det = |t: Tensor<2>| {
        t.clone().slice(s![.., 0..1]) * t.clone().slice(s![.., 1..2]) * t.slice(s![.., 2..3])
    };
    let coef = (det(s2).div(det(s2f))).sqrt().reshape([n]); // sqrt(det1/det2) [N]
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
