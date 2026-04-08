use burn::{
    Tensor,
    module::{Module, Param, ParamId},
    prelude::Backend,
    tensor::{TensorData, TensorPrimitive, activation::sigmoid, s},
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
///
/// SH coefficients are split into DC (band 0) and rest (bands 1+) for separate
/// optimizer treatment (adam-mini on rest).
#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub transforms: Param<Tensor<B, 2>>,
    /// DC (band 0) SH coefficients, shape [N, 1, 3].
    pub sh_coeffs_dc: Param<Tensor<B, 3>>,
    /// Higher-order SH coefficients (bands 1+), shape [N, C-1, 3].
    pub sh_coeffs_rest: Param<Tensor<B, 3>>,
    pub raw_opacities: Param<Tensor<B, 1>>,

    #[module(skip)]
    pub render_mip: bool,
}

pub fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

impl<B: Backend> Splats<B> {
    pub fn from_raw(
        pos_data: Vec<f32>,
        rot_data: Vec<f32>,
        scale_data: Vec<f32>,
        coeffs_data: Vec<f32>,
        opac_data: Vec<f32>,
        mode: SplatRenderMode,
        device: &B::Device,
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
        let n_rest = sh_coeffs_for_degree(sh_degree) as usize - 1;
        let n = self.num_splats() as usize;
        let cur_degree = self.sh_degree();

        self.sh_coeffs_rest = self.sh_coeffs_rest.map(|rest| {
            let device = rest.device();
            if n_rest == 0 {
                Tensor::<B, 3>::zeros([n, 0, 3], &device)
            } else if cur_degree == 0 {
                Tensor::<B, 3>::zeros([n, n_rest, 3], &device)
            } else {
                let cur_rest = rest.dims()[1];
                if cur_rest < n_rest {
                    let zeros = Tensor::<B, 3>::zeros([n, n_rest - cur_rest, 3], &device);
                    Tensor::cat(vec![rest, zeros], 1)
                } else {
                    rest.slice(s![.., 0..n_rest])
                }
            }
            .detach()
            .require_grad()
        });
        self
    }

    pub fn from_tensor_data(
        means: Tensor<B, 2>,
        rotation: Tensor<B, 2>,
        log_scales: Tensor<B, 2>,
        sh_coeffs: Tensor<B, 3>,
        raw_opacity: Tensor<B, 1>,
        mode: SplatRenderMode,
    ) -> Self {
        assert_eq!(means.dims()[1], 3, "Means must be 3D");
        assert_eq!(rotation.dims()[1], 4, "Rotation must be 4D");
        assert_eq!(log_scales.dims()[1], 3, "Scales must be 3D");

        let transforms = Tensor::cat(vec![means, rotation, log_scales], 1);

        // Split SH coefficients: DC (band 0) separate from rest (bands 1+).
        let sh_coeffs_dc = sh_coeffs.clone().slice(s![.., 0..1]);
        let [n, n_coeffs, _] = sh_coeffs.dims();
        let sh_coeffs_rest = if n_coeffs > 1 {
            sh_coeffs.slice(s![.., 1..n_coeffs])
        } else {
            Tensor::<B, 3>::zeros([n, 0, 3], &sh_coeffs.device())
        };

        Self {
            transforms: Param::initialized(ParamId::new(), transforms.detach().require_grad()),
            sh_coeffs_dc: Param::initialized(ParamId::new(), sh_coeffs_dc.detach().require_grad()),
            sh_coeffs_rest: Param::initialized(
                ParamId::new(),
                sh_coeffs_rest.detach().require_grad(),
            ),
            raw_opacities: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            render_mip: mode == SplatRenderMode::Mip,
        }
    }

    /// Get combined SH coefficients as a single f32 tensor (for serialization/export).
    pub fn sh_coeffs_combined(&self) -> Tensor<B, 3> {
        let dc = self.sh_coeffs_dc.val();
        if self.sh_degree() == 0 {
            return dc;
        }
        let rest = self.sh_coeffs_rest.val();
        Tensor::cat(vec![dc, rest], 1)
    }

    /// Get means (positions) — slice of transforms columns 0..3.
    pub fn means(&self) -> Tensor<B, 2> {
        self.transforms.val().slice(s![.., 0..3])
    }

    /// Get rotation quaternions — slice of transforms columns 3..7.
    pub fn rotations(&self) -> Tensor<B, 2> {
        self.transforms.val().slice(s![.., 3..7])
    }

    /// Get log-space scales — slice of transforms columns 7..10.
    pub fn log_scales(&self) -> Tensor<B, 2> {
        self.transforms.val().slice(s![.., 7..10])
    }

    pub fn opacities(&self) -> Tensor<B, 1> {
        sigmoid(self.raw_opacities.val())
    }

    pub fn scales(&self) -> Tensor<B, 2> {
        self.log_scales().exp()
    }

    pub fn num_splats(&self) -> u32 {
        self.transforms.dims()[0] as u32
    }

    pub fn sh_degree(&self) -> u32 {
        let [_, rest_coeffs, _] = self.sh_coeffs_rest.dims();
        sh_degree_from_coeffs((1 + rest_coeffs) as u32)
    }

    pub fn device(&self) -> B::Device {
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
            validate_tensor_val(
                self.sh_coeffs_dc.val(),
                "sh_coeffs_dc",
                Some(-5.0),
                Some(5.0),
            )
            .await;
            validate_tensor_val(
                self.sh_coeffs_rest.val(),
                "sh_coeffs_rest",
                Some(-5.0),
                Some(5.0),
            )
            .await;
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

            assert!(num_splats > 0, "Splats must contain at least one splat");

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
            let [n_dc, dc_coeffs, dc_dims] = self.sh_coeffs_dc.dims();
            assert_eq!(dc_dims, 3, "SH DC must have 3 color channels");
            assert_eq!(dc_coeffs, 1, "SH DC must have exactly 1 coefficient");
            assert_eq!(
                n_dc, num_splats as usize,
                "Inconsistent number of splats in SH DC"
            );
            if self.sh_degree() > 0 {
                let [n_rest, _rest_coeffs, rest_dims] = self.sh_coeffs_rest.dims();
                assert_eq!(rest_dims, 3, "SH rest must have 3 color channels");
                assert_eq!(
                    n_rest, num_splats as usize,
                    "Inconsistent number of splats in SH rest"
                );
            }
        }
    }
}

/// Render splats on a non-differentiable backend.
///
/// NB: This doesn't work on a differentiable backend. Use
/// [`brush_render_bwd::render_splats`] for that.
///
/// Takes ownership of the splats. Clone before calling if you need to reuse them.
pub async fn render_splats<B: Backend + SplatOps<B>>(
    splats: Splats<B>,
    camera: &Camera,
    img_size: glam::UVec2,
    background: Vec3,
    splat_scale: Option<f32>,
    texture_mode: TextureMode,
) -> (Tensor<B, 3>, RenderAux<B>) {
    splats.clone().validate_values().await;

    let sh_degree = splats.sh_degree();
    let sh_coeffs_dc = splats.sh_coeffs_dc.into_value();
    let sh_coeffs_rest = splats.sh_coeffs_rest.into_value();
    let raw_opacities = splats.raw_opacities.into_value();
    let transforms = if let Some(scale) = splat_scale {
        let t = splats.transforms.into_value();
        let adjusted = t.clone().slice(s![.., 7..10]) + scale.ln();
        t.slice_assign(s![.., 7..10], adjusted)
    } else {
        splats.transforms.into_value()
    };

    let render_mode = if splats.render_mip {
        SplatRenderMode::Mip
    } else {
        SplatRenderMode::Default
    };

    let use_float = matches!(texture_mode, TextureMode::Float);

    let output = B::render(
        camera,
        img_size,
        transforms.into_primitive().tensor(),
        sh_coeffs_dc.into_primitive().tensor(),
        sh_coeffs_rest.into_primitive().tensor(),
        raw_opacities.into_primitive().tensor(),
        sh_degree,
        render_mode,
        background,
        use_float,
    )
    .await;

    output.clone().validate().await;

    (
        Tensor::from_primitive(TensorPrimitive::Float(output.out_img)),
        output.aux,
    )
}
