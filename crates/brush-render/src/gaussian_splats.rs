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
#[derive(Module, Debug)]
pub struct Splats<B: Backend> {
    pub transforms: Param<Tensor<B, 2>>,
    pub sh_coeffs: Param<Tensor<B, 3>>,
    pub raw_opacities: Param<Tensor<B, 1>>,

    #[module(skip)] // TODO: Should be SplatRenderMode but not supported atm..
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
        let n_coeffs = sh_coeffs_for_degree(sh_degree) as usize;
        let [n, cur_coeffs, _] = self.sh_coeffs.dims();

        self.sh_coeffs = self.sh_coeffs.map(|coeffs| {
            let device = coeffs.device();
            let tens = if cur_coeffs < n_coeffs {
                Tensor::cat(
                    vec![
                        coeffs,
                        Tensor::zeros([n, n_coeffs - cur_coeffs, 3], &device),
                    ],
                    1,
                )
            } else {
                coeffs.slice(s![.., 0..n_coeffs])
            };
            tens.detach().require_grad()
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

        Self {
            transforms: Param::initialized(ParamId::new(), transforms.detach().require_grad()),
            sh_coeffs: Param::initialized(ParamId::new(), sh_coeffs.detach().require_grad()),
            raw_opacities: Param::initialized(ParamId::new(), raw_opacity.detach().require_grad()),
            render_mip: mode == SplatRenderMode::Mip,
        }
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
        let [_, coeffs, _] = self.sh_coeffs.dims();
        sh_degree_from_coeffs(coeffs as u32)
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
            let [n_sh, _coeffs, sh_dims] = self.sh_coeffs.dims();
            assert_eq!(sh_dims, 3, "SH coefficients must have 3 color channels");
            assert_eq!(
                n_sh, num_splats as usize,
                "Inconsistent number of splats in SH coeffs"
            );
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
        splats.sh_coeffs.into_value().into_primitive().tensor(),
        splats.raw_opacities.into_value().into_primitive().tensor(),
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
