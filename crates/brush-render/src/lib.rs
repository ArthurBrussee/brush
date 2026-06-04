#![recursion_limit = "256"]

use burn::backend::Backend;
use burn::backend::tensor::FloatTensor;
use camera::Camera;
use clap::ValueEnum;
use glam::Vec3;

use crate::gaussian_splats::SplatRenderMode;
pub use crate::gaussian_splats::{Splats, TextureMode, render_splats};
pub use crate::render_aux::{RenderAux, RenderAuxInner, RenderOutput};

pub mod burn_glue;
#[doc(hidden)]
pub mod dim_check;
#[doc(hidden)]
pub mod kernels;
pub mod render_aux;
pub mod shaders;

pub mod sh;

#[cfg(test)]
mod tests;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
#[doc(hidden)]
pub mod get_tile_offset;
pub mod render;
pub mod validation;

/// `DispatchTensorKind` variant for the active wgpu backend. burn-dispatch
/// uses different variant names per backend; brush only ever runs on the
/// `WebGpu` variant, so this macro hides the variant name from match arms.
#[macro_export]
macro_rules! wgpu_kind {
    ($($t:tt)*) => {
        $crate::__wgpu_kind!($($t)*)
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __wgpu_kind {
    ($($t:tt)*) => { ::burn::backend::DispatchTensorKind::Wgpu($($t)*) };
}

/// Trait for the gaussian splatting rendering pipeline.
///
/// A single call performs: cull → readback → rasterize.
pub trait SplatOps<B: Backend> {
    /// Render gaussian splats to an image.
    ///
    /// Full forward pipeline: cull, depth sort, readback, project, rasterize.
    /// `pass` picks forward-only vs. forward+backward-bookkeeping, and (only
    /// for tests) toggles the C^1 smoothstep around the alpha cutoff.
    #[allow(clippy::too_many_arguments)]
    fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<B>,
        sh_coeffs: FloatTensor<B>,
        raw_opacities: FloatTensor<B>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: gaussian_splats::RasterPass,
    ) -> impl Future<Output = RenderOutput<B>>;
}

#[derive(
    Default, ValueEnum, Clone, Copy, Eq, PartialEq, Debug, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum AlphaMode {
    #[default]
    Masked,
    Transparent,
}
