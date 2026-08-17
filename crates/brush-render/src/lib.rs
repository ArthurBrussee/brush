#![recursion_limit = "256"]

use brush_cube::MainBackend as Wgpu;
use burn::backend::tensor::FloatTensor;
use burn::backend::{Autodiff, Backend};
use camera::Camera;
use clap::ValueEnum;
use glam::Vec3;

use crate::gaussian_splats::SplatRenderMode;
pub use crate::gaussian_splats::{Splats, TextureMode, render_splats};
pub use crate::render_aux::{RenderAux, RenderAuxInner, RenderOutput};

pub mod burn_glue;
pub mod bwd;
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
macro_rules! wgpu_kind {
    ($($t:tt)*) => { ::burn::backend::DispatchTensorKind::Wgpu($($t)*) };
}
pub(crate) use wgpu_kind;

/// Trait for the gaussian splatting rendering pipeline.
///
/// A single call performs: cull → readback → rasterize.
///
/// `#[backend_extension(Wgpu)]` generates `impl SplatOps for Dispatch`, which
/// unwraps the type-erased `Tensor<D>` dispatch primitives to the concrete
/// Wgpu backend, calls the hand-written `impl SplatOps for Wgpu`, and re-wraps
/// the `RenderOutput` via its `ExtensionType` derive. Only the non-autodiff
/// arm is generated: the differentiable path is a hand-rolled `Backward` in
/// `brush-render-bwd` and never dispatches `render` through `Autodiff`.
#[burn::backend::backend_extension(Wgpu, Autodiff)]
pub trait SplatOps: Backend {
    /// Render gaussian splats to an image.
    ///
    /// Full forward pipeline: cull, depth sort, readback, project, rasterize.
    ///
    /// `refine_weight` is a zero-filled accumulator that catches the per-splat
    /// refinement weight gradient. Only the `Autodiff` impl reads it; the
    /// concrete backends ignore it.
    /// `features` is an optional `[N, 3]` per-splat feature (e.g. a normal)
    /// alpha-composited into 3 extra output channels alongside RGB. The
    /// dummy-tensor pattern gates it: a rank-1 `[1]` tensor disables the
    /// feature path entirely (the kernels compile to the feature-free
    /// variant); a rank-2 `[N, 3]` tensor enables it, which requires a
    /// backward-bookkeeping `pass` and widens `out_img` to `[H, W, 7]`.
    /// `pass` picks forward-only vs. forward+backward-bookkeeping, and (only
    /// for tests) toggles the C^1 smoothstep around the alpha cutoff.
    #[allow(clippy::too_many_arguments)]
    fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<Self>,
        sh_coeffs: FloatTensor<Self>,
        raw_opacities: FloatTensor<Self>,
        refine_weight: FloatTensor<Self>,
        features: FloatTensor<Self>,
        render_mode: SplatRenderMode,
        background: Vec3,
        pass: gaussian_splats::RasterPass,
    ) -> impl Future<Output = RenderOutput<Self>>;
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
