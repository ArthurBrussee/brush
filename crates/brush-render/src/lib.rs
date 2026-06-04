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

/// Global config for the in-kernel screen-area regulariser. Set once by the
/// training driver before the train loop starts; read inside the rasterizer
/// host code when constructing `ProjectUniforms`. The values flow into the
/// backward kernel which uses them to add an area-loss gradient contribution
/// to v_cov2d. Zero penalty = disabled (the kernel branch short-circuits).
use std::sync::atomic::{AtomicU32, Ordering};
static SCREEN_AREA_PENALTY_BITS: AtomicU32 = AtomicU32::new(0);
static SCREEN_AREA_THRESHOLD_BITS: AtomicU32 = AtomicU32::new(0);

/// Set the screen-area penalty weight + threshold. Call before training.
pub fn set_screen_area_penalty(weight: f32, threshold: f32) {
    SCREEN_AREA_PENALTY_BITS.store(weight.to_bits(), Ordering::Relaxed);
    SCREEN_AREA_THRESHOLD_BITS.store(threshold.to_bits(), Ordering::Relaxed);
}

/// Read the current screen-area penalty config. Used by the rasterizer host
/// code to populate `ProjectUniforms`.
pub fn screen_area_penalty_config() -> (f32, f32) {
    (
        f32::from_bits(SCREEN_AREA_PENALTY_BITS.load(Ordering::Relaxed)),
        f32::from_bits(SCREEN_AREA_THRESHOLD_BITS.load(Ordering::Relaxed)),
    )
}

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
