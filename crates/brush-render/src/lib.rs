#![recursion_limit = "256"]

use burn::backend::Backend;
use burn::backend::tensor::FloatTensor;
use burn_cubecl::CubeBackend;
use burn_wgpu::WgpuRuntime;
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

// `MainBackend` is burn's per-platform alias (which already picks the right
// bool element). `MainBackendBase` is the un-Fusioned CubeBackend that lives
// inside it — burn doesn't export it directly, so we mirror it here using the
// same parameters as the alias on each platform.
#[cfg(target_family = "wasm")]
pub type MainBackend = burn::backend::wgpu::WebGpu;
#[cfg(target_family = "wasm")]
pub type MainBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[cfg(target_os = "macos")]
pub type MainBackend = burn::backend::wgpu::Metal;
#[cfg(target_os = "macos")]
pub type MainBackendBase = CubeBackend<WgpuRuntime, f32, i32, u8>;

#[cfg(all(not(target_family = "wasm"), not(target_os = "macos")))]
pub type MainBackend = burn::backend::wgpu::Vulkan;
#[cfg(all(not(target_family = "wasm"), not(target_os = "macos")))]
pub type MainBackendBase = CubeBackend<WgpuRuntime, f32, i32, u8>;

/// `DispatchTensorKind` variant for the active wgpu backend. burn-dispatch
/// uses different variant names per backend (`Wgpu`, `Vulkan`, `Metal`); brush
/// only ever runs on one, so this alias hides the per-platform name from
/// match arms and constructors. Use as `wgpu_kind!(bt)` in both positions.
#[macro_export]
macro_rules! wgpu_kind {
    ($($t:tt)*) => {
        $crate::__wgpu_kind!($($t)*)
    };
}

#[cfg(target_family = "wasm")]
#[macro_export]
#[doc(hidden)]
macro_rules! __wgpu_kind {
    ($($t:tt)*) => { ::burn::backend::DispatchTensorKind::Wgpu($($t)*) };
}
#[cfg(target_os = "macos")]
#[macro_export]
#[doc(hidden)]
macro_rules! __wgpu_kind {
    ($($t:tt)*) => { ::burn::backend::DispatchTensorKind::Metal($($t)*) };
}
#[cfg(all(not(target_family = "wasm"), not(target_os = "macos")))]
#[macro_export]
#[doc(hidden)]
macro_rules! __wgpu_kind {
    ($($t:tt)*) => { ::burn::backend::DispatchTensorKind::Vulkan($($t)*) };
}

/// Trait for the gaussian splatting rendering pipeline.
///
/// A single call performs: cull → readback → rasterize.
pub trait SplatOps<B: Backend> {
    /// Render gaussian splats to an image.
    ///
    /// This is the full forward pipeline: cull, depth sort, readback, project,
    /// rasterize. When `bwd_info` is true, extra per-splat data is computed
    /// for the backward pass.
    #[allow(clippy::too_many_arguments)]
    fn render(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<B>,
        sh_coeffs: FloatTensor<B>,
        raw_opacities: FloatTensor<B>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
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
