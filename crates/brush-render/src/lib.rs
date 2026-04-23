#![recursion_limit = "256"]

use burn::prelude::Backend;
use burn::tensor::ops::FloatTensor;
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::WgpuRuntime;
use camera::Camera;
use clap::ValueEnum;
use glam::Vec3;

use crate::gaussian_splats::SplatRenderMode;
pub use crate::gaussian_splats::{TextureMode, render_splats};
pub use crate::render_aux::{RenderAux, RenderOutput};

mod burn_glue;
mod dim_check;
pub mod render_aux;
pub mod shaders;

pub mod sh;

#[cfg(test)]
mod tests;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
mod get_tile_offset;
pub mod render;
pub mod validation;

pub type MainBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type MainBackend = Fusion<MainBackendBase>;

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
