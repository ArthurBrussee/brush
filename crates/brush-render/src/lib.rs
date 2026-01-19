#![recursion_limit = "256"]

use burn::prelude::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::WgpuRuntime;
use camera::Camera;
use clap::ValueEnum;
use glam::Vec3;
use render_aux::ProjectOutput;

use crate::gaussian_splats::SplatRenderMode;
pub use crate::gaussian_splats::{TextureMode, render_splats};
pub use crate::render_aux::RenderAux;

mod burn_glue;
mod dim_check;
pub mod render_aux;
pub mod shaders;

pub mod sh;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
mod get_tile_offset;
pub mod render;
pub mod validation;

pub type MainBackendBase = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type MainBackend = Fusion<MainBackendBase>;

/// Trait for the the gaussian splatting rendering pipeline.
///
/// This trait provides two passes:
/// 1. `project`: Culling, depth sort, projection, intersection counting, prefix sum.
/// 2. `rasterize`: Intersection filling, tile sort, tile offsets, rasterization.
///
/// The split allows for an explicit GPU sync point between passes to read back
/// the exact number of intersections needed for buffer allocation.
pub trait SplatOps<B: Backend> {
    /// First pass: project gaussians and count intersections.
    fn project(
        camera: &Camera,
        img_size: glam::UVec2,
        means: FloatTensor<B>,
        log_scales: FloatTensor<B>,
        quats: FloatTensor<B>,
        sh_coeffs: FloatTensor<B>,
        raw_opacities: FloatTensor<B>,
        render_mode: SplatRenderMode,
    ) -> ProjectOutput<B>;

    /// Second pass: rasterize using projection data.
    fn rasterize(
        project_output: &ProjectOutput<B>,
        num_intersections: u32,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<B>, RenderAux<B>, IntTensor<B>);
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
