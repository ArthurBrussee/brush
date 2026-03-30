#![recursion_limit = "256"]

use burn::prelude::Backend;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;
use burn_fusion::Fusion;
use burn_wgpu::WgpuRuntime;
use camera::Camera;
use clap::ValueEnum;
use glam::Vec3;
use render_aux::{CullOutput, CullReadback};

use crate::gaussian_splats::SplatRenderMode;
pub use crate::gaussian_splats::{TextureMode, render_splats};
pub use crate::render_aux::RenderAux;

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
/// The pipeline has two phases with a single async readback point:
/// 1. `project_cull`: Culling, depth sort, tile intersection counting, prefix sum.
/// 2. Readback `num_visible` + `num_intersections` from [`CullReadback::read_counts`].
/// 3. `rasterize`: Projection, intersection filling, tile sort, rasterization.
pub trait SplatOps<B: Backend> {
    /// Phase 1: Cull invisible splats, depth sort, count tile intersections, prefix sum.
    ///
    /// Returns the cull output (for rasterize) and readback tensors (for async count readback).
    fn project_cull(
        camera: &Camera,
        img_size: glam::UVec2,
        transforms: FloatTensor<B>,
        raw_opacities: FloatTensor<B>,
        render_mode: SplatRenderMode,
    ) -> (CullOutput<B>, CullReadback<B>);

    /// Phase 2: Project visible splats, fill intersections, tile sort, rasterize.
    ///
    /// `num_visible` and `num_intersections` are the CPU-side readbacks from
    /// [`CullOutput::read_counts`].
    #[allow(clippy::too_many_arguments)]
    fn rasterize(
        cull_output: &CullOutput<B>,
        num_visible: u32,
        num_intersections: u32,
        transforms: FloatTensor<B>,
        sh_coeffs: FloatTensor<B>,
        raw_opacities: FloatTensor<B>,
        render_mode: SplatRenderMode,
        background: Vec3,
        bwd_info: bool,
    ) -> (FloatTensor<B>, RenderAux<B>, FloatTensor<B>, IntTensor<B>);
    //    out_img,       render_aux,     projected_splats, compact_gid_from_isect
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
