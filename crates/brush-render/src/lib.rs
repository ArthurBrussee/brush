#![allow(clippy::too_many_arguments)]
#![allow(clippy::single_range_in_vec_init)]
use burn::prelude::Tensor;
use burn::tensor::{ElementConversion, Int};
use burn_jit::JitBackend;
use burn_wgpu::WgpuRuntime;
use camera::Camera;

mod burn_glue;
mod dim_check;
mod kernels;
mod safetensor_utils;
mod shaders;

pub mod bounding_box;
pub mod camera;
pub mod gaussian_splats;
pub mod render;

#[derive(Debug, Clone)]
pub struct RenderAux<B: Backend> {
    /// The packed projected splat information, see ProjectedSplat in helpers.wgsl
    pub projected_splats: B::FloatTensorPrimitive,
    pub uniforms_buffer: B::IntTensorPrimitive,
    pub num_intersections: B::IntTensorPrimitive,
    pub num_visible: B::IntTensorPrimitive,
    pub final_index: B::IntTensorPrimitive,
    pub cum_tiles_hit: B::IntTensorPrimitive,
    pub tile_bins: B::IntTensorPrimitive,
    pub compact_gid_from_isect: B::IntTensorPrimitive,
    pub global_from_compact_gid: B::IntTensorPrimitive,
    pub radii: B::FloatTensorPrimitive,
}

#[derive(Debug, Clone)]
pub struct RenderStats {
    pub num_visible: u32,
    pub num_intersections: u32,
}

impl<B: Backend> RenderAux<B> {
    pub async fn read_num_visible(&self) -> u32 {
        Tensor::<B, 1, Int>::from_primitive(self.num_visible.clone())
            .into_scalar_async()
            .await
            .elem()
    }

    pub async fn read_num_intersections(&self) -> u32 {
        Tensor::<B, 1, Int>::from_primitive(self.num_intersections.clone())
            .into_scalar_async()
            .await
            .elem()
    }

    pub fn read_tile_depth(&self) -> Tensor<B, 2, Int> {
        let bins = Tensor::from_primitive(self.tile_bins.clone());
        let [ty, tx, _] = bins.dims();
        let max = bins.clone().slice([0..ty, 0..tx, 1..2]).squeeze(2);
        let min = bins.clone().slice([0..ty, 0..tx, 0..1]).squeeze(2);
        max - min
    }
}

pub struct SplatGrads<B: Backend> {
    v_means: B::FloatTensorPrimitive,
    v_quats: B::FloatTensorPrimitive,
    v_scales: B::FloatTensorPrimitive,
    v_coeffs: B::FloatTensorPrimitive,
    v_raw_opac: B::FloatTensorPrimitive,
    v_xy: B::FloatTensorPrimitive,
}

#[derive(Debug, Clone)]
pub struct GaussianBackwardState<B: Backend> {
    means: B::FloatTensorPrimitive,
    log_scales: B::FloatTensorPrimitive,
    quats: B::FloatTensorPrimitive,
    raw_opac: B::FloatTensorPrimitive,
    out_img: B::FloatTensorPrimitive,
    sh_degree: u32,
    aux: RenderAux<B>,
}

// Custom operations in Burn work by extending the backend with an extra func.
pub trait Backend: burn::tensor::backend::Backend {
    /// Render splats to a buffer.
    ///
    /// This projects the gaussians, sorts them, and rasterizes them to a buffer, in a
    /// differentiable way.
    /// The arguments are all passed as raw tensors. See [`Splats`] for a convenient Module that wraps this fun
    /// The ['xy_dummy'] variable is only used to carry screenspace xy gradients.
    /// This function can optionally render a "u32" buffer, which is a packed RGBA (8 bits per channel)
    /// buffer. This is useful when the results need to be displayed immediatly.
    fn render_splats(
        cam: &Camera,
        img_size: glam::UVec2,
        means: Self::FloatTensorPrimitive,
        xy_grad_dummy: Self::FloatTensorPrimitive,
        log_scales: Self::FloatTensorPrimitive,
        quats: Self::FloatTensorPrimitive,
        sh_coeffs: Self::FloatTensorPrimitive,
        raw_opacity: Self::FloatTensorPrimitive,
        render_u32_buffer: bool,
    ) -> (Self::FloatTensorPrimitive, RenderAux<Self>);

    /// Backward pass for render_splats.
    ///
    /// Do not use directly, render_splats will use this to calculate gradients.
    #[allow(unused_variables)]
    fn render_splats_bwd(
        state: GaussianBackwardState<Self>,
        v_output: Self::FloatTensorPrimitive,
    ) -> SplatGrads<Self> {
        panic!("Do not call this manually.");
    }
}

pub trait AutodiffBackend:
    Backend + burn::tensor::backend::AutodiffBackend<InnerBackend: Backend>
{
}

type InnerWgpu = JitBackend<WgpuRuntime, f32, i32, u32>;
