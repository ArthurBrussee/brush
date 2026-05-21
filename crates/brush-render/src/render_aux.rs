use burn::{
    Tensor,
    backend::{
        Backend,
        tensor::{FloatTensor, IntTensor},
    },
    tensor::Int,
};

use crate::camera::Camera;

/// Per-render dispatch state shared between the forward render and the
/// autodiff backward. The cube-side `ProjectUniforms` launch arg is built
/// on demand via [`Camera::to_project_uniforms_launch`].
#[derive(Debug, Clone, Copy)]
pub struct RenderState {
    pub camera: Camera,
    pub img_size: glam::UVec2,
    pub tile_bounds: glam::UVec2,
    pub sh_degree: u32,
    pub total_splats: u32,
    pub num_visible: u32,
}

/// Internal render output used by kernel impls. Holds backend primitives.
#[derive(Debug, Clone)]
pub struct RenderOutput<B: Backend> {
    pub out_img: FloatTensor<B>,
    pub aux: RenderAuxInner<B>,
    // State needed by the backward pass; non-diff callers can ignore these.
    pub projected_splats: FloatTensor<B>,
    pub compact_gid_from_isect: IntTensor<B>,
    pub render_state: RenderState,
    pub global_from_compact_gid: IntTensor<B>,
}

impl<B: Backend> RenderOutput<B> {
    /// Count-only invariants — cheap (no readback), always on.
    pub fn validate_counts(&self) {
        let num_visible = self.aux.num_visible;
        let num_intersections = self.aux.num_intersections;
        let total_splats = self.render_state.total_splats;
        assert!(
            num_visible <= total_splats,
            "num_visible ({num_visible}) > total_splats ({total_splats})",
        );
        let tile_bounds = self.render_state.tile_bounds;
        let max_isects = (num_visible as u64) * (tile_bounds.x as u64) * (tile_bounds.y as u64);
        assert!(
            (num_intersections as u64) <= max_isects,
            "num_intersections ({num_intersections}) > max possible {max_isects}",
        );
    }

    /// Full validation; gated on `debug-validation` feature / `cfg(test)`.
    /// Takes self by value to avoid Send issues with the async readbacks.
    #[allow(unused_variables)]
    pub async fn validate(self) {
        self.validate_counts();
        // Heavy validation lives at the public API boundary where we have Tensor<D>.
    }
}

/// Internal aux struct holding backend primitives. Used by the kernel
/// pipeline and the backward registration.
#[derive(Debug, Clone)]
pub struct RenderAuxInner<B: Backend> {
    pub num_visible: u32,
    pub num_intersections: u32,
    pub visible: FloatTensor<B>,
    /// Per-splat maximum screen-space radius in pixels (global-gid indexed).
    /// Zero for splats that were culled / invisible in this view.
    pub max_radius: FloatTensor<B>,
    pub tile_offsets: IntTensor<B>,
    pub img_size: glam::UVec2,
}

/// Public, backend-agnostic aux. Holds `Tensor<D>` for the user.
#[derive(Debug, Clone)]
pub struct RenderAux {
    pub num_visible: u32,
    pub num_intersections: u32,
    pub visible: Tensor<1>,
    /// Per-splat maximum screen-space radius in pixels (global-gid indexed).
    /// Zero for splats that were culled / invisible in this view.
    pub max_radius: Tensor<1>,
    pub tile_offsets: Tensor<3, Int>,
    pub img_size: glam::UVec2,
}

impl RenderAux {
    /// Calculate tile depth map for visualization.
    pub fn calc_tile_depth(&self) -> Tensor<2, Int> {
        use crate::shaders::helpers::TILE_WIDTH;
        use burn::tensor::s;

        let tile_offsets = self.tile_offsets.clone();
        let max = tile_offsets.clone().slice(s![.., .., 1]);
        let min = tile_offsets.slice(s![.., .., 0]);
        let [w, h] = self.img_size.into();
        let [ty, tx] = [h.div_ceil(TILE_WIDTH), w.div_ceil(TILE_WIDTH)];
        (max - min).reshape([ty as usize, tx as usize])
    }
}
