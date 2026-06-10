//! GPU-side state for the mesh-extraction binary search.
//!
//! Two small kernels that, together with [`super::integrate`], let the
//! whole bisection loop run on the GPU between an initial upload and a
//! final readback: no per-step CPU round-trips for midpoints, alphas,
//! or bracket state. Every crossing runs all steps; lab measurements
//! showed convergence-tracking machinery saved nothing measurable.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Writes each crossing's bracket midpoint into `mid_pos` for the
/// integrate kernel to consume as its `points` input.
#[cube(launch)]
pub fn compute_midpoints_kernel(
    left_pos: &Tensor<f32>,
    right_pos: &Tensor<f32>,
    mid_pos: &mut Tensor<f32>,
    n: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i >= n {
        terminate!();
    }
    let i3 = (i * 3u32) as usize;
    mid_pos[i3] = (left_pos[i3] + right_pos[i3]) * 0.5f32;
    mid_pos[i3 + 1] = (left_pos[i3 + 1] + right_pos[i3 + 1]) * 0.5f32;
    mid_pos[i3 + 2] = (left_pos[i3 + 2] + right_pos[i3 + 2]) * 0.5f32;
}

/// Folds the per-step integrate result back into the bracket state:
/// shrinks the endpoint whose sdf sign matches the midpoint's.
#[cube(launch)]
pub fn bracket_update_kernel(
    left_pos: &mut Tensor<f32>,
    right_pos: &mut Tensor<f32>,
    left_sdf: &mut Tensor<f32>,
    mid_pos: &Tensor<f32>,
    min_alpha: &Tensor<f32>,
    n: u32,
    iso: f32,
) {
    let i = ABSOLUTE_POS as u32;
    if i >= n {
        terminate!();
    }
    let iu = i as usize;
    let i3 = (i * 3u32) as usize;

    // Resolve alpha back from the running min. The integrate kernel only
    // ever writes values in [0, 1] (the per-view α_int), so the +∞
    // we initialise with means "never visited by any view" — match
    // the CPU path and treat those as fully unoccluded (α = 1).
    // We test `> 1.0` as the "never visited" proxy because `is_finite`
    // isn't available as a cube primitive.
    let a = min_alpha[iu];
    let resolved = if a > 1.0f32 { 1.0f32 } else { 1.0f32 - a };
    let sdf = resolved - iso;

    let lsdf = left_sdf[iu];
    // Same sign as left? Shrink left toward the midpoint.
    if (sdf < 0.0f32 && lsdf < 0.0f32) || (sdf > 0.0f32 && lsdf > 0.0f32) {
        left_pos[i3] = mid_pos[i3];
        left_pos[i3 + 1] = mid_pos[i3 + 1];
        left_pos[i3 + 2] = mid_pos[i3 + 2];
        left_sdf[iu] = sdf;
    } else {
        right_pos[i3] = mid_pos[i3];
        right_pos[i3 + 1] = mid_pos[i3 + 1];
        right_pos[i3 + 2] = mid_pos[i3 + 2];
    }
}
