//! GPU-side state for the mesh-extraction binary search.
//!
//! Two small kernels that, together with [`super::integrate`], let the
//! whole bisection loop run on the GPU between an initial upload and a
//! final readback — no per-step CPU↔GPU round-trips for midpoints,
//! alphas, or bracket state.
//!
//! ## [`compute_midpoints_kernel`]
//! For each active crossing `i`, reads its original-crossing id
//! `orig = active_in[i]`, looks up `left_pos[orig]` and `right_pos[orig]`,
//! and writes the midpoint into `mid_pos[i]`. The midpoint tensor is
//! then handed to the tiled integrate as its `points` input.
//!
//! ## [`bracket_update_kernel`]
//! For each active crossing `i`:
//! - reads the just-integrated `min_alpha[i]`, converts it back to
//!   `sdf = (1 − min_alpha) − iso` (matching the resolve in
//!   `evaluate_alpha`'s alpha return)
//! - if `|sdf| < converge_eps`, locks both endpoints at the current
//!   midpoint and *does not* re-add to the active list
//! - otherwise bisects (shrinks left or right based on sign-match with
//!   `left_sdf`) and atomically appends `orig` to `active_out` via
//!   `Atomic::fetch_add` on `active_count`
//!
//! Host code reads back `active_count[0]` (one u32) after the kernel
//! to size the next step's dispatch — that's the only per-step
//! readback in the fused loop.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Gathers the active crossings' left/right endpoints and writes the
/// midpoints into a contiguous `[n_active, 3]` tensor for the integrate
/// kernel to consume.
#[cube(launch)]
pub fn compute_midpoints_kernel(
    left_pos: &Tensor<f32>,
    right_pos: &Tensor<f32>,
    active_in: &Tensor<u32>,
    mid_pos: &mut Tensor<f32>,
    n_active: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i >= n_active {
        terminate!();
    }
    let orig = active_in[i as usize];
    let src = (orig * 3u32) as usize;
    let dst = (i * 3u32) as usize;
    mid_pos[dst] = (left_pos[src] + right_pos[src]) * 0.5f32;
    mid_pos[dst + 1] = (left_pos[src + 1] + right_pos[src + 1]) * 0.5f32;
    mid_pos[dst + 2] = (left_pos[src + 2] + right_pos[src + 2]) * 0.5f32;
}

/// Folds the per-step integrate result back into the bracket state.
/// Converged crossings (|sdf| < ε) get locked at the current midpoint
/// and drop out of the active list; survivors update one endpoint and
/// get atomically appended to `active_out`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn bracket_update_kernel(
    left_pos: &mut Tensor<f32>,
    right_pos: &mut Tensor<f32>,
    left_sdf: &mut Tensor<f32>,
    right_sdf: &mut Tensor<f32>,
    active_in: &Tensor<u32>,
    mid_pos: &Tensor<f32>,
    min_alpha: &Tensor<f32>,
    active_out: &mut Tensor<u32>,
    active_count: &mut Tensor<Atomic<u32>>,
    n_active: u32,
    iso: f32,
    converge_eps: f32,
) {
    let i = ABSOLUTE_POS as u32;
    if i >= n_active {
        terminate!();
    }
    let orig = active_in[i as usize];
    let orig_u = orig as usize;
    let orig3 = (orig * 3u32) as usize;
    let i3 = (i * 3u32) as usize;

    // Resolve alpha back from the running min. The integrate kernel only
    // ever writes values in [0, 1] (the per-view α_int), so the +∞
    // we initialise with means "never visited by any view" — match
    // the CPU path and treat those as fully unoccluded (α = 1).
    // We test `> 1.0` as the "never visited" proxy because `is_finite`
    // isn't available as a cube primitive.
    let a = min_alpha[i as usize];
    let resolved = if a > 1.0f32 { 1.0f32 } else { 1.0f32 - a };
    let sdf = resolved - iso;

    let mid_x = mid_pos[i3];
    let mid_y = mid_pos[i3 + 1];
    let mid_z = mid_pos[i3 + 2];

    if f32::abs(sdf) < converge_eps {
        // Lock both endpoints at the midpoint; finish() returns the
        // average of left+right so this is the converged position.
        left_pos[orig3] = mid_x;
        left_pos[orig3 + 1] = mid_y;
        left_pos[orig3 + 2] = mid_z;
        right_pos[orig3] = mid_x;
        right_pos[orig3 + 1] = mid_y;
        right_pos[orig3 + 2] = mid_z;
    } else {
        let lsdf = left_sdf[orig_u];
        // Same sign as left? Shrink left toward the midpoint.
        if (sdf < 0.0f32 && lsdf < 0.0f32) || (sdf > 0.0f32 && lsdf > 0.0f32) {
            left_pos[orig3] = mid_x;
            left_pos[orig3 + 1] = mid_y;
            left_pos[orig3 + 2] = mid_z;
            left_sdf[orig_u] = sdf;
        } else {
            right_pos[orig3] = mid_x;
            right_pos[orig3 + 1] = mid_y;
            right_pos[orig3 + 2] = mid_z;
            right_sdf[orig_u] = sdf;
        }
        // Append to next-step's active list.
        let slot = Atomic::fetch_add(&active_count[0], 1u32);
        active_out[slot as usize] = orig;
    }
}
