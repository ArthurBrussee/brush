//! Per-view aggregation into running min-alpha + visibility-weighted
//! colour blend, kept fully on the GPU.
//!
//! Removes the per-view CPU readback + aggregation loop that dominated
//! mesh extraction time: each `evaluate_alpha` call used to fire one
//! kernel + one readback + one host-side O(n) loop *per training view*
//! (~292 syncs per call, ×10 calls per extraction). With aggregators
//! living on the device this becomes one extra kernel launch per view
//! and a single readback at the very end.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 256;

/// Visibility weight exponent for the colour blend. Must stay in sync
/// with `COLOR_BLEND_POWER` in `crates/brush-mesh/src/extract.rs`.
const COLOR_BLEND_POWER: f32 = 8.0;

/// Fold this view's (alpha_int, rgb) into running aggregators.
///
/// `min_alpha` is updated to `min(min_alpha, alpha_int)`.
/// When `track_color` is true, `color_sum` and `weight_sum` get
/// `weight = (1 − alpha_int)^k`, `color_sum += weight · rgb`,
/// `weight_sum += weight`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn aggregate_alpha_kernel(
    view_alpha: &Tensor<f32>,
    view_rgb: &Tensor<f32>,
    min_alpha: &mut Tensor<f32>,
    color_sum: &mut Tensor<f32>,
    weight_sum: &mut Tensor<f32>,
    num_points: u32,
    #[comptime] track_color: bool,
) {
    let pid = ABSOLUTE_POS as u32;
    if pid >= num_points {
        terminate!();
    }

    let pidx = pid as usize;
    let a = view_alpha[pidx];

    if a < min_alpha[pidx] {
        min_alpha[pidx] = a;
    }

    if track_color {
        let vis = max(1.0f32 - a, 0.0f32);
        let w = f32::powf(vis, COLOR_BLEND_POWER);
        if w > 0.0f32 {
            let base = (pid * 3u32) as usize;
            color_sum[base] += w * view_rgb[base];
            color_sum[base + 1] += w * view_rgb[base + 1];
            color_sum[base + 2] += w * view_rgb[base + 2];
            weight_sum[pidx] += w;
        }
    }
}
