//! Per-query-point opacity-along-ray integrator.
//!
//! Implements the GOF "integrate" pass for mesh extraction: given a batch of
//! world-space query points and the per-view tile-sorted Gaussian list
//! already produced by the regular forward render, evaluates
//!
//! ```text
//! α(p) = 1 − ∏_g (1 − α_g(p))
//! ```
//!
//! where `α_g(p) = opac_g · exp(−½ (A t² + B t + C))` evaluated at
//! `t = min(t*, t_point)`, `t* = −B / (2A)` is the depth of the Gaussian's
//! peak along the camera ray, and `(A, B, C)` come from the Gaussian's
//! `Σ⁻¹` projected onto the ray (the "view2gaussian" formulation from
//! GOF's `forward.cu`).
//!
//! One thread per point: each thread projects the point to its tile, walks
//! the tile's depth-sorted Gaussian list (reusing `compact_gid_from_isect`
//! and `tile_offsets` from the forward render), and accumulates
//! transmittance until either the list ends or `T < 1e-4`.
//!
//! Camera ray reconstruction sidesteps the projection inversion (which
//! would be camera-model-dependent): the ray direction in view space is
//! `(x_v, y_v, z_v) / z_v`, with `t = z_v` so `α_g` evaluation is
//! identical across pinhole and fisheye.

use super::helpers::{TILE_WIDTH, read_quat_unorm, read_scale, sigmoid, world_to_cam};
use super::types::{ProjectUniforms, Vec3A};
use crate::kernels::camera_model::{CameraModel, project};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const WG_SIZE: u32 = 128;

// Match GOF's first-pass `NEAR_PLANE = 0.2`: gaussians whose ray-hit
// depth `t* = −B/(2A)` is closer than this are skipped (their alpha
// contribution would be evaluated past the camera near-plane). Brush's
// own rasterizer uses ~0.01, but for *integrate*-style ray-marching we
// want the GOF value to keep close-camera splats from injecting noise.
const NEAR_PLANE: f32 = 0.2;
const ALPHA_CUTOFF: f32 = 1.0 / 255.0;
const ALPHA_CLAMP: f32 = 0.999;
const T_EARLY_OUT: f32 = 1.0e-4;

/// Visibility-weight exponent for the colour blend. Must stay in sync
/// with `COLOR_BLEND_POWER` in `crates/brush-mesh/src/extract.rs`.
const COLOR_BLEND_POWER: f32 = 8.0;

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn integrate_alpha_kernel(
    transforms: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    compact_gid_from_isect: &Tensor<u32>,
    tile_offsets: &Tensor<u32>,
    global_from_compact_gid: &Tensor<u32>,
    rendered_image: &Tensor<f32>,
    points: &Tensor<f32>,
    // Running aggregators (fused — was a separate `aggregate_alpha`
    // kernel before). Each thread owns one `pid` slot so no atomics.
    min_alpha: &mut Tensor<f32>,
    color_sum: &mut Tensor<f32>,
    weight_sum: &mut Tensor<f32>,
    num_points: u32,
    u: ProjectUniforms,
    #[comptime] camera_model: CameraModel,
    #[comptime] track_color: bool,
) {
    let pid = ABSOLUTE_POS as u32;
    if pid >= num_points {
        terminate!();
    }

    let pidx3 = (pid * 3u32) as usize;
    let p_world = Vec3A::new(points[pidx3], points[pidx3 + 1], points[pidx3 + 2]);
    let p_cam = world_to_cam(p_world, u);

    let depth = p_cam.z();
    if !(depth > NEAR_PLANE) {
        // Behind near plane — this view has no line of sight. Skip
        // entirely: not touching `min_alpha` leaves prior views' votes
        // (or the host's +∞ init, which the host maps to alpha = 1.0
        // "open space") in place. Equivalent to GOF's untouched-pixel
        // semantic without the redundant 1.0 write.
        terminate!();
    }

    // View-space ray direction normalised so its z-component is 1, matching
    // GOF's `ray_point` convention. `t = depth` is then the ray parameter
    // at the query point — no projection inverse needed.
    let inv_z = 1.0f32 / depth;
    let ray_dir = Vec3A::new(p_cam.x() * inv_z, p_cam.y() * inv_z, 1.0f32);

    // Find the tile this point projects into. We re-project through the
    // canonical camera model so distortion is handled consistently with
    // the forward renderer that produced `tile_offsets`.
    let (px, py) = project(p_cam, u.pinhole_params, camera_model);
    if !(px >= 0.0f32 && px < u.img_w as f32 && py >= 0.0f32 && py < u.img_h as f32) {
        // Off-screen — same semantic as the behind-near-plane case
        // above. Leaving the aggregators untouched means the host's
        // +∞ init (mapped to alpha = 1.0) wins if every view skips.
        terminate!();
    }
    let tx = (px as u32) / TILE_WIDTH;
    let ty = (py as u32) / TILE_WIDTH;
    let tile_id = ty * u.tile_bw + tx;

    // Sample the brush-rendered RGB at the vertex's projected pixel.
    // The host calls `render(.., RasterPass::Forward)`, which stores
    // `[H, W]` packed u32 RGBA per pixel (one f32 word per pixel whose
    // bits encode `r | g<<8 | b<<16 | a<<24`). We bitcast to u32 and
    // unpack. Forward (vs Backward) is critical here because the host
    // also reads `tile_offsets` from the same render — Backward shrinks
    // it to the *pixel-centre* last_useful_isect, which can be tighter
    // than GOF's 5-sub-pixel-corner contributor range. Forward keeps the
    // untruncated tile range so sub-pixel splats contributing at corners
    // but not centres still reach this kernel's per-query-point walk.
    let pix_x = px as u32;
    let pix_y = py as u32;
    let pix_idx = (pix_y * u.img_w + pix_x) as usize;
    let pix_u32 = u32::reinterpret(rendered_image[pix_idx]);
    let r_u = pix_u32 & 0xFFu32;
    let g_u = (pix_u32 >> 8u32) & 0xFFu32;
    let b_u = (pix_u32 >> 16u32) & 0xFFu32;
    let rgb_r = f32::cast_from(r_u) / 255.0f32;
    let rgb_g = f32::cast_from(g_u) / 255.0f32;
    let rgb_b = f32::cast_from(b_u) / 255.0f32;

    let range_lo = tile_offsets[(tile_id * 2u32) as usize];
    let range_hi = tile_offsets[(tile_id * 2u32 + 1u32) as usize];

    let mut t_acc = 1.0f32;
    let mut k = range_lo;
    while k < range_hi {
        if t_acc < T_EARLY_OUT {
            break;
        }
        let compact_gid = compact_gid_from_isect[k as usize];
        let global_gid = global_from_compact_gid[compact_gid as usize];
        let base = (global_gid * 10u32) as usize;

        let g_mean_w = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
        let g_scale = read_scale(transforms, base);
        let g_quat = read_quat_unorm(transforms, base).normalize();

        // Scales and opacity are already *effective*: `extract_mesh`
        // calls `splats.bake_min_scale()` up front so the mip 3D-filter
        // floor (`sqrt(s² + f²)` for scales, `sqrt(det/det)` for
        // opacity — see `fold_min_scale`) is folded into the raw
        // tensors. Reading `transforms` / `raw_opacities` directly here
        // therefore matches what brush's renderer uses, with no risk of
        // under- or double-counting the filter.
        let g_mean_c = world_to_cam(g_mean_w, u);
        let r_gv = u.view_rotation().mul_mat3(g_quat.to_mat3());

        // v = S⁻¹ · R_gvᵀ · d_view
        let r_gv_t_d = r_gv.transpose_mul_vec3(ray_dir);
        let v = Vec3A::new(
            r_gv_t_d.x() / g_scale.x(),
            r_gv_t_d.y() / g_scale.y(),
            r_gv_t_d.z() / g_scale.z(),
        );
        // o = −S⁻¹ · R_gvᵀ · mean_view
        let r_gv_t_mc = r_gv.transpose_mul_vec3(g_mean_c);
        let o = Vec3A::new(
            -r_gv_t_mc.x() / g_scale.x(),
            -r_gv_t_mc.y() / g_scale.y(),
            -r_gv_t_mc.z() / g_scale.z(),
        );

        let a = v.dot(v);
        let b_coef = 2.0f32 * v.dot(o);
        let c_coef = o.dot(o);

        // Skip degenerate rays (parallel-to-Gaussian or collapsed scales —
        // we should never see them given upstream culling, but the
        // division below requires `A > 0`).
        if a > 1.0e-20f32 {
            // Peak depth along the ray; clamp so we don't integrate past
            // the query point.
            let t_star = -b_coef / (2.0f32 * a);
            let t_eval = min(t_star, depth);

            if t_eval > NEAR_PLANE {
                let power = -0.5f32 * (a * t_eval * t_eval + b_coef * t_eval + c_coef);
                // `power` peaks at 0 (with t_eval = t_star); the min with
                // 0 below catches FP overshoot when the math goes
                // slightly positive at the boundary.
                let power_safe = min(power, 0.0f32);
                let raw_opac = raw_opacities[global_gid as usize];
                let alpha_raw = sigmoid(raw_opac) * f32::exp(power_safe);
                let alpha = min(ALPHA_CLAMP, alpha_raw);
                if alpha >= ALPHA_CUTOFF {
                    t_acc *= 1.0f32 - alpha;
                }
            }
        }
        k += 1u32;
    }

    // Fold this view's contribution into the running per-point
    // aggregators. Each `pid` is owned by a single thread (one thread
    // per query point), so the read-modify-writes are race-free across
    // views (every view = a separate sequential kernel launch).
    let alpha = 1.0f32 - t_acc;
    if alpha < min_alpha[pid as usize] {
        min_alpha[pid as usize] = alpha;
    }
    if track_color {
        let vis = max(1.0f32 - alpha, 0.0f32);
        let w = f32::powf(vis, COLOR_BLEND_POWER);
        if w > 0.0f32 {
            let base = (pid * 3u32) as usize;
            color_sum[base] += w * rgb_r;
            color_sum[base + 1] += w * rgb_g;
            color_sum[base + 2] += w * rgb_b;
            weight_sum[pid as usize] += w;
        }
    }
}
