//! Fused PPISP kernels (port of NVIDIA's `ppisp` reference CUDA kernels).
//!
//! Per-pixel pipeline: exposure (per frame) → vignetting (per camera) →
//! color homography (per frame) → CRF tone curve (per camera). The backward
//! recomputes the forward stage-by-stage and chains gradients in reverse,
//! accumulating the 36 parameter partials per thread, reducing them
//! per-cube (plane sums + shared memory), and writing one `[36]` row per
//! cube to a partials buffer — the host then `sum`s the buffer, which keeps
//! the parameter gradients deterministic with no atomics.
//!
//! Parameter layouts (flattened f32):
//! - exposure `[num_frames]` — log2-exposure offset.
//! - vignetting `[num_cameras, 3, 5]` — per channel `cx, cy, a0, a1, a2`.
//! - color `[num_frames, 8]` — latent 2D offsets for the B/R/G/N control
//!   chromaticities of the homography.
//! - crf `[num_cameras, 3, 4]` — per channel raw `toe, shoulder, gamma,
//!   center` (softplus/sigmoid transformed in-kernel).

use brush_cube::Vec3A;
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub const BLOCK_SIZE: u32 = 128;
/// Number of scalar parameter gradients reduced per pixel:
/// 1 exposure + 15 vignetting + 8 color + 12 CRF.
pub const NUM_PARAM_GRADS: u32 = 36;
/// Worst-case subgroup count per cube (`PLANE_DIM >= 4` everywhere).
const MAX_SUBGROUPS: u32 = BLOCK_SIZE / 4;

use crate::ppisp_math::{
    LN2, color_correct_bwd, color_correct_fwd, crf_channel_bwd, crf_channel_fwd, homography,
    vig_falloff_raw, vig_uv,
};

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------

#[cube(launch)]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub fn ppisp_fwd_kernel(
    exposure: &Tensor<f32>,
    vignetting: &Tensor<f32>,
    color: &Tensor<f32>,
    crf: &Tensor<f32>,
    rgb: &Tensor<f32>,
    out: &mut Tensor<f32>,
    img_h: u32,
    img_w: u32,
    camera_idx: u32,
    frame_idx: u32,
    channels: u32,
    #[comptime] has_alpha: bool,
    #[comptime] with_frame: bool,
    #[comptime] with_vignetting: bool,
    #[comptime] with_crf: bool,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    if idx >= img_h * img_w {
        terminate!();
    }
    let hi = idx / img_w;
    let wi = idx % img_w;
    let base = (idx * channels) as usize;

    let mut c = Vec3A::new(rgb[base], rgb[base + 1], rgb[base + 2]);

    // 1. Exposure (log2 space): 2^e = exp(e · ln 2).
    if with_frame {
        c = c.scale(f32::exp(exposure[frame_idx as usize] * LN2));
    }

    // 2. Vignetting (per channel).
    if with_vignetting {
        let (uvx, uvy) = vig_uv(wi, hi, img_w, img_h);
        let vbase = (camera_idx * 15u32) as usize;
        let f_r = clamp(
            vig_falloff_raw(
                uvx,
                uvy,
                vignetting[vbase],
                vignetting[vbase + 1],
                vignetting[vbase + 2],
                vignetting[vbase + 3],
                vignetting[vbase + 4],
            ),
            0.0f32,
            1.0f32,
        );
        let f_g = clamp(
            vig_falloff_raw(
                uvx,
                uvy,
                vignetting[vbase + 5],
                vignetting[vbase + 6],
                vignetting[vbase + 7],
                vignetting[vbase + 8],
                vignetting[vbase + 9],
            ),
            0.0f32,
            1.0f32,
        );
        let f_b = clamp(
            vig_falloff_raw(
                uvx,
                uvy,
                vignetting[vbase + 10],
                vignetting[vbase + 11],
                vignetting[vbase + 12],
                vignetting[vbase + 13],
                vignetting[vbase + 14],
            ),
            0.0f32,
            1.0f32,
        );
        c = Vec3A::new(c.x() * f_r, c.y() * f_g, c.z() * f_b);
    }

    // 3. Color homography.
    if with_frame {
        let cbase = (frame_idx * 8u32) as usize;
        let (h, _) = homography(
            color[cbase],
            color[cbase + 1],
            color[cbase + 2],
            color[cbase + 3],
            color[cbase + 4],
            color[cbase + 5],
            color[cbase + 6],
            color[cbase + 7],
        );
        c = color_correct_fwd(c, h);
    }

    // 4. CRF (per channel, on [0,1]-clamped input).
    if with_crf {
        let kbase = (camera_idx * 12u32) as usize;
        let xr = clamp(c.x(), 0.0f32, 1.0f32);
        let xg = clamp(c.y(), 0.0f32, 1.0f32);
        let xb = clamp(c.z(), 0.0f32, 1.0f32);
        out[base] = crf_channel_fwd(
            xr,
            crf[kbase],
            crf[kbase + 1],
            crf[kbase + 2],
            crf[kbase + 3],
        );
        out[base + 1] = crf_channel_fwd(
            xg,
            crf[kbase + 4],
            crf[kbase + 5],
            crf[kbase + 6],
            crf[kbase + 7],
        );
        out[base + 2] = crf_channel_fwd(
            xb,
            crf[kbase + 8],
            crf[kbase + 9],
            crf[kbase + 10],
            crf[kbase + 11],
        );
    } else {
        out[base] = c.x();
        out[base + 1] = c.y();
        out[base + 2] = c.z();
    }
    if has_alpha {
        out[base + 3] = rgb[base + 3];
    }
}
// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------

/// Per-pixel backward. Writes `dL/drgb_in` and one `[NUM_PARAM_GRADS]` row of
/// parameter-gradient partials per cube (deterministically summed on the
/// host). All threads participate in the reduction, so out-of-range threads
/// contribute zeros instead of terminating before the barrier.
#[cube(launch)]
#[allow(
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::fn_params_excessive_bools
)]
pub fn ppisp_bwd_kernel(
    exposure: &Tensor<f32>,
    vignetting: &Tensor<f32>,
    color: &Tensor<f32>,
    crf: &Tensor<f32>,
    rgb: &Tensor<f32>,
    v_out: &Tensor<f32>,
    grad_rgb: &mut Tensor<f32>,
    partials: &mut Tensor<f32>,
    img_h: u32,
    img_w: u32,
    camera_idx: u32,
    frame_idx: u32,
    channels: u32,
    #[comptime] has_alpha: bool,
    #[comptime] with_frame: bool,
    #[comptime] with_vignetting: bool,
    #[comptime] with_crf: bool,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;

    let mut pg = Array::<f32>::new(NUM_PARAM_GRADS as usize);
    #[unroll]
    for p in 0u32..NUM_PARAM_GRADS {
        pg[p as usize] = 0.0f32;
    }

    if idx < img_h * img_w {
        let hi = idx / img_w;
        let wi = idx % img_w;
        let base = (idx * channels) as usize;

        let rgb_input = Vec3A::new(rgb[base], rgb[base + 1], rgb[base + 2]);

        // --- Recompute forward, keeping each stage's input. Disabled stages
        // collapse to identity at compile time. ---
        let mut exp_param = 0.0f32;
        let mut rgb_after_exp = rgb_input;
        if with_frame {
            exp_param = exposure[frame_idx as usize];
            rgb_after_exp = rgb_input.scale(f32::exp(exp_param * LN2));
        }

        let mut uvx = 0.0f32;
        let mut uvy = 0.0f32;
        let mut raw_r = 1.0f32;
        let mut raw_g = 1.0f32;
        let mut raw_b = 1.0f32;
        let mut f_r = 1.0f32;
        let mut f_g = 1.0f32;
        let mut f_b = 1.0f32;
        let vbase = (camera_idx * 15u32) as usize;
        if with_vignetting {
            let (ux, uy) = vig_uv(wi, hi, img_w, img_h);
            uvx = ux;
            uvy = uy;
            raw_r = vig_falloff_raw(
                uvx,
                uvy,
                vignetting[vbase],
                vignetting[vbase + 1],
                vignetting[vbase + 2],
                vignetting[vbase + 3],
                vignetting[vbase + 4],
            );
            raw_g = vig_falloff_raw(
                uvx,
                uvy,
                vignetting[vbase + 5],
                vignetting[vbase + 6],
                vignetting[vbase + 7],
                vignetting[vbase + 8],
                vignetting[vbase + 9],
            );
            raw_b = vig_falloff_raw(
                uvx,
                uvy,
                vignetting[vbase + 10],
                vignetting[vbase + 11],
                vignetting[vbase + 12],
                vignetting[vbase + 13],
                vignetting[vbase + 14],
            );
            f_r = clamp(raw_r, 0.0f32, 1.0f32);
            f_g = clamp(raw_g, 0.0f32, 1.0f32);
            f_b = clamp(raw_b, 0.0f32, 1.0f32);
        }
        let rgb_after_vig = Vec3A::new(
            rgb_after_exp.x() * f_r,
            rgb_after_exp.y() * f_g,
            rgb_after_exp.z() * f_b,
        );

        let cbase = (frame_idx * 8u32) as usize;
        let mut c0 = 0.0f32;
        let mut c1 = 0.0f32;
        let mut c2 = 0.0f32;
        let mut c3 = 0.0f32;
        let mut c4 = 0.0f32;
        let mut c5 = 0.0f32;
        let mut c6 = 0.0f32;
        let mut c7 = 0.0f32;
        let mut rgb_after_color = rgb_after_vig;
        if with_frame {
            c0 = color[cbase];
            c1 = color[cbase + 1];
            c2 = color[cbase + 2];
            c3 = color[cbase + 3];
            c4 = color[cbase + 4];
            c5 = color[cbase + 5];
            c6 = color[cbase + 6];
            c7 = color[cbase + 7];
            let (h, _) = homography(c0, c1, c2, c3, c4, c5, c6, c7);
            rgb_after_color = color_correct_fwd(rgb_after_vig, h);
        }

        // --- Backward chain (reverse order) ---
        let mut grad = Vec3A::new(v_out[base], v_out[base + 1], v_out[base + 2]);

        // 4. CRF backward (operates on the [0,1]-clamped input; the clamp's
        // gradient gate matches the reference, which also drops gradient
        // outside the open interval).
        if with_crf {
            let kbase = (camera_idx * 12u32) as usize;
            let xr = clamp(rgb_after_color.x(), 0.0f32, 1.0f32);
            let xg = clamp(rgb_after_color.y(), 0.0f32, 1.0f32);
            let xb = clamp(rgb_after_color.z(), 0.0f32, 1.0f32);
            let (gx_r, gt_r, gs_r, gg_r, gc_r) = crf_channel_bwd(
                xr,
                crf[kbase],
                crf[kbase + 1],
                crf[kbase + 2],
                crf[kbase + 3],
                grad.x(),
            );
            let (gx_g, gt_g, gs_g, gg_g, gc_g) = crf_channel_bwd(
                xg,
                crf[kbase + 4],
                crf[kbase + 5],
                crf[kbase + 6],
                crf[kbase + 7],
                grad.y(),
            );
            let (gx_b, gt_b, gs_b, gg_b, gc_b) = crf_channel_bwd(
                xb,
                crf[kbase + 8],
                crf[kbase + 9],
                crf[kbase + 10],
                crf[kbase + 11],
                grad.z(),
            );
            pg[24] += gt_r;
            pg[25] += gs_r;
            pg[26] += gg_r;
            pg[27] += gc_r;
            pg[28] += gt_g;
            pg[29] += gs_g;
            pg[30] += gg_g;
            pg[31] += gc_g;
            pg[32] += gt_b;
            pg[33] += gs_b;
            pg[34] += gg_b;
            pg[35] += gc_b;
            grad = Vec3A::new(gx_r, gx_g, gx_b);
        }

        // 3. Color correction backward (latent grads land in pg[16..24]).
        if with_frame {
            grad = color_correct_bwd(
                rgb_after_vig,
                c0,
                c1,
                c2,
                c3,
                c4,
                c5,
                c6,
                c7,
                grad,
                &mut pg,
                16u32,
            );
        }

        // 2. Vignetting backward (per channel, replaying the falloff).
        if with_vignetting {
            let inside_r = raw_r >= 0.0f32 && raw_r <= 1.0f32;
            let inside_g = raw_g >= 0.0f32 && raw_g <= 1.0f32;
            let inside_b = raw_b >= 0.0f32 && raw_b <= 1.0f32;

            // Channel R.
            let dx = uvx - vignetting[vbase];
            let dy = uvy - vignetting[vbase + 1];
            let r2 = dx * dx + dy * dy;
            let gf = grad.x() * rgb_after_exp.x();
            let gr2 = gf
                * (vignetting[vbase + 2]
                    + 2.0f32 * vignetting[vbase + 3] * r2
                    + 3.0f32 * vignetting[vbase + 4] * r2 * r2);
            pg[1] += select(inside_r, -gr2 * 2.0f32 * dx, 0.0f32);
            pg[2] += select(inside_r, -gr2 * 2.0f32 * dy, 0.0f32);
            pg[3] += select(inside_r, gf * r2, 0.0f32);
            pg[4] += select(inside_r, gf * r2 * r2, 0.0f32);
            pg[5] += select(inside_r, gf * r2 * r2 * r2, 0.0f32);

            // Channel G.
            let dx = uvx - vignetting[vbase + 5];
            let dy = uvy - vignetting[vbase + 6];
            let r2 = dx * dx + dy * dy;
            let gf = grad.y() * rgb_after_exp.y();
            let gr2 = gf
                * (vignetting[vbase + 7]
                    + 2.0f32 * vignetting[vbase + 8] * r2
                    + 3.0f32 * vignetting[vbase + 9] * r2 * r2);
            pg[6] += select(inside_g, -gr2 * 2.0f32 * dx, 0.0f32);
            pg[7] += select(inside_g, -gr2 * 2.0f32 * dy, 0.0f32);
            pg[8] += select(inside_g, gf * r2, 0.0f32);
            pg[9] += select(inside_g, gf * r2 * r2, 0.0f32);
            pg[10] += select(inside_g, gf * r2 * r2 * r2, 0.0f32);

            // Channel B.
            let dx = uvx - vignetting[vbase + 10];
            let dy = uvy - vignetting[vbase + 11];
            let r2 = dx * dx + dy * dy;
            let gf = grad.z() * rgb_after_exp.z();
            let gr2 = gf
                * (vignetting[vbase + 12]
                    + 2.0f32 * vignetting[vbase + 13] * r2
                    + 3.0f32 * vignetting[vbase + 14] * r2 * r2);
            pg[11] += select(inside_b, -gr2 * 2.0f32 * dx, 0.0f32);
            pg[12] += select(inside_b, -gr2 * 2.0f32 * dy, 0.0f32);
            pg[13] += select(inside_b, gf * r2, 0.0f32);
            pg[14] += select(inside_b, gf * r2 * r2, 0.0f32);
            pg[15] += select(inside_b, gf * r2 * r2 * r2, 0.0f32);

            grad = Vec3A::new(grad.x() * f_r, grad.y() * f_g, grad.z() * f_b);
        }

        // 1. Exposure backward.
        if with_frame {
            let factor = f32::exp(exp_param * LN2);
            pg[0] += grad.dot(rgb_input.scale(factor)) * LN2;
            grad = grad.scale(factor);
        }

        grad_rgb[base] = grad.x();
        grad_rgb[base + 1] = grad.y();
        grad_rgb[base + 2] = grad.z();
        if has_alpha {
            grad_rgb[base + 3] = v_out[base + 3];
        }
    }

    // --- Cube-level reduction of the 36 param grads ---
    let mut sg_partials = Shared::new_slice((MAX_SUBGROUPS * NUM_PARAM_GRADS) as usize);
    let subgroup_id = UNIT_POS_X / PLANE_DIM;
    #[unroll]
    for p in 0u32..NUM_PARAM_GRADS {
        let v = plane_sum(pg[p as usize]);
        if UNIT_POS_PLANE == 0u32 {
            sg_partials[(subgroup_id * NUM_PARAM_GRADS + p) as usize] = v;
        }
    }
    sync_cube();
    if UNIT_POS_X < NUM_PARAM_GRADS {
        let num_subgroups = BLOCK_SIZE / PLANE_DIM;
        let mut tot = 0.0f32;
        let mut i = 0u32;
        while i < num_subgroups {
            tot += sg_partials[(i * NUM_PARAM_GRADS + UNIT_POS_X) as usize];
            i += 1u32;
        }
        partials[(CUBE_POS_X * NUM_PARAM_GRADS + UNIT_POS_X) as usize] = tot;
    }
}
