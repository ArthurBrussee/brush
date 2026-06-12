//! Hybrid "PPISP grid" kernels (spirulae-splat's `bilagrid_type="ppisp"`),
//! with the per-camera vignetting stage fused in.
//!
//! Per-pixel pipeline, all in one pass:
//!
//! ```text
//! rgb → per-camera vignetting → slice grid payload → exposure → color → (CRF)
//! ```
//!
//! The grid payload per cell:
//!
//! | channels | meaning | flag |
//! |---|---|---|
//! | 0 | log2 exposure | always |
//! | 1..9 | latent color-homography offsets (B/R/G/N) | `with_color` |
//! | last 12 | CRF raw-param *offsets from identity* | `with_crf` |
//!
//! The grid is sliced at the same coordinates as the affine bilateral grid,
//! with the grayscale guidance taken from the *vignetted* color (the grid's
//! input). All-zeros is the identity transform for every payload, and zero
//! vignetting params are an identity falloff.
//!
//! The backward chains gradients through every stage in reverse:
//! `dL/dgrid` scatters to the 8 corners (atomic, subgroup-vote merged,
//! optionally plane-subsampled), the 15 per-camera vignetting gradients
//! reduce per cube into a partials buffer (deterministic host-side sum, no
//! atomics — same scheme as the PPISP kernels), and `dL/drgb` is exact and
//! includes the guidance-coordinate term.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use crate::AtomicAddF32;
use crate::bilagrid_kernels::{C2G_B, C2G_G, C2G_R, plane_elected, slice_coords};
use crate::ppisp_math::{
    LN2, color_correct_bwd, color_correct_fwd, crf_channel_bwd, crf_channel_fwd, homography,
    vig_falloff_raw, vig_uv,
};
use brush_cube::{Vec3A, is_finite_f32};

pub const BLOCK_SIZE: u32 = 256;

/// Max payload channels (exposure + color + CRF); local arrays are sized to
/// this, unused entries are dead code after comptime unrolling.
const MAX_PAYLOAD: u32 = 21;

/// Per-camera vignetting gradients reduced per cube (3 channels x 5).
pub const NUM_VIG_GRADS: u32 = 15;
/// Worst-case subgroup count per cube (`PLANE_DIM >= 4` everywhere).
const MAX_SUBGROUPS: u32 = BLOCK_SIZE / 4;

/// CRF identity init in raw space (`softplus_inverse(1, 0.3)` for
/// toe/shoulder, `softplus_inverse(1, 0.1)` for gamma, `sigmoid^-1(0.5)`
/// for center). The grid stores *offsets* from these.
const CRF_IDENT_TOE: f32 = 0.013_658_988;
const CRF_IDENT_SHOULDER: f32 = 0.013_658_988;
const CRF_IDENT_GAMMA: f32 = 0.378_164_53;
const CRF_IDENT_CENTER: f32 = 0.0;

/// Payload channel count for a flag combination (host + comptime helper).
pub const fn payload_channels(with_color: bool, with_crf: bool) -> u32 {
    1 + if with_color { 8 } else { 0 } + if with_crf { 12 } else { 0 }
}

/// CRF channel base offset within the payload.
const fn crf_base(with_color: bool) -> u32 {
    if with_color { 9 } else { 1 }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
pub fn ppisp_grid_fwd_kernel(
    grid: &Tensor<f32>,
    vignetting: &Tensor<f32>,
    rgb: &Tensor<f32>,
    out: &mut Tensor<f32>,
    gl: u32,
    gh: u32,
    gw: u32,
    img_h: u32,
    img_w: u32,
    grid_offset: u32,
    channels: u32,
    camera_idx: u32,
    #[comptime] has_alpha: bool,
    #[comptime] with_vignetting: bool,
    #[comptime] with_color: bool,
    #[comptime] with_crf: bool,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    if idx >= img_h * img_w {
        terminate!();
    }
    let hi = idx / img_w;
    let wi = idx % img_w;
    let base = (idx * channels) as usize;

    let mut col = Vec3A::new(rgb[base], rgb[base + 1], rgb[base + 2]);

    // 1. Per-camera vignetting.
    if with_vignetting {
        let (uvx, uvy) = vig_uv(wi, hi, img_w, img_h);
        let vbase = (camera_idx * NUM_VIG_GRADS) as usize;
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
        col = Vec3A::new(col.x() * f_r, col.y() * f_g, col.z() * f_b);
    }

    // 2. Slice the payload at the vignetted color's guidance coordinate.
    let c = slice_coords(wi, hi, col.x(), col.y(), col.z(), gl, gh, gw, img_h, img_w);
    let cell = gl * gh * gw;
    let pc = comptime!(payload_channels(with_color, with_crf));

    let mut p = Array::<f32>::new(MAX_PAYLOAD as usize);
    #[unroll]
    for ci in 0u32..pc {
        let cbase = grid_offset + ci * cell;
        let v000 = grid[(cbase + (c.z0 * gh + c.y0) * gw + c.x0) as usize];
        let v001 = grid[(cbase + (c.z0 * gh + c.y0) * gw + c.x1) as usize];
        let v010 = grid[(cbase + (c.z0 * gh + c.y1) * gw + c.x0) as usize];
        let v011 = grid[(cbase + (c.z0 * gh + c.y1) * gw + c.x1) as usize];
        let v100 = grid[(cbase + (c.z1 * gh + c.y0) * gw + c.x0) as usize];
        let v101 = grid[(cbase + (c.z1 * gh + c.y0) * gw + c.x1) as usize];
        let v110 = grid[(cbase + (c.z1 * gh + c.y1) * gw + c.x0) as usize];
        let v111 = grid[(cbase + (c.z1 * gh + c.y1) * gw + c.x1) as usize];

        let c00 = v000 * (1.0f32 - c.fx) + v001 * c.fx;
        let c01 = v010 * (1.0f32 - c.fx) + v011 * c.fx;
        let c10 = v100 * (1.0f32 - c.fx) + v101 * c.fx;
        let c11 = v110 * (1.0f32 - c.fx) + v111 * c.fx;
        let c0 = c00 * (1.0f32 - c.fy) + c01 * c.fy;
        let c1 = c10 * (1.0f32 - c.fy) + c11 * c.fy;
        p[ci as usize] = c0 * (1.0f32 - c.fz) + c1 * c.fz;
    }

    // 3. Apply: exposure → color homography → CRF.
    col = col.scale(f32::exp(p[0] * LN2));
    if with_color {
        let (h, _) = homography(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);
        col = color_correct_fwd(col, h);
    }
    if with_crf {
        let kb = comptime!(crf_base(with_color) as usize);
        let xr = clamp(col.x(), 0.0f32, 1.0f32);
        let xg = clamp(col.y(), 0.0f32, 1.0f32);
        let xb = clamp(col.z(), 0.0f32, 1.0f32);
        let o_r = crf_channel_fwd(
            xr,
            CRF_IDENT_TOE + p[kb],
            CRF_IDENT_SHOULDER + p[comptime!(kb + 1)],
            CRF_IDENT_GAMMA + p[comptime!(kb + 2)],
            CRF_IDENT_CENTER + p[comptime!(kb + 3)],
        );
        let o_g = crf_channel_fwd(
            xg,
            CRF_IDENT_TOE + p[comptime!(kb + 4)],
            CRF_IDENT_SHOULDER + p[comptime!(kb + 5)],
            CRF_IDENT_GAMMA + p[comptime!(kb + 6)],
            CRF_IDENT_CENTER + p[comptime!(kb + 7)],
        );
        let o_b = crf_channel_fwd(
            xb,
            CRF_IDENT_TOE + p[comptime!(kb + 8)],
            CRF_IDENT_SHOULDER + p[comptime!(kb + 9)],
            CRF_IDENT_GAMMA + p[comptime!(kb + 10)],
            CRF_IDENT_CENTER + p[comptime!(kb + 11)],
        );
        col = Vec3A::new(o_r, o_g, o_b);
    }

    out[base] = select(is_finite_f32(col.x()), col.x(), 0.5f32);
    out[base + 1] = select(is_finite_f32(col.y()), col.y(), 0.5f32);
    out[base + 2] = select(is_finite_f32(col.z()), col.z(), 0.5f32);
    if has_alpha {
        out[base + 3] = rgb[base + 3];
    }
}

/// Backward: chain through CRF → color → exposure → guidance term →
/// vignetting; scatter `dL/dpayload` to the 8 grid corners and reduce the
/// 15 vignetting gradients per cube. No early `terminate!` — out-of-range
/// threads contribute zeros and still reach the reduction barrier.
///
/// `subsample`/`subsample_seed`: only planes elected by the per-step
/// Bernoulli hash (see `plane_elected`) scatter grid gradients (scaled by
/// `subsample` to stay unbiased); the rgb and vignetting gradients are
/// always exact. `subsample = 1` disables.
#[cube(launch)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_pass_by_ref_mut,
    clippy::fn_params_excessive_bools,
    clippy::manual_range_contains
)]
pub fn ppisp_grid_bwd_kernel<A: AtomicAddF32>(
    grid: &Tensor<f32>,
    vignetting: &Tensor<f32>,
    rgb: &Tensor<f32>,
    v_out: &Tensor<f32>,
    grad_grid: &mut Tensor<Atomic<A::Storage>>,
    grad_rgb: &mut Tensor<f32>,
    vig_partials: &mut Tensor<f32>,
    gl: u32,
    gh: u32,
    gw: u32,
    img_h: u32,
    img_w: u32,
    grid_offset: u32,
    channels: u32,
    camera_idx: u32,
    subsample: u32,
    subsample_seed: u32,
    #[comptime] has_alpha: bool,
    #[comptime] with_vignetting: bool,
    #[comptime] with_color: bool,
    #[comptime] with_crf: bool,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;

    let mut vg = Array::<f32>::new(NUM_VIG_GRADS as usize);
    #[unroll]
    for k in 0u32..NUM_VIG_GRADS {
        vg[k as usize] = 0.0f32;
    }

    if idx < img_h * img_w {
        let hi = idx / img_w;
        let wi = idx % img_w;
        let base = (idx * channels) as usize;

        let sr_raw = rgb[base];
        let sg_raw = rgb[base + 1];
        let sb_raw = rgb[base + 2];
        let sr = select(is_finite_f32(sr_raw), sr_raw, 0.5f32);
        let sg = select(is_finite_f32(sg_raw), sg_raw, 0.5f32);
        let sb = select(is_finite_f32(sb_raw), sb_raw, 0.5f32);
        let rgb_in = Vec3A::new(sr, sg, sb);

        // --- Recompute the vignetting stage ---
        let mut uvx = 0.0f32;
        let mut uvy = 0.0f32;
        let mut raw_r = 1.0f32;
        let mut raw_g = 1.0f32;
        let mut raw_b = 1.0f32;
        let mut f_r = 1.0f32;
        let mut f_g = 1.0f32;
        let mut f_b = 1.0f32;
        let vbase = (camera_idx * NUM_VIG_GRADS) as usize;
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
        // The grid stage's input.
        let gin = Vec3A::new(rgb_in.x() * f_r, rgb_in.y() * f_g, rgb_in.z() * f_b);

        let c = slice_coords(wi, hi, gin.x(), gin.y(), gin.z(), gl, gh, gw, img_h, img_w);

        let dr_raw = v_out[base];
        let dg_raw = v_out[base + 1];
        let db_raw = v_out[base + 2];
        let dr = select(is_finite_f32(dr_raw), dr_raw, 0.0f32);
        let dg = select(is_finite_f32(dg_raw), dg_raw, 0.0f32);
        let db = select(is_finite_f32(db_raw), db_raw, 0.0f32);

        let cell = gl * gh * gw;
        let pc = comptime!(payload_channels(with_color, with_crf));

        // Pass 1: interpolate the payload and its derivative along the
        // guidance axis.
        let mut p = Array::<f32>::new(MAX_PAYLOAD as usize);
        let mut dpdz = Array::<f32>::new(MAX_PAYLOAD as usize);
        #[unroll]
        for ci in 0u32..pc {
            p[ci as usize] = 0.0f32;
            dpdz[ci as usize] = 0.0f32;
        }
        #[unroll]
        for corner in 0u32..8u32 {
            let cx = select((corner & 1u32) != 0u32, c.x1, c.x0);
            let cy = select((corner & 2u32) != 0u32, c.y1, c.y0);
            let cz = select((corner & 4u32) != 0u32, c.z1, c.z0);
            let wx = select((corner & 1u32) != 0u32, c.fx, 1.0f32 - c.fx);
            let wy = select((corner & 2u32) != 0u32, c.fy, 1.0f32 - c.fy);
            let wz = select((corner & 4u32) != 0u32, c.fz, 1.0f32 - c.fz);
            let wt = wx * wy * wz;
            let dfdz = wx * wy * select((corner & 4u32) != 0u32, 1.0f32, -1.0f32);

            #[unroll]
            for ci in 0u32..pc {
                let v = grid[(grid_offset + ci * cell + (cz * gh + cy) * gw + cx) as usize];
                p[ci as usize] += wt * v;
                dpdz[ci as usize] += dfdz * f32::cast_from(gl - 1) * v;
            }
        }

        // Recompute forward stages from the grid input.
        let exp_factor = f32::exp(p[0] * LN2);
        let after_exp = gin.scale(exp_factor);
        let mut after_color = after_exp;
        if with_color {
            let (h, _) = homography(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]);
            after_color = color_correct_fwd(after_exp, h);
        }

        // Backward chain → dL/dpayload + dL/d(grid input).
        let mut grad = Vec3A::new(dr, dg, db);
        let mut dldp = Array::<f32>::new(MAX_PAYLOAD as usize);
        #[unroll]
        for ci in 0u32..pc {
            dldp[ci as usize] = 0.0f32;
        }

        if with_crf {
            let kb = comptime!(crf_base(with_color) as usize);
            let xr = clamp(after_color.x(), 0.0f32, 1.0f32);
            let xg = clamp(after_color.y(), 0.0f32, 1.0f32);
            let xb = clamp(after_color.z(), 0.0f32, 1.0f32);
            let (gx_r, gt_r, gs_r, gg_r, gc_r) = crf_channel_bwd(
                xr,
                CRF_IDENT_TOE + p[kb],
                CRF_IDENT_SHOULDER + p[comptime!(kb + 1)],
                CRF_IDENT_GAMMA + p[comptime!(kb + 2)],
                CRF_IDENT_CENTER + p[comptime!(kb + 3)],
                grad.x(),
            );
            let (gx_g, gt_g, gs_g, gg_g, gc_g) = crf_channel_bwd(
                xg,
                CRF_IDENT_TOE + p[comptime!(kb + 4)],
                CRF_IDENT_SHOULDER + p[comptime!(kb + 5)],
                CRF_IDENT_GAMMA + p[comptime!(kb + 6)],
                CRF_IDENT_CENTER + p[comptime!(kb + 7)],
                grad.y(),
            );
            let (gx_b, gt_b, gs_b, gg_b, gc_b) = crf_channel_bwd(
                xb,
                CRF_IDENT_TOE + p[comptime!(kb + 8)],
                CRF_IDENT_SHOULDER + p[comptime!(kb + 9)],
                CRF_IDENT_GAMMA + p[comptime!(kb + 10)],
                CRF_IDENT_CENTER + p[comptime!(kb + 11)],
                grad.z(),
            );
            dldp[kb] = gt_r;
            dldp[comptime!(kb + 1)] = gs_r;
            dldp[comptime!(kb + 2)] = gg_r;
            dldp[comptime!(kb + 3)] = gc_r;
            dldp[comptime!(kb + 4)] = gt_g;
            dldp[comptime!(kb + 5)] = gs_g;
            dldp[comptime!(kb + 6)] = gg_g;
            dldp[comptime!(kb + 7)] = gc_g;
            dldp[comptime!(kb + 8)] = gt_b;
            dldp[comptime!(kb + 9)] = gs_b;
            dldp[comptime!(kb + 10)] = gg_b;
            dldp[comptime!(kb + 11)] = gc_b;
            grad = Vec3A::new(gx_r, gx_g, gx_b);
        }

        if with_color {
            grad = color_correct_bwd(
                after_exp, p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], grad, &mut dldp, 1u32,
            );
        }

        // Exposure: d out/d e = stage_output · ln 2.
        dldp[0] = grad.dot(after_exp) * LN2;
        grad = grad.scale(exp_factor);

        // Guidance-coordinate chain: payload varies with z = gray(grid in).
        let mut gz = 0.0f32;
        #[unroll]
        for ci in 0u32..pc {
            gz += dldp[ci as usize] * dpdz[ci as usize];
        }
        let guard = f32::cast_from(c.z0) != c.z && f32::cast_from(c.z1) != c.z;
        gz = select(guard, gz, 0.0f32);

        // Total gradient at the grid input (stage chain + guidance term),
        // which is the vignetting stage's output gradient.
        grad = Vec3A::new(
            grad.x() + C2G_R * gz,
            grad.y() + C2G_G * gz,
            grad.z() + C2G_B * gz,
        );

        // Vignetting backward (input is the raw render).
        if with_vignetting {
            let inside_r = raw_r >= 0.0f32 && raw_r <= 1.0f32;
            let inside_g = raw_g >= 0.0f32 && raw_g <= 1.0f32;
            let inside_b = raw_b >= 0.0f32 && raw_b <= 1.0f32;

            // Channel R.
            let dx = uvx - vignetting[vbase];
            let dy = uvy - vignetting[vbase + 1];
            let r2 = dx * dx + dy * dy;
            let gf = grad.x() * rgb_in.x();
            let gr2 = gf
                * (vignetting[vbase + 2]
                    + 2.0f32 * vignetting[vbase + 3] * r2
                    + 3.0f32 * vignetting[vbase + 4] * r2 * r2);
            vg[0] += select(inside_r, -gr2 * 2.0f32 * dx, 0.0f32);
            vg[1] += select(inside_r, -gr2 * 2.0f32 * dy, 0.0f32);
            vg[2] += select(inside_r, gf * r2, 0.0f32);
            vg[3] += select(inside_r, gf * r2 * r2, 0.0f32);
            vg[4] += select(inside_r, gf * r2 * r2 * r2, 0.0f32);

            // Channel G.
            let dx = uvx - vignetting[vbase + 5];
            let dy = uvy - vignetting[vbase + 6];
            let r2 = dx * dx + dy * dy;
            let gf = grad.y() * rgb_in.y();
            let gr2 = gf
                * (vignetting[vbase + 7]
                    + 2.0f32 * vignetting[vbase + 8] * r2
                    + 3.0f32 * vignetting[vbase + 9] * r2 * r2);
            vg[5] += select(inside_g, -gr2 * 2.0f32 * dx, 0.0f32);
            vg[6] += select(inside_g, -gr2 * 2.0f32 * dy, 0.0f32);
            vg[7] += select(inside_g, gf * r2, 0.0f32);
            vg[8] += select(inside_g, gf * r2 * r2, 0.0f32);
            vg[9] += select(inside_g, gf * r2 * r2 * r2, 0.0f32);

            // Channel B.
            let dx = uvx - vignetting[vbase + 10];
            let dy = uvy - vignetting[vbase + 11];
            let r2 = dx * dx + dy * dy;
            let gf = grad.z() * rgb_in.z();
            let gr2 = gf
                * (vignetting[vbase + 12]
                    + 2.0f32 * vignetting[vbase + 13] * r2
                    + 3.0f32 * vignetting[vbase + 14] * r2 * r2);
            vg[10] += select(inside_b, -gr2 * 2.0f32 * dx, 0.0f32);
            vg[11] += select(inside_b, -gr2 * 2.0f32 * dy, 0.0f32);
            vg[12] += select(inside_b, gf * r2, 0.0f32);
            vg[13] += select(inside_b, gf * r2 * r2, 0.0f32);
            vg[14] += select(inside_b, gf * r2 * r2 * r2, 0.0f32);

            grad = Vec3A::new(grad.x() * f_r, grad.y() * f_g, grad.z() * f_b);
        }

        grad_rgb[base] = grad.x();
        grad_rgb[base + 1] = grad.y();
        grad_rgb[base + 2] = grad.z();
        if has_alpha {
            grad_rgb[base + 3] = v_out[base + 3];
        }

        // Pass 2: scatter dL/dpayload to the 8 corners. Plane-uniform
        // gradient subsampling: only planes elected by the per-round hash
        // contribute (scaled to stay unbiased) — the grid is heavily
        // regularised and slow-moving, so a sparse estimate is enough (and
        // atomics dominate this kernel).
        let plane_idx = idx / PLANE_DIM;
        if plane_elected(plane_idx, subsample_seed, subsample) {
            let scale = f32::cast_from(subsample);
            let cell_uniform = plane_min(c.x0) == plane_max(c.x0)
                && plane_min(c.y0) == plane_max(c.y0)
                && plane_min(c.z0) == plane_max(c.z0)
                && plane_min(c.x1) == plane_max(c.x1)
                && plane_min(c.y1) == plane_max(c.y1)
                && plane_min(c.z1) == plane_max(c.z1);

            #[unroll]
            for corner in 0u32..8u32 {
                let cx = select((corner & 1u32) != 0u32, c.x1, c.x0);
                let cy = select((corner & 2u32) != 0u32, c.y1, c.y0);
                let cz = select((corner & 4u32) != 0u32, c.z1, c.z0);
                let wx = select((corner & 1u32) != 0u32, c.fx, 1.0f32 - c.fx);
                let wy = select((corner & 2u32) != 0u32, c.fy, 1.0f32 - c.fy);
                let wz = select((corner & 4u32) != 0u32, c.fz, 1.0f32 - c.fz);
                let wt = wx * wy * wz;

                #[unroll]
                for ci in 0u32..pc {
                    let gidx = (grid_offset + ci * cell + (cz * gh + cy) * gw + cx) as usize;
                    let contrib = wt * dldp[ci as usize] * scale;
                    if cell_uniform {
                        let merged = plane_sum(contrib);
                        if UNIT_POS_PLANE == 0u32 {
                            A::add(&grad_grid[gidx], merged);
                        }
                    } else {
                        A::add(&grad_grid[gidx], contrib);
                    }
                }
            }
        }
    }

    // --- Cube-level reduction of the 15 vignetting grads ---
    if with_vignetting {
        let mut sg_partials = Shared::new_slice((MAX_SUBGROUPS * NUM_VIG_GRADS) as usize);
        let subgroup_id = UNIT_POS_X / PLANE_DIM;
        #[unroll]
        for k in 0u32..NUM_VIG_GRADS {
            let v = plane_sum(vg[k as usize]);
            if UNIT_POS_PLANE == 0u32 {
                sg_partials[(subgroup_id * NUM_VIG_GRADS + k) as usize] = v;
            }
        }
        sync_cube();
        if UNIT_POS_X < NUM_VIG_GRADS {
            let num_subgroups = BLOCK_SIZE / PLANE_DIM;
            let mut tot = 0.0f32;
            let mut i = 0u32;
            while i < num_subgroups {
                tot += sg_partials[(i * NUM_VIG_GRADS + UNIT_POS_X) as usize];
                i += 1u32;
            }
            vig_partials[(CUBE_POS_X * NUM_VIG_GRADS + UNIT_POS_X) as usize] = tot;
        }
    }
}
