//! Fused bilateral-grid kernels (port of `LichtFeld` Studio's CUDA kernels,
//! which match gsplat's `lib_bilagrid` / `F.grid_sample(align_corners=True,
//! padding_mode="border")` semantics).
//!
//! A bilateral grid is a `[12, L, H, W]` field of 3x4 affine color
//! transforms. Slicing trilinearly interpolates the 12 coefficients at
//! `(x, y) = pixel position` (scaled to the grid) and `z = BT601 grayscale
//! of the input RGB`, then applies `rgb_out = A * rgb_in + b`.
//!
//! The grid tensor argument holds *all* views' grids `[N, 12, L, H, W]`;
//! `grid_offset` selects the active view so the backward can scatter into a
//! full-size gradient tensor directly.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use crate::AtomicAddF32;
use brush_cube::is_finite_f32;

/// BT601 RGB→gray weights (used by every reference implementation).
pub const C2G_R: f32 = 0.299;
pub const C2G_G: f32 = 0.587;
pub const C2G_B: f32 = 0.114;

pub const BLOCK_SIZE: u32 = 256;

/// Per-pixel slicing coordinates and interpolation weights.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub(crate) struct SliceCoords {
    pub x0: u32,
    pub x1: u32,
    pub y0: u32,
    pub y1: u32,
    pub z0: u32,
    pub z1: u32,
    pub fx: f32,
    pub fy: f32,
    pub fz: f32,
    /// Continuous guidance coordinate (needed by the backward's exact-plane
    /// gradient guard).
    pub z: f32,
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn slice_coords(
    wi: u32,
    hi: u32,
    sr: f32,
    sg: f32,
    sb: f32,
    gl: u32,
    gh: u32,
    gw: u32,
    img_h: u32,
    img_w: u32,
) -> SliceCoords {
    // align_corners=True: pixel (0..w-1) maps to grid (0..W-1).
    let x = f32::cast_from(wi) / f32::cast_from(max(img_w - 1, 1u32)) * f32::cast_from(gw - 1);
    let y = f32::cast_from(hi) / f32::cast_from(max(img_h - 1, 1u32)) * f32::cast_from(gh - 1);
    let z = (C2G_R * sr + C2G_G * sg + C2G_B * sb) * f32::cast_from(gl - 1);

    let x0 = u32::cast_from(f32::floor(x));
    let y0 = u32::cast_from(f32::floor(y));
    let x1 = min(x0 + 1, gw - 1);
    let y1 = min(y0 + 1, gh - 1);
    // The guidance coordinate can run past the grid (rgb outside [0, 1]) —
    // clamp like padding_mode="border". fx/fy/fz are measured against the
    // *clamped* z0 to match the reference kernels.
    let z0i = i32::cast_from(f32::floor(z));
    let gl_max = i32::cast_from(gl - 1);
    let z0c = clamp(z0i, 0i32, gl_max);
    let z1c = clamp(z0i + 1i32, 0i32, gl_max);

    SliceCoords {
        x0,
        x1,
        y0,
        y1,
        z0: u32::cast_from(z0c),
        z1: u32::cast_from(z1c),
        fx: x - f32::floor(x),
        fy: y - f32::floor(y),
        fz: z - f32::cast_from(z0c),
        z,
    }
}

/// `rgb_out[c] += interp(grid channel) * coeff` over the 12 affine
/// coefficients; one thread per pixel.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn bilagrid_slice_fwd_kernel(
    grid: &Tensor<f32>,
    rgb: &Tensor<f32>,
    out: &mut Tensor<f32>,
    gl: u32,
    gh: u32,
    gw: u32,
    img_h: u32,
    img_w: u32,
    grid_offset: u32,
    channels: u32,
    #[comptime] has_alpha: bool,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    if idx >= img_h * img_w {
        terminate!();
    }
    let hi = idx / img_w;
    let wi = idx % img_w;
    let base = (idx * channels) as usize;

    let sr = rgb[base];
    let sg = rgb[base + 1];
    let sb = rgb[base + 2];

    let c = slice_coords(wi, hi, sr, sg, sb, gl, gh, gw, img_h, img_w);

    let cell = gl * gh * gw;
    let mut dr = 0.0f32;
    let mut dg = 0.0f32;
    let mut db = 0.0f32;

    #[unroll]
    for ci in 0u32..12u32 {
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
        let val = c0 * (1.0f32 - c.fz) + c1 * c.fz;

        let si = ci % 4u32;
        let di = ci / 4u32;
        let coeff = select(
            si == 0u32,
            sr,
            select(si == 1u32, sg, select(si == 2u32, sb, 1.0f32)),
        );
        let contrib = val * coeff;
        dr += select(di == 0u32, contrib, 0.0f32);
        dg += select(di == 1u32, contrib, 0.0f32);
        db += select(di == 2u32, contrib, 0.0f32);
    }

    out[base] = select(is_finite_f32(dr), dr, 0.5f32);
    out[base + 1] = select(is_finite_f32(dg), dg, 0.5f32);
    out[base + 2] = select(is_finite_f32(db), db, 0.5f32);
    if has_alpha {
        out[base + 3] = rgb[base + 3];
    }
}

/// Independent per-plane Bernoulli election with probability `1/every`:
/// a PCG-style hash of the plane index mixed with the per-step seed (see
/// `GradSubsample`). Stochastic per-warp selection instead of a regular
/// stripe pattern.
#[cube]
// `is_multiple_of` is not available inside `#[cube]` kernels.
#[allow(clippy::manual_is_multiple_of)]
pub(crate) fn plane_elected(plane_idx: u32, seed: u32, every: u32) -> bool {
    // Golden-ratio spread of the seed before combining: both inputs may be
    // small integers, and xor alone mixes those poorly.
    let mut x = plane_idx + seed * 2654435769u32;
    x = x * 747796405u32 + 2891336453u32;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u32;
    ((x >> 22) ^ x) % every == 0
}

/// Backward: per-pixel scatter of `dL/dgrid` (atomic) and write of
/// `dL/drgb` (the affine read-out plus the guidance-coordinate term).
#[cube(launch)]
#[allow(clippy::too_many_arguments, clippy::needless_pass_by_ref_mut)]
pub fn bilagrid_slice_bwd_kernel<A: AtomicAddF32>(
    grid: &Tensor<f32>,
    rgb: &Tensor<f32>,
    v_out: &Tensor<f32>,
    grad_grid: &mut Tensor<Atomic<A::Storage>>,
    grad_rgb: &mut Tensor<f32>,
    gl: u32,
    gh: u32,
    gw: u32,
    img_h: u32,
    img_w: u32,
    grid_offset: u32,
    channels: u32,
    subsample: u32,
    subsample_seed: u32,
    #[comptime] has_alpha: bool,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    if idx >= img_h * img_w {
        terminate!();
    }
    let hi = idx / img_w;
    let wi = idx % img_w;
    let base = (idx * channels) as usize;
    // Plane-uniform gradient subsampling (see `GradSubsample`): non-elected
    // planes still compute the exact rgb gradient, but skip the atomic
    // scatter into the grid.
    let do_scatter = plane_elected(idx / PLANE_DIM, subsample_seed, subsample);
    let sscale = f32::cast_from(subsample);

    let sr_raw = rgb[base];
    let sg_raw = rgb[base + 1];
    let sb_raw = rgb[base + 2];
    let sr = select(is_finite_f32(sr_raw), sr_raw, 0.5f32);
    let sg = select(is_finite_f32(sg_raw), sg_raw, 0.5f32);
    let sb = select(is_finite_f32(sb_raw), sb_raw, 0.5f32);

    let c = slice_coords(wi, hi, sr, sg, sb, gl, gh, gw, img_h, img_w);

    let dr_raw = v_out[base];
    let dg_raw = v_out[base + 1];
    let db_raw = v_out[base + 2];
    let dr = select(is_finite_f32(dr_raw), dr_raw, 0.0f32);
    let dg = select(is_finite_f32(dg_raw), dg_raw, 0.0f32);
    let db = select(is_finite_f32(db_raw), db_raw, 0.0f32);

    let cell = gl * gh * gw;
    let mut vr = 0.0f32;
    let mut vg = 0.0f32;
    let mut vb = 0.0f32;
    let mut gz = 0.0f32;

    // Subgroup vote: rendered images are smooth, so a plane's 32 pixels
    // usually land in the same grid cell. When every lane agrees on all
    // eight corner cells, merge the scatter with `plane_sum` and emit one
    // atomic per (corner, coeff) from the first lane — a ~32x cut in
    // global atomics, which dominate this kernel. WGSL subgroup ops act on
    // active lanes and don't require uniform control flow, and the branch
    // below is plane-uniform by construction.
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
        // d(weight)/d(fz): the z-factor flips sign on the near corners.
        let dfdz = wx * wy * select((corner & 4u32) != 0u32, 1.0f32, -1.0f32);

        let mut trilerp = 0.0f32;
        #[unroll]
        for ci in 0u32..12u32 {
            let gidx = (grid_offset + ci * cell + (cz * gh + cy) * gw + cx) as usize;
            let si = ci % 4u32;
            let di = ci / 4u32;
            let r_coeff = select(
                si == 0u32,
                sr,
                select(si == 1u32, sg, select(si == 2u32, sb, 1.0f32)),
            );
            let gout = select(di == 0u32, dr, select(di == 1u32, dg, db));
            let v = grid[gidx];

            // dL/drgb through the affine application (A * rgb): only the
            // first three columns multiply the input color.
            let contrib = v * wt * gout;
            vr += select(si == 0u32, contrib, 0.0f32);
            vg += select(si == 1u32, contrib, 0.0f32);
            vb += select(si == 2u32, contrib, 0.0f32);

            let grad_weight = r_coeff * gout;
            trilerp += v * grad_weight;
            if cell_uniform {
                let merged = plane_sum(wt * grad_weight);
                if do_scatter && UNIT_POS_PLANE == 0u32 {
                    A::add(&grad_grid[gidx], merged * sscale);
                }
            } else if do_scatter {
                A::add(&grad_grid[gidx], wt * grad_weight * sscale);
            }
        }
        gz += dfdz * f32::cast_from(gl - 1) * trilerp;
    }

    // Zero the guidance gradient when z sits exactly on a grid plane
    // (matches the reference kernels' subgradient choice there).
    let guard = f32::cast_from(c.z0) != c.z && f32::cast_from(c.z1) != c.z;
    gz = select(guard, gz, 0.0f32);

    grad_rgb[base] = vr + C2G_R * gz;
    grad_rgb[base + 1] = vg + C2G_G * gz;
    grad_rgb[base + 2] = vb + C2G_B * gz;
    if has_alpha {
        // Alpha passes through the correction untouched.
        grad_rgb[base + 3] = v_out[base + 3];
    }
}

/// Total-variation forward, stage 1: grid-stride partial sums, one value
/// per cube. The host sums the partials (`Tensor::sum`) for the scalar
/// loss. Normalisation matches gsplat's `total_variation_loss`:
/// each direction's squared diffs are averaged over that direction's
/// element count (channels included) and the result is averaged over views.
#[cube(launch)]
pub fn bilagrid_tv_fwd_kernel(
    grids: &Tensor<f32>,
    partials: &mut Tensor<f32>,
    n: u32,
    gl: u32,
    gh: u32,
    gw: u32,
    #[comptime] grid_channels: u32,
) {
    let total = n * gl * gh * gw;
    let stride = CUBE_COUNT_X * BLOCK_SIZE;
    let mut idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;

    let sx = 1.0f32 / f32::cast_from(gl * gh * (gw - 1));
    let sy = 1.0f32 / f32::cast_from(gl * (gh - 1) * gw);
    let sz = 1.0f32 / f32::cast_from((gl - 1) * gh * gw);

    let mut local = 0.0f32;
    while idx < total {
        let wi = idx % gw;
        let hi = (idx / gw) % gh;
        let li = (idx / (gw * gh)) % gl;
        let ni = idx / (gw * gh * gl);

        #[unroll]
        for ci in 0u32..grid_channels {
            let cell_idx = (((ni * grid_channels + ci) * gl + li) * gh + hi) * gw + wi;
            let val = grids[cell_idx as usize];

            if wi > 0u32 {
                let v0 = grids[(cell_idx - 1u32) as usize];
                let d = val - v0;
                local += d * d * sx;
            }
            if hi > 0u32 {
                let v0 = grids[(cell_idx - gw) as usize];
                let d = val - v0;
                local += d * d * sy;
            }
            if li > 0u32 {
                let v0 = grids[(cell_idx - gw * gh) as usize];
                let d = val - v0;
                local += d * d * sz;
            }
        }
        idx += stride;
    }
    local /= f32::cast_from(grid_channels * n);

    // Cube-level reduction: plane sums, then thread 0 folds the planes.
    let sg_sum = plane_sum(local);
    let mut sg_partials = Shared::new_slice(BLOCK_SIZE as usize);
    let subgroup_id = UNIT_POS_X / PLANE_DIM;
    if UNIT_POS_PLANE == 0u32 {
        sg_partials[subgroup_id as usize] = sg_sum;
    }
    sync_cube();
    if UNIT_POS_X == 0u32 {
        let num_subgroups = BLOCK_SIZE / PLANE_DIM;
        let mut tot = 0.0f32;
        let mut i = 0u32;
        while i < num_subgroups {
            tot += sg_partials[i as usize];
            i += 1u32;
        }
        partials[CUBE_POS_X as usize] = tot;
    }
}

/// Total-variation backward: deterministic direct write, one thread per
/// `(n, l, h, w)` cell covering all 12 channels. `v_loss` is the upstream
/// scalar gradient as a 1-element tensor (kept on-GPU).
#[cube(launch)]
pub fn bilagrid_tv_bwd_kernel(
    grids: &Tensor<f32>,
    v_loss: &Tensor<f32>,
    grad_grids: &mut Tensor<f32>,
    n: u32,
    gl: u32,
    gh: u32,
    gw: u32,
    #[comptime] grid_channels: u32,
) {
    let idx = CUBE_POS_X * BLOCK_SIZE + UNIT_POS_X;
    let total = n * gl * gh * gw;
    if idx >= total {
        terminate!();
    }
    let wi = idx % gw;
    let hi = (idx / gw) % gh;
    let li = (idx / (gw * gh)) % gl;
    let ni = idx / (gw * gh * gl);

    // d/dv of the normalised TV sum: each (v - v0) pair contributes
    // 2 (v - v0) * s / (12 n) = (v - v0) * s / (6 n) to both sides.
    let s = v_loss[0] / f32::cast_from(6u32 * n);
    let sx = s / f32::cast_from(gl * gh * (gw - 1));
    let sy = s / f32::cast_from(gl * (gh - 1) * gw);
    let sz = s / f32::cast_from((gl - 1) * gh * gw);

    #[unroll]
    for ci in 0u32..grid_channels {
        let cell_idx = (((ni * grid_channels + ci) * gl + li) * gh + hi) * gw + wi;
        let val = grids[cell_idx as usize];
        let mut g = 0.0f32;

        if wi > 0u32 {
            g += (val - grids[(cell_idx - 1u32) as usize]) * sx;
        }
        if wi < gw - 1 {
            g += (val - grids[(cell_idx + 1u32) as usize]) * sx;
        }
        if hi > 0u32 {
            g += (val - grids[(cell_idx - gw) as usize]) * sy;
        }
        if hi < gh - 1 {
            g += (val - grids[(cell_idx + gw) as usize]) * sy;
        }
        if li > 0u32 {
            g += (val - grids[(cell_idx - gw * gh) as usize]) * sz;
        }
        if li < gl - 1 {
            g += (val - grids[(cell_idx + gw * gh) as usize]) * sz;
        }

        grad_grids[cell_idx as usize] = g;
    }
}
