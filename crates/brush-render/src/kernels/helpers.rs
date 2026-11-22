//! Render-specific cube helpers. Generic math (`Vec3A`, `Quat`, `Mat3`,
//! `Sym2`, `sigmoid`, `is_finite_*`, `calc_sigma`, `inverse_sym2`,
//! `det2_strict`) lives in [`brush_cube`] — re-exported here for
//! convenience.

use burn::cubecl;
use burn::cubecl::cube;
use burn::cubecl::prelude::*;
use burn::cubecl::std::FastDivmod;

use super::types::{ProjectUniforms, Quat, Splat, Sym2, TileBbox, Vec3A};
use crate::kernels::camera_model::{CameraModel, calculate_project_jacobian};
pub use brush_cube::{calc_sigma, is_finite_f32, sigmoid};

pub const TILE_WIDTH: u32 = 16;
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

/// Smoothstep ramp centered at 1/255: zero below `MID - BAND/2`, one
/// above `MID + BAND/2`. C^1 in alpha. Selected at kernel-compile-time
/// via `RasterPass::BackwardSmoothCutoff` — the rasterizer's
/// `smooth_cutoff` comptime gate routes around this for the production
/// hard-step path.
pub const ALPHA_CUTOFF_MID: f32 = 1.0 / 255.0;
pub const ALPHA_CUTOFF_BAND: f32 = 1.0e-3;

#[cube]
pub fn alpha_cutoff_weight(alpha: f32) -> f32 {
    let t = f32::clamp(
        (alpha - (ALPHA_CUTOFF_MID - 0.5f32 * ALPHA_CUTOFF_BAND)) / ALPHA_CUTOFF_BAND,
        0.0f32,
        1.0f32,
    );
    t * t * (3.0f32 - 2.0f32 * t)
}

#[cube]
pub fn alpha_cutoff_weight_deriv(alpha: f32) -> f32 {
    let low = ALPHA_CUTOFF_MID - 0.5f32 * ALPHA_CUTOFF_BAND;
    let high = ALPHA_CUTOFF_MID + 0.5f32 * ALPHA_CUTOFF_BAND;
    let inside = alpha > low && alpha < high;
    let t = (alpha - low) / ALPHA_CUTOFF_BAND;
    select(
        inside,
        (6.0f32 * t - 6.0f32 * t * t) / ALPHA_CUTOFF_BAND,
        0.0f32,
    )
}

/// `f32` lanes per projected splat. Layout matches `Splat`:
///   0:xy_x, 1:xy_y, 2:conic_x, 3:conic_y, 4:conic_z, 5:color_a,
///   6:color_r, 7:color_g, 8:color_b.
pub const PROJECTED_LANES: u32 = 9;
pub const PROJECTED_LANES_USIZE: usize = PROJECTED_LANES as usize;

#[cube]
pub fn compact_bits_16(v: u32) -> u32 {
    let mut x = v & 0x55555555u32;
    x = (x | (x >> 1u32)) & 0x33333333u32;
    x = (x | (x >> 2u32)) & 0x0F0F0F0Fu32;
    x = (x | (x >> 4u32)) & 0x00FF00FFu32;
    x = (x | (x >> 8u32)) & 0x0000FFFFu32;
    x
}

/// Decode a tile-internal Morton id to (px, py) coordinates within the image.
///
/// `tiles_per_row` arrives as a [`FastDivmod`] because the tile row/column
/// split is a runtime divide on the per-pixel path; the magic numbers are
/// precomputed host-side. `TILE_SIZE` is a constant, so those two stay plain.
#[cube]
pub fn map_1d_to_2d(id: u32, tiles_per_row: FastDivmod<u32>) -> (u32, u32) {
    let tile_id = id / TILE_SIZE;
    let within = id % TILE_SIZE;
    let (tile_y, tile_x) = tiles_per_row.div_mod(tile_id);
    let mx = compact_bits_16(within);
    let my = compact_bits_16(within >> 1u32);
    (tile_x * TILE_WIDTH + mx, tile_y * TILE_WIDTH + my)
}

/// Splat half-extent along x / y from the packed conic. Returns
/// `(-1, -1)` when the conic is degenerate so the caller bails on
/// `extent.x < 0`.
#[cube]
pub fn compute_bbox_extent(conic: Sym2, power_threshold: f32) -> (f32, f32) {
    let det = conic.c00 * conic.c11 - conic.c01 * conic.c01;
    let degenerate = det <= 0.0f32;
    let inv_det = select(degenerate, 0.0f32, 1.0f32 / det);
    let ex = f32::sqrt(2.0f32 * power_threshold * conic.c11 * inv_det);
    let ey = f32::sqrt(2.0f32 * power_threshold * conic.c00 * inv_det);
    (
        select(degenerate, -1.0f32, ex),
        select(degenerate, -1.0f32, ey),
    )
}

/// Pixel-space center +/- dims clamped to a `(bw, bh)` viewport. Used by
/// `get_tile_bbox` to compute the tile-grid bbox a splat covers.
#[cube]
pub fn get_bbox(cx: f32, cy: f32, dx: f32, dy: f32, bw: u32, bh: u32) -> TileBbox {
    let bwf = bw as f32;
    let bhf = bh as f32;
    TileBbox {
        min_x: clamp(cx - dx, 0.0f32, bwf) as u32,
        min_y: clamp(cy - dy, 0.0f32, bhf) as u32,
        max_x: clamp(cx + dx + 1.0f32, 0.0f32, bwf) as u32,
        max_y: clamp(cy + dy + 1.0f32, 0.0f32, bhf) as u32,
    }
}

#[cube]
pub fn get_tile_bbox(
    pix_cx: f32,
    pix_cy: f32,
    pix_ex: f32,
    pix_ey: f32,
    tile_bw: u32,
    tile_bh: u32,
) -> TileBbox {
    let tw = TILE_WIDTH as f32;
    get_bbox(
        pix_cx / tw,
        pix_cy / tw,
        pix_ex / tw,
        pix_ey / tw,
        tile_bw,
        tile_bh,
    )
}

/// 2D covariance from scale, quat, mean_c and view params. Returns the
/// symmetric covariance as `Sym2`, with a post-scale clamp so huge-but-
/// finite inputs don't overflow the `det` of the eventual conic.
#[cube]
pub fn calc_cov2d(
    scale: Vec3A,
    quat: Quat,
    mean_c: Vec3A,
    u: ProjectUniforms,
    #[comptime] camera_model: CameraModel,
) -> Sym2 {
    let ns = u.view_rotation().mul_mat3(quat.to_mat3()).mul_diag(scale);
    let cam_jac = calculate_project_jacobian(
        mean_c,
        u.jacobian_clamp_limits,
        u.pinhole_params,
        camera_model,
    );

    // V = J * N_s (J is 2x3, N_s is 3x3, V is 2x3).
    let v = cam_jac.mul_mat3(ns);

    // raw = V * V^T (2x2 symmetric).
    let raw = v.gram_matrix();

    // Clamp so max |entry| <= 1e18 — keeps det inside f32 range and
    // preserves PSD / off-diagonal-to-diagonal ratio for huge log_scale
    // training states.
    let lim = 1.0e18f32;
    let max_abs = raw.max_abs();
    let scale_down = select(max_abs > lim, lim / max_abs, 1.0f32);

    raw.scale(scale_down)
}

/// MIP-aware blur compensation. Adds `cov_blur` to the diagonal of the
/// passed-in cov2d and returns the new `Sym2` plus the compensation
/// factor (1.0 when `mip_splatting=false`).
#[cube]
pub fn compensate_cov2d(c: Sym2, #[comptime] mip_splatting: bool) -> (Sym2, f32) {
    let cov_blur = comptime![if mip_splatting { 0.1f32 } else { 0.3f32 }];
    let blurred = Sym2 {
        c00: c.c00 + cov_blur,
        c01: c.c01,
        c11: c.c11 + cov_blur,
    };
    let mut filter_comp = f32::cast_from(1.0f32);
    if comptime![mip_splatting] {
        let det_raw = max(c.det2_strict(), 0.0f32);
        let det_blurred = blurred.det2_strict();
        filter_comp = f32::sqrt(det_raw / det_blurred);
    }
    (blurred, filter_comp)
}

/// Tile columns `[lo, hi)` that a splat touches within tile row `ty`.
///
/// The splat's footprint is the ellipse where it still clears the alpha
/// cutoff, with half-extents `(ex, ey)` from `compute_bbox_extent`. Its
/// left edge is convex in y and leftmost at `y = b·ex / c` from the mean, and
/// the right edge mirrors that. So within the row's strip each edge is at its
/// extreme at that y clamped to the strip, which gives the exact x range
/// without testing every tile.
#[cube]
pub fn tile_row_span(
    bb: TileBbox,
    ty: u32,
    mx: f32,
    my: f32,
    conic: Sym2,
    ex: f32,
    ey: f32,
) -> (u32, u32) {
    let tw = TILE_WIDTH as f32;
    // The strip's y range relative to the mean, cut to the ellipse's.
    let y0 = max((ty * TILE_WIDTH) as f32 - my, -ey);
    let y1 = min(((ty + 1u32) * TILE_WIDTH) as f32 - my, ey);
    let y_peak = conic.c01 * ex / conic.c11;
    let yl = clamp(y_peak, y0, y1);
    let yr = clamp(-y_peak, y0, y1);

    // On the ellipse, x = (-b·y ± sqrt(det · (ey² - y²))) / a.
    let det = conic.c00 * conic.c11 - conic.c01 * conic.c01;
    let half_l = f32::sqrt(max(det * (ey * ey - yl * yl), 0.0f32));
    let half_r = f32::sqrt(max(det * (ey * ey - yr * yr), 0.0f32));
    let x_min = mx + (-conic.c01 * yl - half_l) / conic.c00;
    let x_max = mx + (-conic.c01 * yr + half_r) / conic.c00;

    let min_x = bb.min_x as f32;
    let max_x = bb.max_x as f32;
    let lo = clamp(f32::floor(x_min / tw), min_x, max_x) as u32;
    let hi = clamp(f32::floor(x_max / tw) + 1.0f32, min_x, max_x) as u32;
    // Rounding can leave the bbox's first or last row just off the ellipse.
    (lo, select(y0 <= y1, max(hi, lo), lo))
}

/// Number of tiles a splat touches. `project_forward` reserves this many
/// slots and `map_gaussians_to_intersect` fills them from the same
/// `tile_row_span` walk.
#[cube]
pub fn count_contributing_tiles(
    bb: TileBbox,
    mx: f32,
    my: f32,
    conic: Sym2,
    ex: f32,
    ey: f32,
) -> u32 {
    let mut num_tiles_hit = 0u32;
    for ty in bb.min_y..bb.max_y {
        let (lo, hi) = tile_row_span(bb, ty, mx, my, conic, ex, ey);
        num_tiles_hit += hi - lo;
    }
    num_tiles_hit
}

/// Read just the spatial fields (xy + conic + alpha) of a projected
/// splat. Used by `map_gaussians_to_intersect`, which doesn't need the
/// colors.
#[cube]
pub fn read_main_splat(projected: &Tensor<f32>, idx: u32) -> (f32, f32, Sym2, f32) {
    let b = (idx * PROJECTED_LANES) as usize;
    (
        projected[b],
        projected[b + 1],
        Sym2 {
            c00: projected[b + 2],
            c01: projected[b + 3],
            c11: projected[b + 4],
        },
        projected[b + 5],
    )
}

/// Read one projected splat from the flat `Tensor<f32>` storage.
#[cube]
pub fn read_projected_splat(projected: &Tensor<f32>, idx: u32) -> Splat {
    let b = (idx * PROJECTED_LANES) as usize;
    Splat {
        xy_x: projected[b],
        xy_y: projected[b + 1],
        conic_x: projected[b + 2],
        conic_y: projected[b + 3],
        conic_z: projected[b + 4],
        color_a: projected[b + 5],
        color_r: projected[b + 6],
        color_g: projected[b + 7],
        color_b: projected[b + 8],
    }
}

#[cube]
pub fn write_projected_splat(projected: &mut Tensor<f32>, idx: u32, splat: Splat) {
    let b = (idx * PROJECTED_LANES) as usize;
    projected[b] = splat.xy_x;
    projected[b + 1] = splat.xy_y;
    projected[b + 2] = splat.conic_x;
    projected[b + 3] = splat.conic_y;
    projected[b + 4] = splat.conic_z;
    projected[b + 5] = splat.color_a;
    projected[b + 6] = splat.color_r;
    projected[b + 7] = splat.color_g;
    projected[b + 8] = splat.color_b;
}

/// View-space transform of a world-space mean using the project uniforms'
/// 3x4 viewmat (column-major).
#[cube]
pub fn world_to_cam(mean: Vec3A, u: ProjectUniforms) -> Vec3A {
    u.view_rotation().mul_vec3(mean).add(u.view_translation())
}

#[cube]
pub fn read_mean_viewspace(transforms: &Tensor<f32>, base: usize, u: ProjectUniforms) -> Vec3A {
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    world_to_cam(mean, u)
}

#[cube]
pub fn read_scale(transforms: &Tensor<f32>, base: usize, log_scale_offset: f32) -> Vec3A {
    Vec3A::new(
        f32::exp(transforms[base + 7] + log_scale_offset),
        f32::exp(transforms[base + 8] + log_scale_offset),
        f32::exp(transforms[base + 9] + log_scale_offset),
    )
}

/// Opacity clamp applied after the floor compensation, matching
/// `fold_min_scale` on the host.
const FLOOR_OPACITY_EPS: f32 = 1e-6;

/// One splat's scale and opacity after the Mip-Splatting 3D filter, plus
/// what the backward needs to differentiate through it.
#[derive(CubeType, Copy, Clone)]
#[expand(derive(Clone, Copy))]
pub struct ScaleFloor {
    /// `s' = sqrt(s² + f²)` per axis.
    pub scale: Vec3A,
    /// Base opacity `sigmoid(raw) · coef`, clamped.
    pub opac: f32,
    /// Energy compensation `Π s / s'`: the inflated splat keeps its mass.
    pub coef: f32,
    /// `d(log s') / d(log s) = s² / (s² + f²)` per axis.
    pub ratio_sq: Vec3A,
    /// Whether the opacity clamp lets gradient through.
    pub opac_open: bool,
}

/// Apply the floor `min_scale[gid]` to a splat's scale and sigmoid opacity,
/// matching `fold_min_scale` on the host. With `has_min_scale` false this is
/// the identity and `min_scale` is never read.
#[cube]
pub fn apply_scale_floor(
    scale: Vec3A,
    opac_sig: f32,
    min_scale: &Tensor<f32>,
    gid: u32,
    #[comptime] has_min_scale: bool,
) -> ScaleFloor {
    let mut floored = scale;
    let mut opac = opac_sig;
    let mut coef = 1.0f32;
    let mut ratio_sq = Vec3A::new(1.0f32, 1.0f32, 1.0f32);
    let mut opac_open = true;
    if has_min_scale {
        let f = min_scale[gid as usize];
        let f2 = f * f;
        floored = Vec3A::new(
            f32::sqrt(scale.x() * scale.x() + f2),
            f32::sqrt(scale.y() * scale.y() + f2),
            f32::sqrt(scale.z() * scale.z() + f2),
        );
        let rx = scale.x() / floored.x();
        let ry = scale.y() / floored.y();
        let rz = scale.z() / floored.z();
        coef = rx * ry * rz;
        ratio_sq = Vec3A::new(rx * rx, ry * ry, rz * rz);
        let o = opac_sig * coef;
        opac_open = o > FLOOR_OPACITY_EPS && o < 1.0f32 - FLOOR_OPACITY_EPS;
        opac = clamp(o, FLOOR_OPACITY_EPS, 1.0f32 - FLOOR_OPACITY_EPS);
    }
    ScaleFloor {
        scale: floored,
        opac,
        coef,
        ratio_sq,
        opac_open,
    }
}

#[cube]
pub fn read_quat_unorm(transforms: &Tensor<f32>, base: usize) -> Quat {
    Quat::new(
        transforms[base + 3],
        transforms[base + 4],
        transforms[base + 5],
        transforms[base + 6],
    )
}
