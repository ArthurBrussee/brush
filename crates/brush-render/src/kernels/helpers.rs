//! Render-specific cube helpers. Generic math (`Vec3A`, `Quat`, `Mat3`,
//! `Sym2`, `sigmoid`, `is_finite_*`, `calc_sigma`, `inverse_sym2`,
//! `det2_strict`) lives in [`brush_cube`] — re-exported here for
//! convenience.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

pub use brush_cube::{
    calc_sigma, is_finite_f32, sigmoid,
};

use super::types::{PixelRect, ProjectUniforms, Quat, Splat, Sym2, TileBbox, Vec3A};

pub const TILE_WIDTH: u32 = 16;
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;

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
#[cube]
pub fn map_1d_to_2d(id: u32, tiles_per_row: u32) -> (u32, u32) {
    let tile_id = id / TILE_SIZE;
    let within = id % TILE_SIZE;
    let tile_x = tile_id % tiles_per_row;
    let tile_y = tile_id / tiles_per_row;
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

#[cube]
pub fn tile_rect(tx: u32, ty: u32) -> PixelRect {
    let min_x = (tx * TILE_WIDTH) as f32;
    let min_y = (ty * TILE_WIDTH) as f32;
    PixelRect {
        min_x,
        min_y,
        max_x: min_x + TILE_WIDTH as f32,
        max_y: min_y + TILE_WIDTH as f32,
    }
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
pub fn calc_cov2d(scale: Vec3A, quat: Quat, mean_c: Vec3A, u: ProjectUniforms, #[comptime] camera_model_id: i32) -> Sym2 {
    let ns = u.view_rotation().mul_mat3(quat.to_mat3()).mul_diag(scale);
    let cam_jac = u.camera.calc_jacobian(mean_c, u.img_w, u.img_h, camera_model_id);

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

/// Walk the tiles in `bb` in linearised mod/div order and count those
/// that pass `will_primitive_contribute`. Shared between
/// `project_forward` and `map_gaussians_to_intersect` so both dispatches
/// run *byte-identical* loop bodies. Drift between the two counts would
/// leave uninitialised slots in `compact_gid_from_isect`; map_gaussians
/// pads with a sentinel `tile_id` defensively in case it still happens.
#[cube]
pub fn count_contributing_tiles(
    bb: TileBbox,
    xy_x: f32,
    xy_y: f32,
    conic: Sym2,
    power_threshold: f32,
) -> u32 {
    let bb_w = bb.max_x - bb.min_x;
    let num_tiles_bbox = (bb.max_y - bb.min_y) * bb_w;
    let mut num_tiles_hit = 0u32;
    for tile_idx in 0u32..num_tiles_bbox {
        let tx = (tile_idx % bb_w) + bb.min_x;
        let ty = (tile_idx / bb_w) + bb.min_y;
        let rect = tile_rect(tx, ty);
        if will_primitive_contribute(rect, xy_x, xy_y, conic, power_threshold) {
            num_tiles_hit += 1u32;
        }
    }
    num_tiles_hit
}

/// Conservative tile-vs-gaussian intersection test (StopThePop).
#[cube]
pub fn will_primitive_contribute(
    rect: PixelRect,
    mx: f32,
    my: f32,
    conic: Sym2,
    power_threshold: f32,
) -> bool {
    let x_left = mx < rect.min_x;
    let x_right = mx > rect.max_x;
    let in_x_range = !(x_left || x_right);
    let y_above = my < rect.min_y;
    let y_below = my > rect.max_y;
    let in_y_range = !(y_above || y_below);

    let mut hit = in_x_range && in_y_range;
    if !hit {
        let corner_x = select(x_left, rect.min_x, rect.max_x);
        let corner_y = select(y_above, rect.min_y, rect.max_y);
        let width = rect.max_x - rect.min_x;
        let height = rect.max_y - rect.min_y;
        let dxf = select(x_left, width, -width);
        let dyf = select(y_above, height, -height);
        let diff_x = mx - corner_x;
        let diff_y = my - corner_y;

        let tx_raw =
            (dxf * conic.c00 * diff_x + dxf * conic.c01 * diff_y) / (dxf * conic.c00 * dxf);
        let ty_raw =
            (dyf * conic.c01 * diff_x + dyf * conic.c11 * diff_y) / (dyf * conic.c11 * dyf);
        let tx = select(in_y_range, 0.0f32, clamp(tx_raw, 0.0f32, 1.0f32));
        let ty = select(in_x_range, 0.0f32, clamp(ty_raw, 0.0f32, 1.0f32));

        let max_x = corner_x + tx * dxf;
        let max_y = corner_y + ty * dyf;
        hit = calc_sigma(max_x, max_y, conic, mx, my) <= power_threshold;
    }
    hit
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
pub fn get_camera_mean(transforms: &Tensor<f32>, base: usize, u: ProjectUniforms) -> Vec3A {
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    world_to_cam(mean, u)
}

#[cube]
pub fn get_scale(transforms: &Tensor<f32>, base: usize) -> Vec3A {
    Vec3A::new(
        f32::exp(transforms[base + 7]),
        f32::exp(transforms[base + 8]),
        f32::exp(transforms[base + 9]),
    )
}

#[cube]
pub fn get_quat_unorm(transforms: &Tensor<f32>, base: usize) -> Quat {
    Quat::new(
        transforms[base + 3],
        transforms[base + 4],
        transforms[base + 5],
        transforms[base + 6],
    )
}
