//! Render-specific cube helpers. Generic math (`Vec3A`, `Quat`, `Mat3`,
//! `Sym2`, `sigmoid`, `is_finite_*`, `calc_sigma`, `inverse_sym2`,
//! `det2_strict`) lives in [`brush_cube`] — re-exported here for
//! convenience.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use super::types::{Mat3, PixelRect, ProjectUniforms, Quat, Splat, Sym2, TileBbox, Vec3A};
use crate::kernels::camera_model::{CameraModel, calculate_project_jacobian};
pub use brush_cube::{Sym3, calc_sigma, is_finite_f32, sigmoid};

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

/// `f32` lanes per projected splat in the *geometry* side-buffer (RaDe-GS).
/// Layout: 0:grad_x, 1:grad_y, 2:depth_c (ray-plane: radial center depth +
/// its per-pixel gradient), 3:n_x, 4:n_y, 5:n_z (view-space normal). Only
/// written/read when geometry is requested; the color pipeline is untouched.
pub const PROJECTED_GEO_LANES: u32 = 6;
pub const PROJECTED_GEO_LANES_USIZE: usize = PROJECTED_GEO_LANES as usize;

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
pub fn read_mean_viewspace(transforms: &Tensor<f32>, base: usize, u: ProjectUniforms) -> Vec3A {
    let mean = Vec3A::new(transforms[base], transforms[base + 1], transforms[base + 2]);
    world_to_cam(mean, u)
}

#[cube]
pub fn read_scale(transforms: &Tensor<f32>, base: usize) -> Vec3A {
    Vec3A::new(
        f32::exp(transforms[base + 7]),
        f32::exp(transforms[base + 8]),
        f32::exp(transforms[base + 9]),
    )
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

/// RaDe-GS per-Gaussian ray-plane depth + covariance normal (camera space).
///
/// Returns `(grad_depth, normal)` where `grad_depth = (grad_x, grad_y,
/// depth_c)` describes a depth that varies *linearly* across the splat
/// footprint: at pixel `(px, py)` the radial depth is
/// `grad_x·(cx-px) + grad_y·(cy-py) + depth_c` (`(cx,cy)` = 2D mean, px).
/// `depth_c = ‖mean_c‖` is the radial center depth. The normal is derived
/// from the camera-space inverse covariance (no min-axis flattening
/// assumption). Follows RaDe-GS `computeCov2D` (arxiv 2406.01467).
/// `quat` must already be normalized.
#[cube]
pub fn splat_view_rayplane(
    scale: Vec3A,
    quat: Quat,
    mean_c: Vec3A,
    u: ProjectUniforms,
) -> (Vec3A, Vec3A) {
    splat_view_rayplane_core(
        scale,
        quat,
        mean_c,
        u.view_rotation(),
        u.pinhole_params.fx,
        u.pinhole_params.fy,
    )
}

/// Core of [`splat_view_rayplane`] with the camera inputs passed explicitly
/// (so the backward finite-diff harness can drive it without a full
/// `ProjectUniforms`).
#[cube]
pub fn splat_view_rayplane_core(
    scale: Vec3A,
    quat: Quat,
    mean_c: Vec3A,
    view_rot: Mat3,
    fx: f32,
    fy: f32,
) -> (Vec3A, Vec3A) {
    let tx = mean_c.x();
    let ty = mean_c.y();
    let tz = mean_c.z();
    let l = mean_c.length();
    let uu = tx / tz;
    let vv = ty / tz;
    let uvh = Vec3A::new(uu, vv, 1.0f32);

    // Camera-space inverse covariance Σ_c⁻¹ = R_c diag(1/s²) R_cᵀ, with
    // R_c = view_rot·R(quat). Built as M Mᵀ for M = R_c diag(1/s).
    let r_c = view_rot.mul_mat3(quat.to_mat3());
    // Floor `1/s` at 1e6 (i.e. s >= 1e-6). A tighter floor (1e-9) lets a
    // flatten-collapsed axis push `1/s²` toward f32 overflow, flipping the
    // splat between the ray-plane and the degenerate fallback frame-to-frame.
    let inv_s = Vec3A::new(
        f32::min(1.0f32 / scale.x(), 1e6f32),
        f32::min(1.0f32 / scale.y(), 1e6f32),
        f32::min(1.0f32 / scale.z(), 1e6f32),
    );
    let cov_cam_inv = r_c.mul_diag(inv_s).outer_product_self();

    let uvh_m = cov_cam_inv.mul_vec3(uvh);

    let mut grad_depth = Vec3A::new(0.0f32, 0.0f32, l);
    let mut normal = Vec3A::new(0.0f32, 0.0f32, -1.0f32);
    if uvh_m.is_finite() && uvh_m.length() > 1e-20f32 {
        let uvh_mn = uvh_m.normalize();
        let vbn = uvh_mn.dot(uvh);

        let u2 = uu * uu;
        let v2 = vv * vv;
        let uv = uu * vv;
        let ray_len2 = u2 + v2 + 1.0f32;
        let factor_normal = l / ray_len2;

        // plane = nJ_inv · (uvh_mn / max(vbn, eps)); nJ_inv columns below.
        let q = uvh_mn.scale(1.0f32 / f32::max(vbn, 1e-7f32));
        let n_j_inv = Mat3::from_cols(
            Vec3A::new(v2 + 1.0f32, -uv, 0.0f32),
            Vec3A::new(-uv, u2 + 1.0f32, 0.0f32),
            Vec3A::new(-uu, -vv, 0.0f32),
        );
        let plane = n_j_inv.mul_vec3(q);

        // Sanitize the depth gradient: fall back to a flat plane (grad 0, depth
        // `l`) for a degenerate plane. Besides non-finite, we reject a gradient
        // that changes depth by more than `l` per pixel: that means the plane is
        // edge-on, or its covariance is so anisotropic (a flatten-collapsed axis
        // hitting the `1/s` floor) that `vbn` underflows and `q` blows the plane
        // up to ~1e7. Such a gradient is huge-but-finite, so a plain finite check
        // misses it and the rendered depth explodes.
        let grad_x = plane.x() * factor_normal / fx;
        let grad_y = plane.y() * factor_normal / fy;
        let gd_ok = is_finite_f32(grad_x)
            && is_finite_f32(grad_y)
            && f32::abs(grad_x) < l
            && f32::abs(grad_y) < l;
        grad_depth = Vec3A::new(
            select(gd_ok, grad_x, 0.0f32),
            select(gd_ok, grad_y, 0.0f32),
            l,
        );

        // normal = normalize(nJ · ray_n), ray_n = (-plane.x·fn, -plane.y·fn, -1).
        // Safe-normalize: fall back to camera-facing (0,0,-1) if `cn` ≈ 0.
        let ray_n = Vec3A::new(
            -plane.x() * factor_normal,
            -plane.y() * factor_normal,
            -1.0f32,
        );
        let n_j = Mat3::from_cols(
            Vec3A::new(1.0f32 / tz, 0.0f32, -tx / (tz * tz)),
            Vec3A::new(0.0f32, 1.0f32 / tz, -ty / (tz * tz)),
            Vec3A::new(tx / l, ty / l, tz / l),
        );
        let cn = n_j.mul_vec3(ray_n);
        let cn_len = cn.length();
        // Also fall back to camera-facing when the depth plane was rejected, so
        // a degenerate splat's normal stays consistent with its flat depth.
        let n_ok = gd_ok && cn.is_finite() && cn_len > 1e-12f32;
        let cn_unit = cn.scale(1.0f32 / f32::max(cn_len, 1e-12f32));
        normal = Vec3A::new(
            select(n_ok, cn_unit.x(), 0.0f32),
            select(n_ok, cn_unit.y(), 0.0f32),
            select(n_ok, cn_unit.z(), -1.0f32),
        );
    }
    (grad_depth, normal)
}

#[cube]
pub fn read_projected_geo(geo: &Tensor<f32>, idx: u32) -> (Vec3A, Vec3A) {
    let b = (idx * PROJECTED_GEO_LANES) as usize;
    (
        Vec3A::new(geo[b], geo[b + 1], geo[b + 2]),
        Vec3A::new(geo[b + 3], geo[b + 4], geo[b + 5]),
    )
}

#[cube]
pub fn write_projected_geo(geo: &mut Tensor<f32>, idx: u32, grad_depth: Vec3A, normal: Vec3A) {
    let b = (idx * PROJECTED_GEO_LANES) as usize;
    geo[b] = grad_depth.x();
    geo[b + 1] = grad_depth.y();
    geo[b + 2] = grad_depth.z();
    geo[b + 3] = normal.x();
    geo[b + 4] = normal.y();
    geo[b + 5] = normal.z();
}
