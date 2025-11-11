use burn::cubecl;
use burn_cubecl::cubecl::prelude::*;

pub const TILE_WIDTH: u32 = 16;
pub const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;
pub const COV_BLUR: f32 = 0.3;

// Note: These struct definitions cause WGSL codegen issues in CubeCL
// where it tries to vectorize them and creates invalid vec3→vec4 casts.
// They are kept here commented out for reference but should not be used.
// Instead, use individual scalar variables in kernels.
//
// #[derive(Clone, Copy, CubeType)]
// pub struct PackedVec3 {
//     pub x: f32,
//     pub y: f32,
//     pub z: f32,
// }
//
// #[derive(Clone, Copy, CubeType)]
// pub struct ProjectedSplat {
//     pub xy_x: f32,
//     pub xy_y: f32,
//     pub conic_x: f32,
//     pub conic_y: f32,
//     pub conic_z: f32,
//     pub color_r: f32,
//     pub color_g: f32,
//     pub color_b: f32,
//     pub color_a: f32,
// }
//
// #[derive(Clone, Copy, CubeType)]
// pub struct CameraView {
//     pub r00: f32,
//     pub r01: f32,
//     pub r02: f32,
//     pub r10: f32,
//     pub r11: f32,
//     pub r12: f32,
//     pub r20: f32,
//     pub r21: f32,
//     pub r22: f32,
//     pub t_x: f32,
//     pub t_y: f32,
//     pub t_z: f32,
// }
//
// #[derive(Clone, Copy, CubeType)]
// pub struct Projection {
//     pub focal_x: f32,
//     pub focal_y: f32,
//     pub pixel_center_x: f32,
//     pub pixel_center_y: f32,
// }
//
// #[derive(Clone, Copy, CubeType)]
// pub struct ImageSize {
//     pub width: u32,
//     pub height: u32,
// }

// Helper function to compact bits for 2D z-order decoding
#[cube]
pub fn compact_bits_16(v: u32) -> u32 {
    let mut x = v & 0x55555555u32;
    x = (x | (x >> 1u32)) & 0x33333333u32;
    x = (x | (x >> 2u32)) & 0x0F0F0F0Fu32;
    x = (x | (x >> 4u32)) & 0x00FF00FFu32;
    x = (x | (x >> 8u32)) & 0x0000FFFFu32;
    x
}

// Decode z-order to 2D coordinates
#[cube]
pub fn decode_morton_2d(morton: u32) -> (u32, u32) {
    let x = compact_bits_16(morton);
    let y = compact_bits_16(morton >> 1u32);
    (x, y)
}

#[cube]
pub fn map_1d_to_2d(id: u32, tiles_per_row: u32) -> (u32, u32) {
    let tile_id = id / TILE_SIZE;
    let within_tile_id = id % TILE_SIZE;

    let tile_x = tile_id % tiles_per_row;
    let tile_y = tile_id / tiles_per_row;

    let morton = decode_morton_2d(within_tile_id);
    let x = tile_x * TILE_WIDTH + morton.0;
    let y = tile_y * TILE_WIDTH + morton.1;
    (x, y)
}

#[cube]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

// device helper to get 3D covariance from scale and quat parameters
#[cube]
pub fn quat_to_mat(
    quat_w: f32,
    quat_x: f32,
    quat_y: f32,
    quat_z: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    let w = quat_w;
    let x = quat_x;
    let y = quat_y;
    let z = quat_z;

    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    // Return mat3x3 as 9 values (row-major)
    // WGSL version returns columns, but we need rows, so we transpose
    // WGSL columns become our rows:
    let m00 = 1.0 - 2.0 * (y2 + z2); // was col0[0], now row0[0]
    let m01 = 2.0 * (xy - wz); // was col1[0], now row0[1]
    let m02 = 2.0 * (xz + wy); // was col2[0], now row0[2]

    let m10 = 2.0 * (xy + wz); // was col0[1], now row1[0]
    let m11 = 1.0 - 2.0 * (x2 + z2); // was col1[1], now row1[1]
    let m12 = 2.0 * (yz - wx); // was col2[1], now row1[2]

    let m20 = 2.0 * (xz - wy); // was col0[2], now row2[0]
    let m21 = 2.0 * (yz + wx); // was col1[2], now row2[1]
    let m22 = 1.0 - 2.0 * (x2 + y2); // was col2[2], now row2[2]

    (m00, m01, m02, m10, m11, m12, m20, m21, m22)
}

#[cube]
pub fn scale_to_mat(
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    (scale_x, 0.0, 0.0, 0.0, scale_y, 0.0, 0.0, 0.0, scale_z)
}

// Matrix multiplication for 3x3 matrices
#[cube]
pub fn mat3_mul(
    a00: f32,
    a01: f32,
    a02: f32,
    a10: f32,
    a11: f32,
    a12: f32,
    a20: f32,
    a21: f32,
    a22: f32,
    b00: f32,
    b01: f32,
    b02: f32,
    b10: f32,
    b11: f32,
    b12: f32,
    b20: f32,
    b21: f32,
    b22: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    let c00 = a00 * b00 + a01 * b10 + a02 * b20;
    let c01 = a00 * b01 + a01 * b11 + a02 * b21;
    let c02 = a00 * b02 + a01 * b12 + a02 * b22;
    let c10 = a10 * b00 + a11 * b10 + a12 * b20;
    let c11 = a10 * b01 + a11 * b11 + a12 * b21;
    let c12 = a10 * b02 + a11 * b12 + a12 * b22;
    let c20 = a20 * b00 + a21 * b10 + a22 * b20;
    let c21 = a20 * b01 + a21 * b11 + a22 * b21;
    let c22 = a20 * b02 + a21 * b12 + a22 * b22;
    (c00, c01, c02, c10, c11, c12, c20, c21, c22)
}

// Transpose of 3x3 matrix
#[cube]
pub fn mat3_transpose(
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m20: f32,
    m21: f32,
    m22: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    (m00, m10, m20, m01, m11, m21, m02, m12, m22)
}

#[cube]
pub fn calc_cov3d(
    scale_x: f32,
    scale_y: f32,
    scale_z: f32,
    quat_w: f32,
    quat_x: f32,
    quat_y: f32,
    quat_z: f32,
) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    let r = quat_to_mat(quat_w, quat_x, quat_y, quat_z);
    let s = scale_to_mat(scale_x, scale_y, scale_z);
    let m = mat3_mul(
        r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, s.0, s.1, s.2, s.3, s.4, s.5, s.6, s.7, s.8,
    );
    let mt = mat3_transpose(m.0, m.1, m.2, m.3, m.4, m.5, m.6, m.7, m.8);
    mat3_mul(
        m.0, m.1, m.2, m.3, m.4, m.5, m.6, m.7, m.8, mt.0, mt.1, mt.2, mt.3, mt.4, mt.5, mt.6,
        mt.7, mt.8,
    )
}

#[cube]
pub fn calc_cam_j(
    mean_c_x: f32,
    mean_c_y: f32,
    mean_c_z: f32,
    focal_x: f32,
    focal_y: f32,
    img_size_x: u32,
    img_size_y: u32,
    pixel_center_x: f32,
    pixel_center_y: f32,
) -> (f32, f32, f32, f32, f32, f32) {
    let lims_pos_x = (1.15 * f32::cast_from(img_size_x) - pixel_center_x) / focal_x;
    let lims_pos_y = (1.15 * f32::cast_from(img_size_y) - pixel_center_y) / focal_y;
    let lims_neg_x = (-0.15 * f32::cast_from(img_size_x) - pixel_center_x) / focal_x;
    let lims_neg_y = (-0.15 * f32::cast_from(img_size_y) - pixel_center_y) / focal_y;
    let rz = 1.0 / mean_c_z;

    let uv_x = f32::clamp(mean_c_x * rz, lims_neg_x, lims_pos_x);
    let uv_y = f32::clamp(mean_c_y * rz, lims_neg_y, lims_pos_y);

    let duv_dxy_x = focal_x * rz;
    let duv_dxy_y = focal_y * rz;

    // Return 2x3 matrix J in row-major format
    // J = [
    //   [duv_dxy_x, 0.0, -duv_dxy_x * uv_x],
    //   [0.0, duv_dxy_y, -duv_dxy_y * uv_y]
    // ]
    let j00 = duv_dxy_x;
    let j01 = 0.0;
    let j02 = -duv_dxy_x * uv_x;
    let j10 = 0.0;
    let j11 = duv_dxy_y;
    let j12 = -duv_dxy_y * uv_y;

    (j00, j01, j02, j10, j11, j12)
}

#[cube]
pub fn calc_cov2d(
    cov3d: (f32, f32, f32, f32, f32, f32, f32, f32, f32),
    mean_c_x: f32,
    mean_c_y: f32,
    mean_c_z: f32,
    focal_x: f32,
    focal_y: f32,
    img_size_x: u32,
    img_size_y: u32,
    pixel_center_x: f32,
    pixel_center_y: f32,
    viewmat: (f32, f32, f32, f32, f32, f32, f32, f32, f32),
) -> (f32, f32, f32, f32) {
    // Extract rotation matrix R from viewmat (first 3x3)
    let r = viewmat;

    // covar_cam = R * cov3d * R^T
    let rc = mat3_mul(
        r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, cov3d.0, cov3d.1, cov3d.2, cov3d.3, cov3d.4,
        cov3d.5, cov3d.6, cov3d.7, cov3d.8,
    );
    let rt = mat3_transpose(r.0, r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8);
    let covar_cam = mat3_mul(
        rc.0, rc.1, rc.2, rc.3, rc.4, rc.5, rc.6, rc.7, rc.8, rt.0, rt.1, rt.2, rt.3, rt.4, rt.5,
        rt.6, rt.7, rt.8,
    );

    let j = calc_cam_j(
        mean_c_x,
        mean_c_y,
        mean_c_z,
        focal_x,
        focal_y,
        img_size_x,
        img_size_y,
        pixel_center_x,
        pixel_center_y,
    );

    // cov2d = J * covar_cam * J^T
    // J is 2x3, covar_cam is 3x3, so J * covar_cam gives 2x3
    // J = [(j00, j01, j02), (j10, j11, j12)]
    let jc00 = j.0 * covar_cam.0 + j.1 * covar_cam.3 + j.2 * covar_cam.6;
    let jc01 = j.0 * covar_cam.1 + j.1 * covar_cam.4 + j.2 * covar_cam.7;
    let jc02 = j.0 * covar_cam.2 + j.1 * covar_cam.5 + j.2 * covar_cam.8;
    let jc10 = j.3 * covar_cam.0 + j.4 * covar_cam.3 + j.5 * covar_cam.6;
    let jc11 = j.3 * covar_cam.1 + j.4 * covar_cam.4 + j.5 * covar_cam.7;
    let jc12 = j.3 * covar_cam.2 + j.4 * covar_cam.5 + j.5 * covar_cam.8;

    // J^T is 3x2, multiply (2x3) * (3x2) gives 2x2
    // JC = [(jc00, jc01, jc02), (jc10, jc11, jc12)]
    // J^T = [(j00, j10), (j01, j11), (j02, j12)]
    let c00 = jc00 * j.0 + jc01 * j.1 + jc02 * j.2;
    let c01 = jc00 * j.3 + jc01 * j.4 + jc02 * j.5;
    let c10 = jc10 * j.0 + jc11 * j.1 + jc12 * j.2;
    let c11 = jc10 * j.3 + jc11 * j.4 + jc12 * j.5;

    // Add blur
    let c00 = c00 + COV_BLUR;
    let c11 = c11 + COV_BLUR;

    (c00, c01, c10, c11)
}

#[cube]
pub fn mat2_inverse(m00: f32, m01: f32, m10: f32, m11: f32) -> (f32, f32, f32, f32) {
    let det = m00 * m11 - m01 * m10;
    let mut inv00 = 0.0;
    let mut inv01 = 0.0;
    let mut inv10 = 0.0;
    let mut inv11 = 0.0;

    if det > 0.0 {
        let inv_det = 1.0 / det;
        inv00 = m11 * inv_det;
        inv01 = -m01 * inv_det;
        inv10 = -m10 * inv_det;
        inv11 = m00 * inv_det;
    }

    (inv00, inv01, inv10, inv11)
}

#[cube]
pub fn mat2_determinant(m00: f32, m01: f32, m10: f32, m11: f32) -> f32 {
    m00 * m11 - m01 * m10
}

#[cube]
pub fn cov_compensation(cov2d_x: f32, cov2d_y: f32, cov2d_z: f32) -> f32 {
    let cov_orig_x = cov2d_x - COV_BLUR;
    let cov_orig_y = cov2d_y;
    let cov_orig_z = cov2d_z - COV_BLUR;
    let det_orig = cov_orig_x * cov_orig_z - cov_orig_y * cov_orig_y;
    let det = cov2d_x * cov2d_z - cov2d_y * cov2d_y;
    f32::sqrt(f32::max(0.0, det_orig / det))
}

#[cube]
pub fn calc_sigma(
    pixel_coord_x: f32,
    pixel_coord_y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    xy_x: f32,
    xy_y: f32,
) -> f32 {
    let delta_x = pixel_coord_x - xy_x;
    let delta_y = pixel_coord_y - xy_y;
    0.5 * (conic_x * delta_x * delta_x + conic_z * delta_y * delta_y) + conic_y * delta_x * delta_y
}

#[cube]
pub fn calc_vis(
    pixel_coord_x: f32,
    pixel_coord_y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    xy_x: f32,
    xy_y: f32,
) -> f32 {
    let sigma = calc_sigma(
        pixel_coord_x,
        pixel_coord_y,
        conic_x,
        conic_y,
        conic_z,
        xy_x,
        xy_y,
    );
    f32::exp(-sigma)
}

#[cube]
pub fn compute_bbox_extent(cov2d: (f32, f32, f32, f32), power_threshold: f32) -> (f32, f32) {
    let extent_x = f32::sqrt(2.0 * power_threshold * cov2d.0);
    let extent_y = f32::sqrt(2.0 * power_threshold * cov2d.3);
    (extent_x, extent_y)
}

#[cube]
pub fn tile_rect(tile_x: u32, tile_y: u32) -> (f32, f32, f32, f32) {
    let rect_min_x = f32::cast_from(tile_x * TILE_WIDTH);
    let rect_min_y = f32::cast_from(tile_y * TILE_WIDTH);
    let rect_max_x = rect_min_x + f32::cast_from(TILE_WIDTH);
    let rect_max_y = rect_min_y + f32::cast_from(TILE_WIDTH);
    (rect_min_x, rect_min_y, rect_max_x, rect_max_y)
}

#[cube]
pub fn will_primitive_contribute(
    rect: (f32, f32, f32, f32),
    mean_x: f32,
    mean_y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    power_threshold: f32,
) -> bool {
    let x_left = mean_x < rect.0;
    let x_right = mean_x > rect.2;
    let in_x_range = !(x_left || x_right);

    let y_above = mean_y < rect.1;
    let y_below = mean_y > rect.3;
    let in_y_range = !(y_above || y_below);

    let in_rect = in_x_range && in_y_range;

    let closest_corner_x = if x_left { rect.0 } else { rect.2 };
    let closest_corner_y = if y_above { rect.1 } else { rect.3 };

    let width = rect.2 - rect.0;
    let height = rect.3 - rect.1;
    // WGSL: select(-width, width, x_left) means "if x_left then width else -width"
    let d_x = if x_left { width } else { -width };
    let d_y = if y_above { height } else { -height };

    let diff_x = mean_x - closest_corner_x;
    let diff_y = mean_y - closest_corner_y;

    let t_val_x = f32::clamp(
        (d_x * conic_x * diff_x + d_x * conic_y * diff_y) / (d_x * conic_x * d_x),
        0.0,
        1.0,
    );
    // WGSL: select(clamp_result, 0.0, in_y_range) means "if in_y_range then 0.0 else clamp_result"
    // CubeCL: Use branching with mutable variable
    let mut t_max_x = t_val_x;
    if in_y_range {
        t_max_x = 0.0;
    }

    let t_val_y = f32::clamp(
        (d_y * conic_y * diff_x + d_y * conic_z * diff_y) / (d_y * conic_z * d_y),
        0.0,
        1.0,
    );
    // WGSL: select(clamp_result, 0.0, in_x_range) means "if in_x_range then 0.0 else clamp_result"
    // CubeCL: Use branching with mutable variable
    let mut t_max_y = t_val_y;
    if in_x_range {
        t_max_y = 0.0;
    }

    // If mean is inside the rect, it definitely contributes (matches WGSL early return)
    // Use select to avoid computing max_power_in_tile if in_rect is true
    let max_contribution_point_x = closest_corner_x + t_max_x * d_x;
    let max_contribution_point_y = closest_corner_y + t_max_y * d_y;
    let max_power_in_tile = calc_sigma(
        max_contribution_point_x,
        max_contribution_point_y,
        conic_x,
        conic_y,
        conic_z,
        mean_x,
        mean_y,
    );

    let contributes = max_power_in_tile <= power_threshold;
    // If in_rect, return true; otherwise return contributes
    let mut result = contributes;
    if in_rect {
        result = true;
    }
    result
}

#[cube]
pub fn get_bbox(
    center_x: f32,
    center_y: f32,
    dims_x: f32,
    dims_y: f32,
    bounds_x: u32,
    bounds_y: u32,
) -> (u32, u32, u32, u32) {
    let bounds_x_f = f32::cast_from(bounds_x);
    let bounds_y_f = f32::cast_from(bounds_y);

    let min_x_f = f32::clamp(center_x - dims_x, 0.0, bounds_x_f);
    let min_y_f = f32::clamp(center_y - dims_y, 0.0, bounds_y_f);
    let max_x_f = f32::clamp(center_x + dims_x + 1.0, 0.0, bounds_x_f);
    let max_y_f = f32::clamp(center_y + dims_y + 1.0, 0.0, bounds_y_f);

    let min_x = u32::cast_from(min_x_f);
    let min_y = u32::cast_from(min_y_f);
    let max_x = u32::cast_from(max_x_f);
    let max_y = u32::cast_from(max_y_f);

    (min_x, min_y, max_x, max_y)
}

#[cube]
pub fn get_tile_bbox(
    pix_center_x: f32,
    pix_center_y: f32,
    pix_extent_x: f32,
    pix_extent_y: f32,
    tile_bounds_x: u32,
    tile_bounds_y: u32,
) -> (u32, u32, u32, u32) {
    let tile_width_f = f32::cast_from(TILE_WIDTH);
    let tile_center_x = pix_center_x / tile_width_f;
    let tile_center_y = pix_center_y / tile_width_f;
    let tile_extent_x = pix_extent_x / tile_width_f;
    let tile_extent_y = pix_extent_y / tile_width_f;
    get_bbox(
        tile_center_x,
        tile_center_y,
        tile_extent_x,
        tile_extent_y,
        tile_bounds_x,
        tile_bounds_y,
    )
}

#[cube]
pub fn ceil_div(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}
