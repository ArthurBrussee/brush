use super::{helpers::*, sh::*};
use burn::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// Project visible gaussians and compute their view-dependent colors
/// This kernel takes the visible gaussians identified by project_forward,
/// computes their 2D projections, and evaluates spherical harmonics for colors
#[cube(launch_unchecked)]
pub fn project_visible(
    means: &Tensor<f32>,
    log_scales: &Tensor<f32>,
    quats: &Tensor<f32>,
    coeffs: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    projected: &mut Tensor<f32>,
    viewmat: &Tensor<f32>,    // 16 floats (4x4 matrix, column-major)
    camera_pos: &Tensor<f32>, // 3 floats
    uniforms: &Tensor<f32>, // 8 floats: [focal_x, focal_y, pixel_center_x, pixel_center_y, img_size_x, img_size_y, num_visible, sh_degree]
) {
    let compact_gid = ABSOLUTE_POS;

    // Note: num_visible is at uniforms[6] but we use dynamic dispatch so don't need to check it
    // The GPU will only launch the correct number of threads based on the dynamic cube count
    let sh_degree = u32::cast_from(uniforms[7]);

    // Extract camera view matrix from buffer
    // viewmat is stored as columns: [col0[0..4], col1[0..4], col2[0..4], col3[0..4]]
    // Columns 0-2 contain rotation matrix R (3x3), column 3 contains translation t (3x1)
    let cam_r00 = viewmat[0]; // col0[0]
    let cam_r10 = viewmat[1]; // col0[1]
    let cam_r20 = viewmat[2]; // col0[2]
    let cam_t_x = viewmat[12]; // col3[0]
    let cam_r01 = viewmat[4]; // col1[0]
    let cam_r11 = viewmat[5]; // col1[1]
    let cam_r21 = viewmat[6]; // col1[2]
    let cam_t_y = viewmat[13]; // col3[1]
    let cam_r02 = viewmat[8]; // col2[0]
    let cam_r12 = viewmat[9]; // col2[1]
    let cam_r22 = viewmat[10]; // col2[2]
    let cam_t_z = viewmat[14]; // col3[2]

    let focal_x = uniforms[0];
    let focal_y = uniforms[1];
    let pixel_center_x = uniforms[2];
    let pixel_center_y = uniforms[3];

    let img_width = u32::cast_from(uniforms[4]);
    let img_height = u32::cast_from(uniforms[5]);

    // Read camera position as scalars (avoid PackedVec3 which may cause codegen issues)
    let cam_pos_x = camera_pos[0];
    let cam_pos_y = camera_pos[1];
    let cam_pos_z = camera_pos[2];

    // Get the global gaussian ID from the compacted array
    let global_gid = global_from_compact_gid[compact_gid];

    // Read mean position
    let mean_x = means[global_gid * 3];
    let mean_y = means[global_gid * 3 + 1];
    let mean_z = means[global_gid * 3 + 2];

    // Read log scales and apply exponential
    let log_scale_x = log_scales[global_gid * 3];
    let log_scale_y = log_scales[global_gid * 3 + 1];
    let log_scale_z = log_scales[global_gid * 3 + 2];
    let scale_x = f32::exp(log_scale_x);
    let scale_y = f32::exp(log_scale_y);
    let scale_z = f32::exp(log_scale_z);

    // Read quaternion and normalize
    let quat_w = quats[global_gid * 4];
    let quat_x = quats[global_gid * 4 + 1];
    let quat_y = quats[global_gid * 4 + 2];
    let quat_z = quats[global_gid * 4 + 3];

    // Normalize quaternion (safe since project_forward already filtered invalid ones)
    let quat_norm =
        f32::sqrt(quat_w * quat_w + quat_x * quat_x + quat_y * quat_y + quat_z * quat_z);
    let quat_w = quat_w / quat_norm;
    let quat_x = quat_x / quat_norm;
    let quat_y = quat_y / quat_norm;
    let quat_z = quat_z / quat_norm;

    // Read opacity and apply sigmoid
    let raw_opacity = raw_opacities[global_gid];
    let opac = sigmoid(raw_opacity);

    // Transform mean to camera space: mean_c = R * mean + t
    let mean_c_x = cam_r00 * mean_x + cam_r01 * mean_y + cam_r02 * mean_z + cam_t_x;
    let mean_c_y = cam_r10 * mean_x + cam_r11 * mean_y + cam_r12 * mean_z + cam_t_y;
    let mean_c_z = cam_r20 * mean_x + cam_r21 * mean_y + cam_r22 * mean_z + cam_t_z;

    // Calculate 3D covariance
    let cov3d = calc_cov3d(scale_x, scale_y, scale_z, quat_w, quat_x, quat_y, quat_z);

    // Project to 2D covariance
    // Note: viewmat buffer is column-major, but we need to pass R in row-major to calc_cov2d
    // We need to arrange them as rows: (r00,r01,r02), (r10,r11,r12), (r20,r21,r22)
    let viewmat_tuple = (
        cam_r00, cam_r01, cam_r02, cam_r10, cam_r11, cam_r12, cam_r20, cam_r21, cam_r22,
    );
    let cov2d = calc_cov2d(
        cov3d,
        mean_c_x,
        mean_c_y,
        mean_c_z,
        focal_x,
        focal_y,
        img_width,
        img_height,
        pixel_center_x,
        pixel_center_y,
        viewmat_tuple,
    );

    // Invert 2D covariance to get conic (precision matrix)
    let conic = mat2_inverse(cov2d.0, cov2d.1, cov2d.2, cov2d.3);

    // Project mean to 2D screen space
    let rz = 1.0 / mean_c_z;
    let mean2d_x = focal_x * mean_c_x * rz + pixel_center_x;
    let mean2d_y = focal_y * mean_c_y * rz + pixel_center_y;

    // Calculate view direction (normalized)
    let viewdir_x = mean_x - cam_pos_x;
    let viewdir_y = mean_y - cam_pos_y;
    let viewdir_z = mean_z - cam_pos_z;
    let viewdir_len =
        f32::sqrt(viewdir_x * viewdir_x + viewdir_y * viewdir_y + viewdir_z * viewdir_z);
    let viewdir_x = viewdir_x / viewdir_len;
    let viewdir_y = viewdir_y / viewdir_len;
    let viewdir_z = viewdir_z / viewdir_len;

    // Calculate number of SH coefficients based on degree
    let num_coeffs = num_sh_coeffs(sh_degree);

    // Base index for this gaussian's SH coefficients
    // Coefficients are stored as [N, num_coeffs, 3] where 3 is RGB
    let base_sh_idx = global_gid * num_coeffs * 3;

    // Evaluate spherical harmonics to get color
    let mut color_r = 0.0;
    let mut color_g = 0.0;
    let mut color_b = 0.0;
    sh_coeffs_to_color(
        sh_degree,
        viewdir_x,
        viewdir_y,
        viewdir_z,
        coeffs,
        base_sh_idx,
        &mut color_r,
        &mut color_g,
        &mut color_b,
    );

    // Add 0.5 offset (matching WGSL version which adds vec3f(0.5))
    let color_r = color_r + 0.5;
    let color_g = color_g + 0.5;
    let color_b = color_b + 0.5;

    // Write ProjectedSplat structure (9 floats per splat)
    // Layout: [xy_x, xy_y, conic_x, conic_y, conic_z, color_r, color_g, color_b, color_a]
    let base_out = compact_gid * 9;
    projected[base_out] = mean2d_x;
    projected[base_out + 1] = mean2d_y;
    projected[base_out + 2] = conic.0; // conic_x (diagonal)
    projected[base_out + 3] = conic.1; // conic_y (off-diagonal)
    projected[base_out + 4] = conic.3; // conic_z (diagonal)

    projected[base_out + 5] = color_r;
    projected[base_out + 6] = color_g;
    projected[base_out + 7] = color_b;
    projected[base_out + 8] = opac;
}
