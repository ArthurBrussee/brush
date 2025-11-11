use super::helpers::*;
use burn::cubecl;
use burn_cubecl::cubecl::prelude::*;

// Note: num_visible parameter should be &Atomic<u32> for proper atomic operations
// This requires the tensor to be created with atomic type

#[cube(launch_unchecked)]
pub fn project_forward(
    means: &Tensor<f32>,
    quats: &Tensor<f32>,
    log_scales: &Tensor<f32>,
    raw_opacities: &Tensor<f32>,
    global_from_compact_gid: &mut Tensor<u32>,
    depths: &mut Tensor<f32>,
    num_visible: &mut Tensor<Atomic<u32>>,
    total_splats: u32,
    viewmat: &Tensor<f32>,  // 12 floats
    uniforms: &Tensor<f32>, // 6 floats: [focal_x, focal_y, pixel_center_x, pixel_center_y, img_size_x, img_size_y]
) {
    let global_gid = ABSOLUTE_POS;

    if global_gid >= total_splats {
        terminate!();
    }

    // Extract camera view matrix from buffer
    let cam_r00 = viewmat[0];
    let cam_r10 = viewmat[1];
    let cam_r20 = viewmat[2];
    let cam_t_x = viewmat[3];
    let cam_r01 = viewmat[4];
    let cam_r11 = viewmat[5];
    let cam_r21 = viewmat[6];
    let cam_t_y = viewmat[7];
    let cam_r02 = viewmat[8];
    let cam_r12 = viewmat[9];
    let cam_r22 = viewmat[10];
    let cam_t_z = viewmat[11];

    let focal_x = uniforms[0];
    let focal_y = uniforms[1];
    let pixel_center_x = uniforms[2];
    let pixel_center_y = uniforms[3];

    let img_width = u32::cast_from(uniforms[4]);
    let img_height = u32::cast_from(uniforms[5]);

    // Project world space to camera space
    let mean_x = means[global_gid * 3];
    let mean_y = means[global_gid * 3 + 1];
    let mean_z = means[global_gid * 3 + 2];

    // mean_c = R * mean + t
    let mean_c_x = cam_r00 * mean_x + cam_r01 * mean_y + cam_r02 * mean_z + cam_t_x;
    let mean_c_y = cam_r10 * mean_x + cam_r11 * mean_y + cam_r12 * mean_z + cam_t_y;
    let mean_c_z = cam_r20 * mean_x + cam_r21 * mean_y + cam_r22 * mean_z + cam_t_z;

    // Check if this splat is 'valid' (aka visible). Phrase as positive to bail on NaN.
    if mean_c_z < 0.01 || mean_c_z > 1e10 {
        terminate!();
    }

    let log_scale_x = log_scales[global_gid * 3];
    let log_scale_y = log_scales[global_gid * 3 + 1];
    let log_scale_z = log_scales[global_gid * 3 + 2];
    let scale_x = f32::exp(log_scale_x);
    let scale_y = f32::exp(log_scale_y);
    let scale_z = f32::exp(log_scale_z);

    let mut quat_w = quats[global_gid * 4];
    let mut quat_x = quats[global_gid * 4 + 1];
    let mut quat_y = quats[global_gid * 4 + 2];
    let mut quat_z = quats[global_gid * 4 + 3];

    // Skip any invalid rotations
    let quat_norm_sqr = quat_w * quat_w + quat_x * quat_x + quat_y * quat_y + quat_z * quat_z;
    if quat_norm_sqr < 1e-6 {
        terminate!();
    }

    let inv_norm = 1.0 / f32::sqrt(quat_norm_sqr);
    quat_w *= inv_norm;
    quat_x *= inv_norm;
    quat_y *= inv_norm;
    quat_z *= inv_norm;

    let cov3d = calc_cov3d(scale_x, scale_y, scale_z, quat_w, quat_x, quat_y, quat_z);

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

    let det = mat2_determinant(cov2d.0, cov2d.1, cov2d.2, cov2d.3);
    if f32::abs(det) < 1e-24 {
        terminate!();
    }

    // Compute the projected mean (2D screen coordinates)
    let mean2d_x = focal_x * mean_c_x * (1.0 / mean_c_z) + pixel_center_x;
    let mean2d_y = focal_y * mean_c_y * (1.0 / mean_c_z) + pixel_center_y;

    let raw_opacity = raw_opacities[global_gid];
    let opac = sigmoid(raw_opacity);

    if opac < 1.0 / 255.0 {
        terminate!();
    }

    let power_threshold = Log::log(255.0 * opac);
    let extent = compute_bbox_extent(cov2d, power_threshold);
    let extent_x = extent.0;
    let extent_y = extent.1;

    if extent_x < 0.0 || extent_y < 0.0 {
        terminate!();
    }

    // Check if the splat is within image bounds
    let img_width_f = f32::cast_from(img_width);
    let img_height_f = f32::cast_from(img_height);

    if mean2d_x + extent_x <= 0.0
        || mean2d_x - extent_x >= img_width_f
        || mean2d_y + extent_y <= 0.0
        || mean2d_y - extent_y >= img_height_f
    {
        terminate!();
    }

    // If we got here, this splat is visible - atomically increment counter
    let write_id = Atomic::add(&num_visible[0], 1u32);

    // Write to output buffers
    global_from_compact_gid[write_id] = global_gid;
    depths[write_id] = mean_c_z;
}
