use crate::kernels::camera_model::{CameraParams, JacobianClampLimits};
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

#[cube]
pub fn project_pinhole(point: Vec3A, camera_params: CameraParams) -> (f32, f32) {
    let inv_z = 1.0f32 / point.z();
    let u = camera_params.focal_x * point.x() * inv_z + camera_params.pixel_center_x;
    let v = camera_params.focal_y * point.y() * inv_z + camera_params.pixel_center_y;
    (u, v)
}

#[cube]
pub fn calculate_project_jacobian_pinhole(
    point: Vec3A,
    limits: JacobianClampLimits,
    camera_params: CameraParams,
) -> Mat2x3 {
    let x = point.x();
    let y = point.y();
    let z = point.z();

    let inv_z = 1.0f32 / z;
    let dx = camera_params.focal_x * inv_z;
    let dy = camera_params.focal_y * inv_z;

    let clamped_x = clamp(x * inv_z, limits.lim_neg_x, limits.lim_pos_x);
    let clamped_y = clamp(y * inv_z, limits.lim_neg_y, limits.lim_pos_y);

    Mat2x3 {
        c0_x: dx,
        c0_y: 0.0,
        c1_x: 0.0,
        c1_y: dy,
        c2_x: -dx * clamped_x,
        c2_y: -dy * clamped_y,
    }
}

#[cube]
pub fn calculate_projection_vjp_pinhole(
    project_jacobian: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d_x: f32,
    v_mean2d_y: f32,
) -> Vec3A {
    let fx = u.camera_params.focal_x;
    let fy = u.camera_params.focal_y;
    let JacobianClampLimits {
        lim_pos_x,
        lim_pos_y,
        lim_neg_x,
        lim_neg_y,
    } = u.jacobian_clamp_limits;

    let mx = mean_c.x();
    let my = mean_c.y();
    let mz = mean_c.z();
    let inv_z = 1.0f32 / mz;

    let mx_rz_raw = mx * inv_z;
    let my_rz_raw = my * inv_z;
    let mx_rz = clamp(mx_rz_raw, lim_neg_x, lim_pos_x);
    let my_rz = clamp(my_rz_raw, lim_neg_y, lim_pos_y);

    let in_x = mx_rz_raw <= lim_pos_x && mx_rz_raw >= lim_neg_x;
    let in_y = my_rz_raw <= lim_pos_y && my_rz_raw >= lim_neg_y;

    let inv_z2 = inv_z * inv_z;
    let inv_z3 = inv_z2 * inv_z;

    let mut v_mx = fx * inv_z * v_mean2d_x;
    let mut v_my = fy * inv_z * v_mean2d_y;
    let mut v_mz = -(fx * mx * v_mean2d_x + fy * my * v_mean2d_y) * inv_z2;

    // tmp = v_cov2d * J (2x3, col-major).
    let tmp = v_cov2d.mul_mat2x3(project_jacobian);
    // v_J = 2 * tmp * cov_c (only the four entries that feed v_mean3d).
    let vj00 = 2.0f32 * tmp.row0().dot(cov_c.row0());
    let vj11 = 2.0f32 * tmp.row1().dot(cov_c.row1());
    let vj20 = 2.0f32 * tmp.row0().dot(cov_c.row2());
    let vj21 = 2.0f32 * tmp.row1().dot(cov_c.row2());

    let tx = mz * clamp(mx_rz, lim_neg_x, lim_pos_x);
    let ty = mz * clamp(my_rz, lim_neg_y, lim_pos_y);

    if in_x {
        v_mx += -fx * inv_z2 * vj20;
    } else {
        v_mz += -fx * inv_z3 * vj20 * tx;
    }
    if in_y {
        v_my += -fy * inv_z2 * vj21;
    } else {
        v_mz += -fy * inv_z3 * vj21 * ty;
    }
    v_mz += -fx * inv_z2 * vj00 - fy * inv_z2 * vj11
        + 2.0f32 * fx * tx * inv_z3 * vj20
        + 2.0f32 * fy * ty * inv_z3 * vj21;

    Vec3A::new(v_mx, v_my, v_mz)
}
