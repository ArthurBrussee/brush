use crate::kernels::camera_model::JacobianClampLimits;
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec2, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

#[derive(CubeLaunch, CubeType, Copy, Clone, PartialEq, Debug, Default)]
pub struct PinholeParams {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl PinholeParams {
    pub fn to_launch_object<R: Runtime>(&self) -> PinholeParamsLaunch<R> {
        PinholeParamsLaunch::new(self.fx, self.fy, self.cx, self.cy)
    }
}

#[cube]
pub fn project_pinhole(point: Vec3A, params: PinholeParams) -> (f32, f32) {
    let inv_z = 1.0f32 / point.z();
    let u = params.fx * point.x() * inv_z + params.cx;
    let v = params.fy * point.y() * inv_z + params.cy;
    (u, v)
}

#[cube]
pub fn calculate_project_jacobian_pinhole(
    point: Vec3A,
    limits: JacobianClampLimits,
    params: PinholeParams,
) -> Mat2x3 {
    let PinholeParams {
        fx: focal_x,
        fy: focal_y,
        ..
    } = params;

    let x = point.x();
    let y = point.y();
    let z = point.z();

    let inv_z = 1.0f32 / z;
    let dx = focal_x * inv_z;
    let dy = focal_y * inv_z;

    let clamped_x = clamp(x * inv_z, limits.lim_neg_x, limits.lim_pos_x);
    let clamped_y = clamp(y * inv_z, limits.lim_neg_y, limits.lim_pos_y);

    Mat2x3 {
        c0: Vec2::new(dx, 0.0),
        c1: Vec2::new(0.0, dy),
        c2: Vec2::new(-dx * clamped_x, -dy * clamped_y),
    }
}

#[cube]
pub fn calculate_projection_vjp_pinhole(
    project_jacobian: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d: Vec2,
) -> Vec3A {
    let PinholeParams { fx, fy, .. } = u.pinhole_params;
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

    let mut v_mx = fx * inv_z * v_mean2d.x();
    let mut v_my = fy * inv_z * v_mean2d.y();
    let mut v_mz = -(fx * mx * v_mean2d.x() + fy * my * v_mean2d.y()) * inv_z2;

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
