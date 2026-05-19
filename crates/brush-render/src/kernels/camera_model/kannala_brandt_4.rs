use crate::kernels::camera_model::pinhole::PinholeParams;
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;
use bytemuck::{ByteHash, NoUninit};

#[derive(CubeLaunch, CubeType, Copy, Clone, NoUninit, ByteHash, PartialEq, Debug)]
#[repr(C)]
pub struct KannalaBrandt4Params {
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub k4: f32,
}

#[cube]
pub fn project_kb4(
    point: Vec3A,
    pinhole_params: PinholeParams,
    #[comptime] params: KannalaBrandt4Params,
) -> (f32, f32) {
    let x = point.x();
    let y = point.y();
    let z = point.z();

    let PinholeParams { fx, fy, cx, cy } = pinhole_params;
    let KannalaBrandt4Params { k1, k2, k3, k4 } = params;

    let inv_z = 1.0f32 / z;
    let pinhole_u = fx * x * inv_z + cx;
    let pinhole_v = fy * y * inv_z + cy;

    let r = f32::sqrt(x * x + y * y);

    let theta = r.atan2(z);
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;
    let theta6 = theta2 * theta4;
    let theta8 = theta4 * theta4;
    let d = theta * (1.0f32 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

    let inv_r = 1.0f32 / r;
    let fisheye_u = fx * (d * x * inv_r) + cx;
    let fisheye_v = fy * (d * y * inv_r) + cy;

    let near_axis = r < 1e-6f32;
    (
        select(near_axis, pinhole_u, fisheye_u),
        select(near_axis, pinhole_v, fisheye_v),
    )
}

// This Jacobian calculation does not clamp the Jacobian,
// since the values do not blow up when theta increases
#[cube]
pub fn calculate_project_jacobian_kb4(
    point: Vec3A,
    pinhole_params: PinholeParams,
    #[comptime] params: KannalaBrandt4Params,
) -> Mat2x3 {
    let PinholeParams { fx, fy, .. } = pinhole_params;
    let KannalaBrandt4Params { k1, k2, k3, k4 } = params;

    let x = point.x();
    let y = point.y();
    let z = point.z();

    let inv_z = 1.0f32 / z;

    let x2 = x * x;
    let y2 = y * y;
    let xy = x * y;
    let r2 = x2 + y2;
    let r = r2.sqrt();

    // --- Radial / depth intermediates ---
    let inv_r = 1.0f32 / r;
    let inv_r3 = inv_r * inv_r * inv_r;
    let rho2 = r2 + z * z; // x² + y² + z²
    let inv_rho2 = 1.0f32 / rho2;
    let inv_rho2_r = inv_rho2 * inv_r;

    // --- KB4 angular distortion ---
    let theta = r.atan2(z);
    let theta2 = theta * theta;
    let theta4 = theta2 * theta2;
    let theta6 = theta4 * theta2;
    let theta8 = theta4 * theta4;
    let d = theta * (1.0f32 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
    let dd_dthetha = 1.0f32
        + 3.0f32 * k1 * theta2
        + 5.0f32 * k2 * theta4
        + 7.0f32 * k3 * theta6
        + 9.0f32 * k4 * theta8;

    // --- d theta / d (x, y, z) ---
    //   dtheta/dx =  x z / (rho² r),  dtheta/dy =  y z / (rho² r),
    //   dtheta/dz = -r / rho²
    let dth_dx = x * z * inv_rho2_r;
    let dth_dy = y * z * inv_rho2_r;
    let dth_dz = -r * inv_rho2;

    // --- d theta_d / d (x, y, z) (chain rule through theta) ---
    let dd_dx = dd_dthetha * dth_dx;
    let dd_dy = dd_dthetha * dth_dy;
    let dd_dz = dd_dthetha * dth_dz;

    // --- u-row of J ---
    // d(x/r)/dx = y²/r³,  d(x/r)/dy = -xy/r³,  d(x/r)/dz = 0
    let xr = x * inv_r; // x / r
    let dxr_dx = y2 * inv_r3;
    let dxr_dy = -xy * inv_r3;
    let du_dx = fx * (dd_dx * xr + d * dxr_dx);
    let du_dy = fx * (dd_dy * xr + d * dxr_dy);
    let du_dz = fx * (dd_dz * xr); // dxr_dz = 0

    // --- v-row of J ---
    // d(y/r)/dx = -xy/r³,  d(y/r)/dy = x²/r³,  d(y/r)/dz = 0
    let yr = y * inv_r; // y / r
    let dyr_dx = -xy * inv_r3;
    let dyr_dy = x2 * inv_r3;
    let dv_dx = fy * (dd_dx * yr + d * dyr_dx);
    let dv_dy = fy * (dd_dy * yr + d * dyr_dy);
    let dv_dz = fy * (dd_dz * yr); // dyr_dz = 0

    // On-axis (r → 0): Fall back to the pinhole Jacobian
    let near_axis = r < 1e-6f32;
    let dx = fx * inv_z;
    let dy = fy * inv_z;
    let pinhole_du_dx = dx;
    let pinhole_dv_dy = dy;
    let pinhole_du_dz = -dx * x * inv_z;
    let pinhole_dv_dz = -dy * y * inv_z;

    Mat2x3 {
        c0_x: select(near_axis, pinhole_du_dx, du_dx),
        c0_y: select(near_axis, 0.0, dv_dx),
        c1_x: select(near_axis, 0.0, du_dy),
        c1_y: select(near_axis, pinhole_dv_dy, dv_dy),
        c2_x: select(near_axis, pinhole_du_dz, du_dz),
        c2_y: select(near_axis, pinhole_dv_dz, dv_dz),
    }
}

#[cube]
pub fn calculate_projection_vjp_kb4(
    project_jacobian: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d_x: f32,
    v_mean2d_y: f32,
    #[comptime] params: KannalaBrandt4Params,
) -> Vec3A {
    let PinholeParams { fx, fy, .. } = u.pinhole_params;
    let KannalaBrandt4Params { k1, k2, k3, k4 } = params;

    let mx = mean_c.x();
    let my = mean_c.y();
    let mz = mean_c.z();

    // --- Forward intermediates (identical to calc_jacobian) ---
    let r2 = mx * mx + my * my;
    let r = r2.sqrt().max(1.0e-8f32);
    let rho2 = r2 + mz * mz;

    let theta = r.atan2(mz);
    let th2 = theta * theta;
    let th4 = th2 * th2;
    let th6 = th4 * th2;
    let th8 = th4 * th4;

    let theta_d = theta * (1.0f32 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8);
    // P1 = d theta_d / d theta
    let p1 = 1.0f32 + 3.0f32 * k1 * th2 + 5.0f32 * k2 * th4 + 7.0f32 * k3 * th6 + 9.0f32 * k4 * th8;
    // P2 = d^2 theta_d / d theta^2
    let p2 = 6.0f32 * k1 * theta
        + 20.0f32 * k2 * theta * th2
        + 42.0f32 * k3 * theta * th4
        + 72.0f32 * k4 * theta * th6;

    let inv_r = 1.0f32 / r;
    let inv_r3 = inv_r * inv_r * inv_r;
    let inv_r5 = inv_r3 * inv_r * inv_r;
    let inv_rho2 = 1.0f32 / rho2;
    let inv_rho2_sq = inv_rho2 * inv_rho2;
    let inv_rho2_r = inv_rho2 * inv_r;

    // d theta / d {x, y, z}
    let dth_x = mx * mz * inv_rho2_r;
    let dth_y = my * mz * inv_rho2_r;
    let dth_z = -r * inv_rho2;

    // x/r, y/r and their first derivatives (z-derivatives are zero)
    let xr = mx * inv_r;
    let yr = my * inv_r;
    let dxr_x = my * my * inv_r3;
    let dxr_y = -mx * my * inv_r3;
    let dyr_x = dxr_y;
    let dyr_y = mx * mx * inv_r3;

    // dg/d {x, y, z} where g = theta_d
    let dg_x = p1 * dth_x;
    let dg_y = p1 * dth_y;
    let dg_z = p1 * dth_z;

    let Mat2x3 {
        c0_x: j00,
        c0_y: j10,
        c1_x: j01,
        c1_y: j11,
        c2_x: j02,
        c2_y: j12,
    } = project_jacobian;

    // --- Path 1: v_mean_c = J^T v_mean2d ---
    let mut v_mx = j00 * v_mean2d_x + j10 * v_mean2d_y;
    let mut v_my = j01 * v_mean2d_x + j11 * v_mean2d_y;
    let mut v_mz = j02 * v_mean2d_x + j12 * v_mean2d_y;

    // --- v_J = 2 * sym(v_cov2d) * J * cov_c   (2x3) ---
    // tmp = v_cov2d * J  (works because Sym2.mul_mat2x3 already does
    //   the symmetric multiply; v_J = 2 * tmp * cov_c).
    let tmp = v_cov2d.mul_mat2x3(project_jacobian);
    // Rows of (tmp * cov_c) — same trick as in the pinhole branch.
    let vj_u0 = 2.0f32 * tmp.row0().dot(cov_c.row0());
    let vj_u1 = 2.0f32 * tmp.row0().dot(cov_c.row1());
    let vj_u2 = 2.0f32 * tmp.row0().dot(cov_c.row2());
    let vj_v0 = 2.0f32 * tmp.row1().dot(cov_c.row0());
    let vj_v1 = 2.0f32 * tmp.row1().dot(cov_c.row1());
    let vj_v2 = 2.0f32 * tmp.row1().dot(cov_c.row2());

    // --- Hessian of theta (symmetric 3x3, all 6 entries used) ---
    // d2 theta / dxx = z * (r2*rho2 - x^2 (3 r2 + z^2)) / (r^3 rho2^2)
    // d2 theta / dyy = z * (r2*rho2 - y^2 (3 r2 + z^2)) / (r^3 rho2^2)
    // d2 theta / dxy = -x y z (3 r2 + z^2) / (r^3 rho2^2)
    // d2 theta / dxz = x (r2 - z^2) / (r rho2^2)
    // d2 theta / dyz = y (r2 - z^2) / (r rho2^2)
    // d2 theta / dzz = 2 z r / rho2^2
    let three_r2_z2 = 3.0f32 * r2 + mz * mz;
    let r2_minus_z2 = r2 - mz * mz;
    let h_th_00 = mz * (r2 * rho2 - mx * mx * three_r2_z2) * inv_r3 * inv_rho2_sq;
    let h_th_11 = mz * (r2 * rho2 - my * my * three_r2_z2) * inv_r3 * inv_rho2_sq;
    let h_th_01 = -mx * my * mz * three_r2_z2 * inv_r3 * inv_rho2_sq;
    let h_th_02 = mx * r2_minus_z2 * inv_r * inv_rho2_sq;
    let h_th_12 = my * r2_minus_z2 * inv_r * inv_rho2_sq;
    let h_th_22 = 2.0f32 * mz * r * inv_rho2_sq;

    // --- Hessian of x/r and y/r (only xy-block is nonzero) ---
    // d2(x/r)/dxx = -3 x y^2 / r^5
    // d2(x/r)/dxy =  y (2 x^2 - y^2) / r^5
    // d2(x/r)/dyy =  x (2 y^2 - x^2) / r^5
    // d2(y/r)/dxx =  y (2 x^2 - y^2) / r^5
    // d2(y/r)/dxy =  x (2 y^2 - x^2) / r^5
    // d2(y/r)/dyy = -3 x^2 y / r^5
    let two_x2_my2 = 2.0f32 * mx * mx - my * my;
    let two_y2_mx2 = 2.0f32 * my * my - mx * mx;
    let h_xr_00 = -3.0f32 * mx * my * my * inv_r5;
    let h_xr_01 = my * two_x2_my2 * inv_r5;
    let h_xr_11 = mx * two_y2_mx2 * inv_r5;
    let h_yr_00 = my * two_x2_my2 * inv_r5;
    let h_yr_01 = mx * two_y2_mx2 * inv_r5;
    let h_yr_11 = -3.0f32 * mx * mx * my * inv_r5;
    // h_xr_02, h_xr_12, h_xr_22, h_yr_02, h_yr_12, h_yr_22 are all zero.

    // --- Path 2 contraction ---
    // === k = 0 (d/dx) ===
    {
        // (j, k) = (0, 0)
        let d2g = p2 * dth_x * dth_x + p1 * h_th_00;
        let d_ju = fx * (d2g * xr + dg_x * dxr_x + dg_x * dxr_x + theta_d * h_xr_00);
        let d_jv = fy * (d2g * yr + dg_x * dyr_x + dg_x * dyr_x + theta_d * h_yr_00);
        v_mx += vj_u0 * d_ju + vj_v0 * d_jv;
    }
    {
        // (j, k) = (1, 0)
        let d2g = p2 * dth_y * dth_x + p1 * h_th_01;
        let d_ju = fx * (d2g * xr + dg_y * dxr_x + dg_x * dxr_y + theta_d * h_xr_01);
        let d_jv = fy * (d2g * yr + dg_y * dyr_x + dg_x * dyr_y + theta_d * h_yr_01);
        v_mx += vj_u1 * d_ju + vj_v1 * d_jv;
    }
    {
        // (j, k) = (2, 0)  -- dh_*[2]=0, H_h_*[2,0]=0
        let d2g = p2 * dth_z * dth_x + p1 * h_th_02;
        let d_ju = fx * (d2g * xr + dg_x * 0.0f32 + dg_z * dxr_x);
        let d_jv = fy * (d2g * yr + dg_x * 0.0f32 + dg_z * dyr_x);
        v_mx += vj_u2 * d_ju + vj_v2 * d_jv;
    }

    // === k = 1 (d/dy) ===
    {
        // (j, k) = (0, 1)
        let d2g = p2 * dth_x * dth_y + p1 * h_th_01;
        let d_ju = fx * (d2g * xr + dg_x * dxr_y + dg_y * dxr_x + theta_d * h_xr_01);
        let d_jv = fy * (d2g * yr + dg_x * dyr_y + dg_y * dyr_x + theta_d * h_yr_01);
        v_my += vj_u0 * d_ju + vj_v0 * d_jv;
    }
    {
        // (j, k) = (1, 1)
        let d2g = p2 * dth_y * dth_y + p1 * h_th_11;
        let d_ju = fx * (d2g * xr + dg_y * dxr_y + dg_y * dxr_y + theta_d * h_xr_11);
        let d_jv = fy * (d2g * yr + dg_y * dyr_y + dg_y * dyr_y + theta_d * h_yr_11);
        v_my += vj_u1 * d_ju + vj_v1 * d_jv;
    }
    {
        // (j, k) = (2, 1)
        let d2g = p2 * dth_z * dth_y + p1 * h_th_12;
        let d_ju = fx * (d2g * xr + dg_y * 0.0f32 + dg_z * dxr_y);
        let d_jv = fy * (d2g * yr + dg_y * 0.0f32 + dg_z * dyr_y);
        v_my += vj_u2 * d_ju + vj_v2 * d_jv;
    }

    // === k = 2 (d/dz) ===
    {
        // (j, k) = (0, 2)  -- dh_*[2]=0, H_h_*[0,2]=0
        let d2g = p2 * dth_x * dth_z + p1 * h_th_02;
        let d_ju = fx * (d2g * xr + dg_z * dxr_x + dg_x * 0.0f32);
        let d_jv = fy * (d2g * yr + dg_z * dyr_x + dg_x * 0.0f32);
        v_mz += vj_u0 * d_ju + vj_v0 * d_jv;
    }
    {
        // (j, k) = (1, 2)
        let d2g = p2 * dth_y * dth_z + p1 * h_th_12;
        let d_ju = fx * (d2g * xr + dg_z * dxr_y + dg_y * 0.0f32);
        let d_jv = fy * (d2g * yr + dg_z * dyr_y + dg_y * 0.0f32);
        v_mz += vj_u1 * d_ju + vj_v1 * d_jv;
    }
    {
        // (j, k) = (2, 2)  -- all h_* and dh_* z-parts are zero
        let d2g = p2 * dth_z * dth_z + p1 * h_th_22;
        let d_ju = fx * (d2g * xr);
        let d_jv = fy * (d2g * yr);
        v_mz += vj_u2 * d_ju + vj_v2 * d_jv;
    }

    Vec3A::new(v_mx, v_my, v_mz)
}
