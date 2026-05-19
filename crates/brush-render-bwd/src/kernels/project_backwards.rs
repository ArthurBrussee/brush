//! Backward projection.

use brush_cube::{Mat2x3, Sym3};
use brush_render::camera::{KANNALA_BRANDT_4, PINHOLE};
use brush_render::kernels::helpers::{
    calc_cov2d, compensate_cov2d, get_quat_unorm, get_scale, sigmoid, world_to_cam,
};
use brush_render::kernels::sh::{num_sh_coeffs, sh_coeffs_to_color_vjp};
use brush_render::kernels::types::{Mat3, ProjectUniforms, Quat, Sym2, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use brush_render::kernels::helpers::{
    calc_cam_j, calc_cov2d, compensate_cov2d, inverse_sym2, is_finite_f32, sigmoid, world_to_cam,
};

pub const WG_SIZE: u32 = 256;

/// Apply the VJP of `q -> q / |q|` to a downstream quaternion gradient.
#[cube]
fn apply_normalize_vjp(q: Quat, g: Quat) -> Quat {
    let lsq = q.dot(q);
    let l = f32::sqrt(lsq);
    let inv = 1.0f32 / (l * lsq);
    let qw = q.w();
    let qx = q.x();
    let qy = q.y();
    let qz = q.z();
    let gw = g.w();
    let gx = g.x();
    let gy = g.y();
    let gz = g.z();
    // Quat stored as `(w, x, y, z)`:
    //   cross_complex = -((w,x,y) * (x,y,w)) = (-w*x, -x*y, -y*w)
    //   cross_scalar  = -((w,x,y) * z)       = (-w*z, -x*z, -y*z)
    let cc0 = -qw * qx;
    let cc1 = -qx * qy;
    let cc2 = -qy * qw;
    let cs0 = -qw * qz;
    let cs1 = -qx * qz;
    let cs2 = -qy * qz;
    let q_sqr_w = qw * qw;
    let q_sqr_x = qx * qx;
    let q_sqr_y = qy * qy;
    let q_sqr_z = qz * qz;
    Quat::new(
        ((lsq - q_sqr_w) * gw + cc0 * gx + cc2 * gy + cs0 * gz) * inv,
        (cc0 * gw + (lsq - q_sqr_x) * gx + cc1 * gy + cs1 * gz) * inv,
        (cc2 * gw + cc1 * gx + (lsq - q_sqr_y) * gy + cs2 * gz) * inv,
        (cs0 * gw + cs1 * gx + cs2 * gy + (lsq - q_sqr_z) * gz) * inv,
    )
}

/// VJP of `quat_to_mat`. `v_r` is column-major like `quat_to_mat`'s output.
#[cube]
fn quat_to_mat_vjp(q: Quat, v_r: Mat3) -> Quat {
    let qw = q.w();
    let qx = q.x();
    let qy = q.y();
    let qz = q.z();
    let w_grad =
        qx * (v_r.c1_z - v_r.c2_y) + qy * (v_r.c2_x - v_r.c0_z) + qz * (v_r.c0_y - v_r.c1_x);
    let x_grad = -2.0f32 * qx * (v_r.c1_y + v_r.c2_z)
        + qy * (v_r.c0_y + v_r.c1_x)
        + qz * (v_r.c0_z + v_r.c2_x)
        + qw * (v_r.c1_z - v_r.c2_y);
    let y_grad = qx * (v_r.c0_y + v_r.c1_x) - 2.0f32 * qy * (v_r.c0_x + v_r.c2_z)
        + qz * (v_r.c1_z + v_r.c2_y)
        + qw * (v_r.c2_x - v_r.c0_z);
    let z_grad = qx * (v_r.c0_z + v_r.c2_x) + qy * (v_r.c1_z + v_r.c2_y)
        - 2.0f32 * qz * (v_r.c0_x + v_r.c1_y)
        + qw * (v_r.c0_y - v_r.c1_x);
    Quat::new(
        2.0f32 * w_grad,
        2.0f32 * x_grad,
        2.0f32 * y_grad,
        2.0f32 * z_grad,
    )
}

/// VJP of `Minv = inverse(M)` for symmetric 2x2 matrices.
///
/// Returns the gradient w.r.t. `M` as a `Sym2` (the upstream grad is
/// also symmetric since the rasterize backward writes the conic grad
/// in symmetric form).
#[cube]
fn inverse2x2_vjp(minv: Sym2, v_minv: Sym2) -> Sym2 {
    // -P * dP/dP * P. Writing both as symmetric Sym2 keeps the math at
    // 5 muladd-pairs instead of expanding to 4x4 dense.
    let tmp00 = -minv.c00 * v_minv.c00 + -minv.c01 * v_minv.c01;
    let tmp01 = -minv.c01 * v_minv.c00 + -minv.c11 * v_minv.c01;
    let tmp10 = -minv.c00 * v_minv.c01 + -minv.c01 * v_minv.c11;
    let tmp11 = -minv.c01 * v_minv.c01 + -minv.c11 * v_minv.c11;
    Sym2 {
        c00: tmp00 * minv.c00 + tmp10 * minv.c01,
        c01: tmp01 * minv.c00 + tmp11 * minv.c01,
        c11: tmp01 * minv.c01 + tmp11 * minv.c11,
    }
}

/// VJP of the J-stage (calc_cam_j + persp). Returns gradient w.r.t.
/// `mean3d` given grads w.r.t. cov2d (`v_cov2d`) and mean2d (`v_mean2d`).
/// `cov_c` is the 3D covariance in camera space.
#[allow(clippy::too_many_arguments)]
#[cube]
fn persp_proj_vjp(
    cam_jac: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d_x: f32,
    v_mean2d_y: f32,
    #[comptime] camera_model_id: i32,
) -> Vec3A {
    if comptime![camera_model_id == PINHOLE] {
        let inv_z = 1.0f32 / mean_c.z();
        let inv_z2 = inv_z * inv_z;
        let inv_z3 = inv_z2 * inv_z;

        let mut v_mx = u.camera.focal_x * inv_z * v_mean2d_x;
        let mut v_my = u.camera.focal_y * inv_z * v_mean2d_y;
        let mut v_mz = -(u.camera.focal_x * mean_c.x() * v_mean2d_x
            + u.camera.focal_y * mean_c.y() * v_mean2d_y)
            * inv_z2;

        // tmp = v_cov2d * J (2x3, col-major).
        let tmp = v_cov2d.mul_mat2x3(cam_jac);
        // v_J = 2 * tmp * cov_c (only the four entries that feed v_mean3d).
        let vj00 = 2.0f32 * tmp.row0().dot(cov_c.row0());
        let vj11 = 2.0f32 * tmp.row1().dot(cov_c.row1());
        let vj20 = 2.0f32 * tmp.row0().dot(cov_c.row2());
        let vj21 = 2.0f32 * tmp.row1().dot(cov_c.row2());

        // FOV clipping limits — matches the forward `calc_cam_j`.
        let img_w_f = u.img_w as f32;
        let img_h_f = u.img_h as f32;

        let lim_x_pos = (1.15f32 * img_w_f - u.camera.pixel_center_x) / u.camera.focal_x;
        let lim_y_pos = (1.15f32 * img_h_f - u.camera.pixel_center_y) / u.camera.focal_y;
        let lim_x_neg = (-0.15f32 * img_w_f - u.camera.pixel_center_x) / u.camera.focal_x;
        let lim_y_neg = (-0.15f32 * img_h_f - u.camera.pixel_center_y) / u.camera.focal_y;

        let mx_rz = mean_c.x() * inv_z;
        let my_rz = mean_c.y() * inv_z;
        let tx = mean_c.z() * clamp(mx_rz, lim_x_neg, lim_x_pos);
        let ty = mean_c.z() * clamp(my_rz, lim_y_neg, lim_y_pos);

        let in_x = mx_rz <= lim_x_pos && mx_rz >= lim_x_neg;
        let in_y = my_rz <= lim_y_pos && my_rz >= lim_y_neg;

        if in_x {
            v_mx += -u.camera.focal_x * inv_z2 * vj20;
        } else {
            v_mz += -u.camera.focal_x * inv_z3 * vj20 * tx;
        }
        if in_y {
            v_my += -u.camera.focal_y * inv_z2 * vj21;
        } else {
            v_mz += -u.camera.focal_y * inv_z3 * vj21 * ty;
        }
        v_mz += -u.camera.focal_x * inv_z2 * vj00 - u.camera.focal_y * inv_z2 * vj11
            + 2.0f32 * u.camera.focal_x * tx * inv_z3 * vj20
            + 2.0f32 * u.camera.focal_y * ty * inv_z3 * vj21;

        Vec3A::new(v_mx, v_my, v_mz)
    } else if comptime![camera_model_id == KANNALA_BRANDT_4] {
        let mx = mean_c.x();
        let my = mean_c.y();
        let mz = mean_c.z();

        let fx = u.camera.focal_x;
        let fy = u.camera.focal_y;
        let k1 = u.camera.k1;
        let k2 = u.camera.k2;
        let k3 = u.camera.k3;
        let k4 = u.camera.k4;

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
        let p1 =
            1.0f32 + 3.0f32 * k1 * th2 + 5.0f32 * k2 * th4 + 7.0f32 * k3 * th6 + 9.0f32 * k4 * th8;
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

        // J entries (same as calc_jacobian; we could pass cam_jac through
        // instead but recomputing is cheap and keeps this self-contained).
        let j00 = fx * (dg_x * xr + theta_d * dxr_x);
        let j01 = fx * (dg_y * xr + theta_d * dxr_y);
        let j02 = fx * (dg_z * xr);
        let j10 = fy * (dg_x * yr + theta_d * dyr_x);
        let j11 = fy * (dg_y * yr + theta_d * dyr_y);
        let j12 = fy * (dg_z * yr);

        // --- Path 1: v_mean_c = J^T v_mean2d ---
        let mut v_mx = j00 * v_mean2d_x + j10 * v_mean2d_y;
        let mut v_my = j01 * v_mean2d_x + j11 * v_mean2d_y;
        let mut v_mz = j02 * v_mean2d_x + j12 * v_mean2d_y;

        // --- v_J = 2 * sym(v_cov2d) * J * cov_c   (2x3) ---
        // tmp = v_cov2d * J  (works because Sym2.mul_mat2x3 already does
        //   the symmetric multiply; v_J = 2 * tmp * cov_c).
        let tmp = v_cov2d.mul_mat2x3(Mat2x3 {
            c0_x: j00,
            c0_y: j10,
            c1_x: j01,
            c1_y: j11,
            c2_x: j02,
            c2_y: j12,
        });
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
    } else {
        panic!("not implemented")
    }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn project_backwards_kernel(
    transforms: &Tensor<f32>,
    raw_opac: &Tensor<f32>,
    global_from_compact_gid: &Tensor<u32>,
    v_rasterize_grads: &Tensor<f32>,
    v_transforms: &mut Tensor<f32>,
    v_coeffs: &mut Tensor<f32>,
    v_raw_opac: &mut Tensor<f32>,
    v_refine_weight: &mut Tensor<f32>,
    u: ProjectUniforms,
    #[comptime] mip_splatting: bool,
    #[comptime] sh_degree: u32,
    #[comptime] camera_model_id: i32,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= u.num_visible {
        terminate!();
    }

    let global_gid = global_from_compact_gid[compact_gid as usize];

    // Read upstream rasterize grads first. rasterize_bwd only writes for
    // splats that contributed to a pixel; non-contributing splats leave
    // v_rasterize_grads at zero and (since the dense outputs are zero-
    // init) we can return without writing anything at all.
    let rg_base = (compact_gid * 10u32) as usize;
    let v_mean2d_x = v_rasterize_grads[rg_base];
    let v_mean2d_y = v_rasterize_grads[rg_base + 1];
    let v_conics_x = v_rasterize_grads[rg_base + 2];
    let v_conics_y = v_rasterize_grads[rg_base + 3];
    let v_conics_z = v_rasterize_grads[rg_base + 4];
    let v_color_r = v_rasterize_grads[rg_base + 5];
    let v_color_g = v_rasterize_grads[rg_base + 6];
    let v_color_b = v_rasterize_grads[rg_base + 7];
    let v_alpha_in = v_rasterize_grads[rg_base + 8];
    let v_refine_in = v_rasterize_grads[rg_base + 9];

    let any_grad = v_mean2d_x != 0.0f32
        || v_mean2d_y != 0.0f32
        || v_conics_x != 0.0f32
        || v_conics_y != 0.0f32
        || v_conics_z != 0.0f32
        || v_color_r != 0.0f32
        || v_color_g != 0.0f32
        || v_color_b != 0.0f32
        || v_alpha_in != 0.0f32
        || v_refine_in != 0.0f32;
    if !any_grad {
        terminate!();
    }

    let tbase = (global_gid * 10u32) as usize;
    let mean = Vec3A::new(
        transforms[tbase],
        transforms[tbase + 1],
        transforms[tbase + 2],
    );
    let scale = get_scale(transforms, tbase);
    let quat_unorm = get_quat_unorm(transforms, tbase);
    let quat = quat_unorm.normalize();

    // viewdir + SH VJP
    let v = mean.sub(u.camera_pos()).normalize();
    let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let v_color = Vec3A::new(v_color_r, v_color_g, v_color_b);
    sh_coeffs_to_color_vjp(v_coeffs, coeff_base, sh_degree, v, v_color);

    let mean_c = world_to_cam(mean, u);

    let r = quat.to_mat3();
    let m = r.mul_diag(scale);

    let raw_cov = calc_cov2d(scale, quat, mean_c, u, camera_model_id);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac_sig = sigmoid(raw_opac[global_gid as usize]);
    v_raw_opac[global_gid as usize] = filter_comp * v_alpha_in * opac_sig * (1.0f32 - opac_sig);

    // Make sure to keep refine weight >= 0 and finite. Helps with super large degenerate splats
    // that sum up their refine weight to some massive value.
    let refine_clean = select(is_finite_f32(v_refine_in), v_refine_in, 0.0f32);
    v_refine_weight[global_gid as usize] = clamp(refine_clean, 0.0f32, 1.0e32f32);

    let conic_inv = cov.inverse();
    let v_inv = Sym2 {
        c00: v_conics_x,
        c01: v_conics_y * 0.5f32,
        c11: v_conics_z,
    };
    let v_cov2d = inverse2x2_vjp(conic_inv, v_inv);

    // covar = M * M^T (symmetric).
    let covar = m.outer_product_self();

    // covar_c = R_cam * covar * R_cam^T (symmetric).
    let view_rot = u.view_rotation();
    let cov_c = covar.congruence(view_rot);

    let cam_jac = u
        .camera
        .calc_jacobian(mean_c, u.img_w, u.img_h, camera_model_id);
    let v_mean_c = persp_proj_vjp(
        cam_jac,
        mean_c,
        cov_c,
        u,
        v_cov2d,
        v_mean2d_x,
        v_mean2d_y,
        camera_model_id,
    );

    // v_covar_c = J^T * v_cov2d * J (2x2 sym → 3x3 sym).
    let vcc = cam_jac.transpose_congruence_sym2(v_cov2d);

    // v_mean = R^T * v_mean_c.
    let v_mean = view_rot.transpose_mul_vec3(v_mean_c);

    // v_covar = R^T * v_covar_c * R (symmetric).
    // v_M = (v_covar + v_covar^T) * M = 2 * v_covar * M.
    let v_m = vcc.transpose_congruence(view_rot).scale(2.0f32).mul_mat3(m);

    // v_scale = (R[i] dot v_M[i]) * exp(log_scale).
    let v_scale_exp = Vec3A::new(
        r.col0().dot(v_m.col0()) * scale.x(),
        r.col1().dot(v_m.col1()) * scale.y(),
        r.col2().dot(v_m.col2()) * scale.z(),
    );

    // grad for quat from covar: v_quat = normalize_vjp(quat) *
    // quat_to_mat_vjp(quat, v_M * diag(scale)).
    let q_grad = quat_to_mat_vjp(quat, v_m.mul_diag(scale));
    let v_q = apply_normalize_vjp(quat_unorm, q_grad);

    // Write gradients to dense v_transforms.
    let vbase = (global_gid * 10u32) as usize;
    v_transforms[vbase] = v_mean.x();
    v_transforms[vbase + 1] = v_mean.y();
    v_transforms[vbase + 2] = v_mean.z();
    v_transforms[vbase + 3] = v_q.w();
    v_transforms[vbase + 4] = v_q.x();
    v_transforms[vbase + 5] = v_q.y();
    v_transforms[vbase + 6] = v_q.z();
    v_transforms[vbase + 7] = v_scale_exp.x();
    v_transforms[vbase + 8] = v_scale_exp.y();
    v_transforms[vbase + 9] = v_scale_exp.z();
}
