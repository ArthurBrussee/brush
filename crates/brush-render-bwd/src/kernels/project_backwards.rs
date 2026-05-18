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
        let Mat2x3 {
            c0_x: j00,
            c0_y: j01,
            c1_x: j10,
            c1_y: j11,
            c2_x: j20,
            c2_y: j21,
        } = cam_jac;

        // tmp = v_cov2d * J (2x3, col-major).
        let t00 = v_cov2d.c00 * j00 + v_cov2d.c01 * j01;
        let t01 = v_cov2d.c01 * j00 + v_cov2d.c11 * j01;
        let t10 = v_cov2d.c00 * j10 + v_cov2d.c01 * j11;
        let t11 = v_cov2d.c01 * j10 + v_cov2d.c11 * j11;
        let t20 = v_cov2d.c00 * j20 + v_cov2d.c01 * j21;
        let t21 = v_cov2d.c01 * j20 + v_cov2d.c11 * j21;

        let mut v_mx = j00 * v_mean2d_x + j01 * v_mean2d_y;
        let mut v_my = j10 * v_mean2d_x + j11 * v_mean2d_y;
        let mut v_mz = j20 * v_mean2d_x + j21 * v_mean2d_y;

        // v_J = 2 * tmp * cov3d — all 6 entries needed for fisheye.
        let vj00 = 2.0f32 * (t00 * cov_c.c00 + t10 * cov_c.c01 + t20 * cov_c.c02);
        let vj10 = 2.0f32 * (t00 * cov_c.c01 + t10 * cov_c.c11 + t20 * cov_c.c12);
        let vj20 = 2.0f32 * (t00 * cov_c.c02 + t10 * cov_c.c12 + t20 * cov_c.c22);
        let vj01 = 2.0f32 * (t01 * cov_c.c00 + t11 * cov_c.c01 + t21 * cov_c.c02);
        let vj11 = 2.0f32 * (t01 * cov_c.c01 + t11 * cov_c.c11 + t21 * cov_c.c12);
        let vj21 = 2.0f32 * (t01 * cov_c.c02 + t11 * cov_c.c12 + t21 * cov_c.c22);

        let x = mean_c.x();
        let y = mean_c.y();
        let z = mean_c.z();
        let r2 = x * x + y * y;
        let r = f32::sqrt(r2);
        let near_axis = r < 1e-6f32;
        let rz = 1.0f32 / z;
        let rz2 = rz * rz;
        let rz3 = rz2 * rz;

        // Pinhole fallback for d(J)/d(mean_c) used when r ≈ 0.
        let pin_v_mx_c = -u.camera.focal_x * rz2 * vj20;
        let pin_v_my_c = -u.camera.focal_y * rz2 * vj21;
        let pin_v_mz_c = -u.camera.focal_x * rz2 * vj00 - u.camera.focal_y * rz2 * vj11
            + 2.0f32 * u.camera.focal_x * x * rz3 * vj20
            + 2.0f32 * u.camera.focal_y * y * rz3 * vj21;

        // KB4 parameters.
        let d2 = r2 + z * z;
        let theta = f32::atan2(r, z);
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;
        let poly = 1.0f32
            + u.camera.k1 * theta2
            + u.camera.k2 * theta4
            + u.camera.k3 * theta6
            + u.camera.k4 * theta8;
        let dpoly = 2.0f32 * u.camera.k1 * theta
            + 4.0f32 * u.camera.k2 * theta2 * theta
            + 6.0f32 * u.camera.k3 * theta4 * theta
            + 8.0f32 * u.camera.k4 * theta6 * theta;
        let theta_d = theta * poly;
        let dtheta_d = poly + theta * dpoly;
        let ddtheta_d = 6.0f32 * u.camera.k1 * theta
            + 20.0f32 * u.camera.k2 * theta2 * theta
            + 42.0f32 * u.camera.k3 * theta4 * theta
            + 72.0f32 * u.camera.k4 * theta6 * theta;

        // A = dtheta_d * z / (r² * d2)
        // B = theta_d / r³
        // C = dtheta_d / d2
        let r3 = r2 * r;
        let r4 = r2 * r2;
        let d22 = d2 * d2;
        let a_val = dtheta_d * z / (r2 * d2);
        let b_val = theta_d / r3;
        let c_val = dtheta_d / d2;
        let fx = u.camera.focal_x;
        let fy = u.camera.focal_y;

        // Derivative scale factors: dA/dx = EA*x, dB/dx = EB*x, dC/dx = EC*x.
        let ea =
            z / (r2 * d22) * (ddtheta_d * z / r - 2.0f32 * dtheta_d * (2.0f32 * r2 + z * z) / r2);
        let eb = dtheta_d * z / (r4 * d2) - 3.0f32 * theta_d / (r4 * r);
        let ec = ddtheta_d * z / (r * d22) - 2.0f32 * dtheta_d / d22;
        // z-derivatives of A, B, C.
        let ea_z = (dtheta_d * (r2 - z * z) / r2 - ddtheta_d * z / r) / d22;
        let eb_z = -dtheta_d / (r2 * d2);
        let ec_z = (-ddtheta_d * r - 2.0f32 * dtheta_d * z) / d22;

        // Weighted sums.
        let vja = vj00 * fx * x * x + vj11 * fy * y * y;
        let vjb = vj00 * fx * y * y + vj11 * fy * x * x;
        let vjab = (vj10 * fx + vj01 * fy) * x * y;
        let vjc = vj20 * fx * x + vj21 * fy * y;

        let scale = ea * vja + eb * vjb + (ea - eb) * vjab - ec * vjc;

        let dir_x = 2.0f32 * a_val * x * vj00 * fx + (a_val - b_val) * y * (vj10 * fx + vj01 * fy)
            - c_val * vj20 * fx
            + 2.0f32 * b_val * x * vj11 * fy;
        let dir_y = 2.0f32 * a_val * y * vj11 * fy + (a_val - b_val) * x * (vj10 * fx + vj01 * fy)
            - c_val * vj21 * fy
            + 2.0f32 * b_val * y * vj00 * fx;

        let kb4_v_mx_c = x * scale + dir_x;
        let kb4_v_my_c = y * scale + dir_y;
        let kb4_v_mz_c = ea_z * vja + eb_z * vjb + (ea_z - eb_z) * vjab - ec_z * vjc;

        v_mx += select(near_axis, pin_v_mx_c, kb4_v_mx_c);
        v_my += select(near_axis, pin_v_my_c, kb4_v_my_c);
        v_mz += select(near_axis, pin_v_mz_c, kb4_v_mz_c);

        Vec3A::new(v_mx, v_my, v_mz)
    } else {
        todo!()
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
