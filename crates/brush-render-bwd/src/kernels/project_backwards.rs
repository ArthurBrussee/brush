//! Backward projection.

use brush_cube::{Vec2, is_finite_f32, sigmoid};
use brush_render::kernels::camera_model::CameraKind;
use brush_render::kernels::helpers::{
    calc_cov2d, compensate_cov2d, read_quat_unorm, read_scale, world_to_cam,
};
use brush_render::kernels::sh::{num_sh_coeffs, sh_coeffs_to_color_vjp};
use brush_render::kernels::types::{Mat3, ProjectUniforms, Quat, Sym2, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

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
    num_visible: u32,
    #[comptime] mip_splatting: bool,
    #[comptime] sh_degree: u32,
    #[comptime] camera_kind: CameraKind,
) {
    let compact_gid = ABSOLUTE_POS as u32;
    if compact_gid >= num_visible {
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
    let scale = read_scale(transforms, tbase);
    let quat_unorm = read_quat_unorm(transforms, tbase);
    let quat = quat_unorm.normalize();

    // viewdir + SH VJP
    let v = mean.sub(u.camera_pos()).normalize();
    let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let v_color = Vec3A::new(v_color_r, v_color_g, v_color_b);
    sh_coeffs_to_color_vjp(v_coeffs, coeff_base, sh_degree, v, v_color);

    let mean_c = world_to_cam(mean, u);

    let r = quat.to_mat3();
    let m = r.mul_diag(scale);

    let raw_cov = calc_cov2d(scale, quat, mean_c, u, camera_kind);
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

    let cam_jac = u.calculate_project_jacobian(mean_c, camera_kind);
    let v_mean_c = u.calculate_projection_vjp(
        cam_jac,
        mean_c,
        cov_c,
        v_cov2d,
        Vec2::new(v_mean2d_x, v_mean2d_y),
        camera_kind,
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
