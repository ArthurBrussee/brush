//! Backward projection. Mirrors `project_backwards.wgsl`.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::cube;
use burn_cubecl::cubecl::prelude::*;

use brush_render::kernels::helpers::{
    calc_cam_j, calc_cov2d, compensate_cov2d, inverse_sym2, sigmoid, world_to_cam,
};
use brush_render::kernels::sh::{num_sh_coeffs, sh_coeffs_to_color_vjp};
use brush_render::kernels::types::{Mat3, ProjectUniforms, Quat, Sym2, Vec3A};

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
    // WGSL `quat=(w,x,y,z)` stored as fields .x/.y/.z/.w gives:
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
    j00: f32,
    j01: f32,
    j10: f32,
    j11: f32,
    j20: f32,
    j21: f32,
    mean_c: Vec3A,
    cov_c: Mat3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d_x: f32,
    v_mean2d_y: f32,
) -> Vec3A {
    let rz = 1.0f32 / mean_c.z();
    let rz2 = rz * rz;
    let rz3 = rz2 * rz;

    let mut v_mx = u.focal_x * rz * v_mean2d_x;
    let mut v_my = u.focal_y * rz * v_mean2d_y;
    let mut v_mz =
        -(u.focal_x * mean_c.x() * v_mean2d_x + u.focal_y * mean_c.y() * v_mean2d_y) * rz2;

    // tmp = v_cov2d * J (2x3, col-major).
    let t00 = v_cov2d.c00 * j00 + v_cov2d.c01 * j01;
    let t01 = v_cov2d.c01 * j00 + v_cov2d.c11 * j01;
    let t10 = v_cov2d.c00 * j10 + v_cov2d.c01 * j11;
    let t11 = v_cov2d.c01 * j10 + v_cov2d.c11 * j11;
    let t20 = v_cov2d.c00 * j20 + v_cov2d.c01 * j21;
    let t21 = v_cov2d.c01 * j20 + v_cov2d.c11 * j21;
    // v_J = 2 * tmp * cov3d (cov3d is symmetric, so cov3d == cov3d^T).
    // Only the trailing column (vj20, vj21) and the diagonal-ish entries
    // (vj00, vj11) feed the v_mean3d formula below; the off-diagonal
    // entries are dropped.
    let vj00 = 2.0f32 * (t00 * cov_c.c0_x + t10 * cov_c.c1_x + t20 * cov_c.c2_x);
    let vj11 = 2.0f32 * (t01 * cov_c.c0_y + t11 * cov_c.c1_y + t21 * cov_c.c2_y);
    let vj20 = 2.0f32 * (t00 * cov_c.c0_z + t10 * cov_c.c1_z + t20 * cov_c.c2_z);
    let vj21 = 2.0f32 * (t01 * cov_c.c0_z + t11 * cov_c.c1_z + t21 * cov_c.c2_z);

    // FOV clipping limits — matches `helpers.wgsl::calc_cam_J`.
    let img_w_f = u.img_w as f32;
    let img_h_f = u.img_h as f32;
    let tan_fov_x = 0.5f32 * img_w_f / u.focal_x;
    let tan_fov_y = 0.5f32 * img_h_f / u.focal_y;
    let lim_x_pos = (img_w_f - u.pixel_center_x) / u.focal_x + 0.3f32 * tan_fov_x;
    let lim_x_neg = u.pixel_center_x / u.focal_x + 0.3f32 * tan_fov_x;
    let lim_y_pos = (img_h_f - u.pixel_center_y) / u.focal_y + 0.3f32 * tan_fov_y;
    let lim_y_neg = u.pixel_center_y / u.focal_y + 0.3f32 * tan_fov_y;
    let mx_rz = mean_c.x() * rz;
    let my_rz = mean_c.y() * rz;
    let tx = mean_c.z() * clamp(mx_rz, -lim_x_neg, lim_x_pos);
    let ty = mean_c.z() * clamp(my_rz, -lim_y_neg, lim_y_pos);

    let in_x = mx_rz <= lim_x_pos && mx_rz >= -lim_x_neg;
    let in_y = my_rz <= lim_y_pos && my_rz >= -lim_y_neg;

    if in_x {
        v_mx += -u.focal_x * rz2 * vj20;
    } else {
        v_mz += -u.focal_x * rz3 * vj20 * tx;
    }
    if in_y {
        v_my += -u.focal_y * rz2 * vj21;
    } else {
        v_mz += -u.focal_y * rz3 * vj21 * ty;
    }
    v_mz += -u.focal_x * rz2 * vj00 - u.focal_y * rz2 * vj11
        + 2.0f32 * u.focal_x * tx * rz3 * vj20
        + 2.0f32 * u.focal_y * ty * rz3 * vj21;

    Vec3A::new(v_mx, v_my, v_mz)
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
    let scale = Vec3A::new(
        f32::exp(transforms[tbase + 7]),
        f32::exp(transforms[tbase + 8]),
        f32::exp(transforms[tbase + 9]),
    );
    let quat_unorm = Quat::new(
        transforms[tbase + 3],
        transforms[tbase + 4],
        transforms[tbase + 5],
        transforms[tbase + 6],
    );
    let quat = quat_unorm.normalize();

    // viewdir + SH VJP
    let v = mean.sub(u.camera_pos()).normalize();
    let coeff_base = global_gid * comptime![num_sh_coeffs(sh_degree) * 3u32];
    let v_color = Vec3A::new(v_color_r, v_color_g, v_color_b);
    sh_coeffs_to_color_vjp(v_coeffs, coeff_base, sh_degree, v, v_color);

    let mean_c = world_to_cam(mean, u);

    let r = quat.to_mat3();
    let m = r.mul_diag(scale);

    let raw_cov = calc_cov2d(scale, quat, mean_c, u);
    let (cov, filter_comp) = compensate_cov2d(raw_cov, mip_splatting);
    let opac_sig = sigmoid(raw_opac[global_gid as usize]);
    v_raw_opac[global_gid as usize] = filter_comp * v_alpha_in * opac_sig * (1.0f32 - opac_sig);
    v_refine_weight[global_gid as usize] = v_refine_in;

    let conic_inv = inverse_sym2(cov);
    let v_inv = Sym2 {
        c00: v_conics_x,
        c01: v_conics_y * 0.5f32,
        c11: v_conics_z,
    };
    let v_cov2d = inverse2x2_vjp(conic_inv, v_inv);

    // covar = M * M^T (3x3 col-major).
    let cov_00 = m.c0_x * m.c0_x + m.c1_x * m.c1_x + m.c2_x * m.c2_x;
    let cov_01 = m.c0_x * m.c0_y + m.c1_x * m.c1_y + m.c2_x * m.c2_y;
    let cov_02 = m.c0_x * m.c0_z + m.c1_x * m.c1_z + m.c2_x * m.c2_z;
    let cov_11 = m.c0_y * m.c0_y + m.c1_y * m.c1_y + m.c2_y * m.c2_y;
    let cov_12 = m.c0_y * m.c0_z + m.c1_y * m.c1_z + m.c2_y * m.c2_z;
    let cov_22 = m.c0_z * m.c0_z + m.c1_z * m.c1_z + m.c2_z * m.c2_z;

    // covar_c = R_cam * covar * R_cam^T. R_cam is the top-left 3x3 of
    // the viewmat (column-major). covar is symmetric (cov_10 == cov_01,
    // cov_20 == cov_02, cov_21 == cov_12).
    let tmp00 = u.vm0_x * cov_00 + u.vm1_x * cov_01 + u.vm2_x * cov_02;
    let tmp01 = u.vm0_y * cov_00 + u.vm1_y * cov_01 + u.vm2_y * cov_02;
    let tmp02 = u.vm0_z * cov_00 + u.vm1_z * cov_01 + u.vm2_z * cov_02;
    let tmp10 = u.vm0_x * cov_01 + u.vm1_x * cov_11 + u.vm2_x * cov_12;
    let tmp11 = u.vm0_y * cov_01 + u.vm1_y * cov_11 + u.vm2_y * cov_12;
    let tmp12 = u.vm0_z * cov_01 + u.vm1_z * cov_11 + u.vm2_z * cov_12;
    let tmp20 = u.vm0_x * cov_02 + u.vm1_x * cov_12 + u.vm2_x * cov_22;
    let tmp21 = u.vm0_y * cov_02 + u.vm1_y * cov_12 + u.vm2_y * cov_22;
    let tmp22 = u.vm0_z * cov_02 + u.vm1_z * cov_12 + u.vm2_z * cov_22;
    let cov_c = Mat3 {
        c0_x: tmp00 * u.vm0_x + tmp10 * u.vm1_x + tmp20 * u.vm2_x,
        c0_y: tmp01 * u.vm0_x + tmp11 * u.vm1_x + tmp21 * u.vm2_x,
        c0_z: tmp02 * u.vm0_x + tmp12 * u.vm1_x + tmp22 * u.vm2_x,
        c1_x: tmp00 * u.vm0_y + tmp10 * u.vm1_y + tmp20 * u.vm2_y,
        c1_y: tmp01 * u.vm0_y + tmp11 * u.vm1_y + tmp21 * u.vm2_y,
        c1_z: tmp02 * u.vm0_y + tmp12 * u.vm1_y + tmp22 * u.vm2_y,
        c2_x: tmp00 * u.vm0_z + tmp10 * u.vm1_z + tmp20 * u.vm2_z,
        c2_y: tmp01 * u.vm0_z + tmp11 * u.vm1_z + tmp21 * u.vm2_z,
        c2_z: tmp02 * u.vm0_z + tmp12 * u.vm1_z + tmp22 * u.vm2_z,
    };

    let (j00, j01, j10, j11, j20, j21) = calc_cam_j(mean_c, u);
    let v_mc = persp_proj_vjp(
        j00, j01, j10, j11, j20, j21, mean_c, cov_c, u, v_cov2d, v_mean2d_x, v_mean2d_y,
    );

    // v_covar_c = J^T * v_covar2d * J (2x2 sym -> 3x3 sym).
    let a00 = v_cov2d.c00 * j00 + v_cov2d.c01 * j01;
    let a01 = v_cov2d.c01 * j00 + v_cov2d.c11 * j01;
    let a10 = v_cov2d.c00 * j10 + v_cov2d.c01 * j11;
    let a11 = v_cov2d.c01 * j10 + v_cov2d.c11 * j11;
    let a20 = v_cov2d.c00 * j20 + v_cov2d.c01 * j21;
    let a21 = v_cov2d.c01 * j20 + v_cov2d.c11 * j21;
    let vcc_00 = j00 * a00 + j01 * a01;
    let vcc_01 = j00 * a10 + j01 * a11;
    let vcc_02 = j00 * a20 + j01 * a21;
    let vcc_10 = j10 * a00 + j11 * a01;
    let vcc_11 = j10 * a10 + j11 * a11;
    let vcc_12 = j10 * a20 + j11 * a21;
    let vcc_20 = j20 * a00 + j21 * a01;
    let vcc_21 = j20 * a10 + j21 * a11;
    let vcc_22 = j20 * a20 + j21 * a21;

    // v_mean = R^T * v_mean_c.
    let v_mean_world = u.view_rotation().transpose_mul_vec3(v_mc);

    // v_covar = R^T * v_covar_c * R.
    let vt00 = vcc_00 * u.vm0_x + vcc_10 * u.vm0_y + vcc_20 * u.vm0_z;
    let vt01 = vcc_01 * u.vm0_x + vcc_11 * u.vm0_y + vcc_21 * u.vm0_z;
    let vt02 = vcc_02 * u.vm0_x + vcc_12 * u.vm0_y + vcc_22 * u.vm0_z;
    let vt10 = vcc_00 * u.vm1_x + vcc_10 * u.vm1_y + vcc_20 * u.vm1_z;
    let vt11 = vcc_01 * u.vm1_x + vcc_11 * u.vm1_y + vcc_21 * u.vm1_z;
    let vt12 = vcc_02 * u.vm1_x + vcc_12 * u.vm1_y + vcc_22 * u.vm1_z;
    let vt20 = vcc_00 * u.vm2_x + vcc_10 * u.vm2_y + vcc_20 * u.vm2_z;
    let vt21 = vcc_01 * u.vm2_x + vcc_11 * u.vm2_y + vcc_21 * u.vm2_z;
    let vt22 = vcc_02 * u.vm2_x + vcc_12 * u.vm2_y + vcc_22 * u.vm2_z;
    // v_M = (v_covar + v_covar^T) * M = 2 * v_covar * M (symmetric).
    let vc00 = 2.0f32 * (u.vm0_x * vt00 + u.vm0_y * vt01 + u.vm0_z * vt02);
    let vc01 = 2.0f32 * (u.vm0_x * vt10 + u.vm0_y * vt11 + u.vm0_z * vt12);
    let vc02 = 2.0f32 * (u.vm0_x * vt20 + u.vm0_y * vt21 + u.vm0_z * vt22);
    let vc10 = 2.0f32 * (u.vm1_x * vt00 + u.vm1_y * vt01 + u.vm1_z * vt02);
    let vc11 = 2.0f32 * (u.vm1_x * vt10 + u.vm1_y * vt11 + u.vm1_z * vt12);
    let vc12 = 2.0f32 * (u.vm1_x * vt20 + u.vm1_y * vt21 + u.vm1_z * vt22);
    let vc20 = 2.0f32 * (u.vm2_x * vt00 + u.vm2_y * vt01 + u.vm2_z * vt02);
    let vc21 = 2.0f32 * (u.vm2_x * vt10 + u.vm2_y * vt11 + u.vm2_z * vt12);
    let vc22 = 2.0f32 * (u.vm2_x * vt20 + u.vm2_y * vt21 + u.vm2_z * vt22);

    let v_m = Mat3 {
        c0_x: vc00 * m.c0_x + vc10 * m.c0_y + vc20 * m.c0_z,
        c0_y: vc01 * m.c0_x + vc11 * m.c0_y + vc21 * m.c0_z,
        c0_z: vc02 * m.c0_x + vc12 * m.c0_y + vc22 * m.c0_z,
        c1_x: vc00 * m.c1_x + vc10 * m.c1_y + vc20 * m.c1_z,
        c1_y: vc01 * m.c1_x + vc11 * m.c1_y + vc21 * m.c1_z,
        c1_z: vc02 * m.c1_x + vc12 * m.c1_y + vc22 * m.c1_z,
        c2_x: vc00 * m.c2_x + vc10 * m.c2_y + vc20 * m.c2_z,
        c2_y: vc01 * m.c2_x + vc11 * m.c2_y + vc21 * m.c2_z,
        c2_z: vc02 * m.c2_x + vc12 * m.c2_y + vc22 * m.c2_z,
    };

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
    v_transforms[vbase] = v_mean_world.x();
    v_transforms[vbase + 1] = v_mean_world.y();
    v_transforms[vbase + 2] = v_mean_world.z();
    v_transforms[vbase + 3] = v_q.w();
    v_transforms[vbase + 4] = v_q.x();
    v_transforms[vbase + 5] = v_q.y();
    v_transforms[vbase + 6] = v_q.z();
    v_transforms[vbase + 7] = v_scale_exp.x();
    v_transforms[vbase + 8] = v_scale_exp.y();
    v_transforms[vbase + 9] = v_scale_exp.z();
}
