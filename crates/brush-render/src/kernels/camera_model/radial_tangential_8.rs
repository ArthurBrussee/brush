use crate::kernels::camera_model::{CameraParams, JacobianClampLimits};
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

#[cube]
pub fn project_rt8(point: Vec3A, camera_params: CameraParams) -> (f32, f32) {
    let x = point.x();
    let y = point.y();
    let z = point.z();
    let k1 = camera_params.param0;
    let k2 = camera_params.param1;
    let p1 = camera_params.param2;
    let p2 = camera_params.param3;
    let k3 = camera_params.param4;
    let k4 = camera_params.param5;
    let k5 = camera_params.param6;
    let k6 = camera_params.param7;

    let x_ = x / z;
    let y_ = y / z;

    let x_2 = x_ * x_;
    let y_2 = y_ * y_;
    let r2 = x_2 + y_2;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    let d = (1.0f32 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0f32 + k4 * r2 + k5 * r4 + k6 * r6);

    let x_y_ = x_ * y_;
    let x__ = x_ * d + 2.0f32 * p1 * x_y_ + p2 * (r2 + 2.0f32 * x_2);
    let y__ = y_ * d + 2.0f32 * p2 * x_y_ + p1 * (r2 + 2.0f32 * y_2);

    let u = camera_params.focal_x * x__ + camera_params.pixel_center_x;
    let v = camera_params.focal_y * y__ + camera_params.pixel_center_y;

    (u, v)
}

#[cube]
pub fn calculate_project_jacobian_rt8(
    point: Vec3A,
    limits: JacobianClampLimits,
    camera_params: CameraParams,
) -> Mat2x3 {
    let k1 = camera_params.param0;
    let k2 = camera_params.param1;
    let p1 = camera_params.param2;
    let p2 = camera_params.param3;
    let k3 = camera_params.param4;
    let k4 = camera_params.param5;
    let k5 = camera_params.param6;
    let k6 = camera_params.param7;

    let x = point.x();
    let y = point.y();
    let z = point.z();

    let inv_z = 1.0f32 / z;
    let inv_z2 = inv_z * inv_z;

    let x_n = clamp(x * inv_z, limits.lim_neg_x, limits.lim_pos_x);
    let y_n = clamp(y * inv_z, limits.lim_neg_y, limits.lim_pos_y);
    let xc = x_n * z;
    let yc = y_n * z;

    // --- Radial term R = N(r2) / Dn(r2) and its r2-derivative R' ---
    let r2 = x_n * x_n + y_n * y_n;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    let n_poly = 1.0f32 + k1 * r2 + k2 * r4 + k3 * r6;
    let dn_poly = 1.0f32 + k4 * r2 + k5 * r4 + k6 * r6;
    let np_poly = k1 + 2.0f32 * k2 * r2 + 3.0f32 * k3 * r4; // dN/d(r2)
    let dnp_poly = k4 + 2.0f32 * k5 * r2 + 3.0f32 * k6 * r4; // dDn/d(r2)

    let inv_dn = 1.0f32 / dn_poly;
    let inv_dn2 = inv_dn * inv_dn;

    let r_val = n_poly * inv_dn; // R
    let rp_val = (np_poly * dn_poly - n_poly * dnp_poly) * inv_dn2; // R'

    // --- Distortion Jacobian D = d(x_d, y_d) / d(x_n, y_n) ---
    //   D00 = R + 2 x_n² R' + 2 p1 y_n + 6 p2 x_n
    //   D01 = 2 x_n y_n R' + 2 p1 x_n + 2 p2 y_n
    //   D10 = D01
    //   D11 = R + 2 y_n² R' + 6 p1 y_n + 2 p2 x_n
    let d00 = r_val + 2.0f32 * x_n * x_n * rp_val + 2.0f32 * p1 * y_n + 6.0f32 * p2 * x_n;
    let d01 = 2.0f32 * x_n * y_n * rp_val + 2.0f32 * p1 * x_n + 2.0f32 * p2 * y_n;
    let d10 = d01;
    let d11 = r_val + 2.0f32 * y_n * y_n * rp_val + 6.0f32 * p1 * y_n + 2.0f32 * p2 * x_n;

    // --- J = diag(fx, fy) * D * M ---
    //   J[0,0] = fx * D00 / Z
    //   J[0,1] = fx * D01 / Z
    //   J[0,2] = -fx * (D00 * xc + D01 * yc) / Z²
    //   J[1,0] = fy * D10 / Z
    //   J[1,1] = fy * D11 / Z
    //   J[1,2] = -fy * (D10 * xc + D11 * yc) / Z²
    let fx = camera_params.focal_x;
    let fy = camera_params.focal_y;
    let du_dx = fx * d00 * inv_z;
    let du_dy = fx * d01 * inv_z;
    let du_dz = -fx * (d00 * xc + d01 * yc) * inv_z2;
    let dv_dx = fy * d10 * inv_z;
    let dv_dy = fy * d11 * inv_z;
    let dv_dz = -fy * (d10 * xc + d11 * yc) * inv_z2;

    Mat2x3 {
        c0_x: du_dx,
        c0_y: dv_dx,
        c1_x: du_dy,
        c1_y: dv_dy,
        c2_x: du_dz,
        c2_y: dv_dz,
    }
}

#[cube]
pub fn calculate_projection_vjp_rt8(
    _: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d_x: f32,
    v_mean2d_y: f32,
) -> Vec3A {
    let fx = u.camera_params.focal_x;
    let fy = u.camera_params.focal_y;
    let k1 = u.camera_params.param0;
    let k2 = u.camera_params.param1;
    let p1 = u.camera_params.param2;
    let p2 = u.camera_params.param3;
    let k3 = u.camera_params.param4;
    let k4 = u.camera_params.param5;
    let k5 = u.camera_params.param6;
    let k6 = u.camera_params.param7;
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

    let xc = mx_rz * mz;
    let yc = my_rz * mz;

    // Normalized image-plane coordinates at the surrogate point.
    // When unclamped, x = X/Z; when clamped, x = clamped_const (independent of X).
    let inv_z2 = inv_z * inv_z;
    let inv_z3 = inv_z2 * inv_z;
    let x = xc * inv_z;
    let y = yc * inv_z;

    let r2 = x * x + y * y;
    let r4 = r2 * r2;

    // Radial polynomials and their first/second derivatives w.r.t. r2.
    let n_poly = 1.0f32 + k1 * r2 + k2 * r4 + k3 * r2 * r4;
    let dn_poly = 1.0f32 + k4 * r2 + k5 * r4 + k6 * r2 * r4;
    let np_poly = k1 + 2.0f32 * k2 * r2 + 3.0f32 * k3 * r4; // dN/d(r2)
    let dnp_poly = k4 + 2.0f32 * k5 * r2 + 3.0f32 * k6 * r4; // dDn/d(r2)
    let npp_poly = 2.0f32 * k2 + 6.0f32 * k3 * r2; // d²N/d(r2)²
    let dnpp_poly = 2.0f32 * k5 + 6.0f32 * k6 * r2; // d²Dn/d(r2)²

    let inv_dn = 1.0f32 / dn_poly;
    let inv_dn2 = inv_dn * inv_dn;
    let inv_dn3 = inv_dn2 * inv_dn;

    let rr = n_poly * inv_dn; // R
    let rrp = (np_poly * dn_poly - n_poly * dnp_poly) * inv_dn2; // R' = dR/d(r2)
    let rrpp = (npp_poly * dn_poly * dn_poly
        - 2.0f32 * np_poly * dn_poly * dnp_poly
        - n_poly * dnpp_poly * dn_poly
        + 2.0f32 * n_poly * dnp_poly * dnp_poly)
        * inv_dn3; // R''

    // dR/dx, dR/dy and d(R')/dx, d(R')/dy via r2 = x²+y², dr2/dx = 2x.
    let rx = 2.0f32 * x * rrp;
    let ry = 2.0f32 * y * rrp;
    let rpx = 2.0f32 * x * rrpp; // dR'/dx
    let rpy = 2.0f32 * y * rrpp; // dR'/dy

    // Distortion Jacobian D[i, j] = d (x_d, y_d)_i / d (x, y)_j (2x2).
    //   D00 = R + 2 x² R' + 2 p1 y + 6 p2 x
    //   D01 = 2 x y R' + 2 p1 x + 2 p2 y
    //   D10 = D01
    //   D11 = R + 2 y² R' + 6 p1 y + 2 p2 x
    let d00 = rr + 2.0f32 * x * x * rrp + 2.0f32 * p1 * y + 6.0f32 * p2 * x;
    let d01 = 2.0f32 * x * y * rrp + 2.0f32 * p1 * x + 2.0f32 * p2 * y;
    let d10 = d01;
    let d11 = rr + 2.0f32 * y * y * rrp + 6.0f32 * p1 * y + 2.0f32 * p2 * x;

    // J_surr (2x3): J = diag(fx, fy) D M  with  M = [[1/Z, 0, -xc/Z²],
    //                                                 [0, 1/Z, -yc/Z²]]
    //   J[0,0] = fx * D00/Z         J[0,1] = fx * D01/Z         J[0,2] = -fx (D00 xc + D01 yc)/Z²
    //   J[1,0] = fy * D10/Z         J[1,1] = fy * D11/Z         J[1,2] = -fy (D10 xc + D11 yc)/Z²
    let js00 = fx * d00 * inv_z;
    let js01 = fx * d01 * inv_z;
    let js02 = -fx * (d00 * xc + d01 * yc) * inv_z2;
    let js10 = fy * d10 * inv_z;
    let js11 = fy * d11 * inv_z;
    let js12 = -fy * (d10 * xc + d11 * yc) * inv_z2;

    // --- J_eff = J_surr * S  (route through clamp) ---
    // Same idea as in the KB4 branch.
    let je00 = select(in_x, js00, 0.0f32);
    let je10 = select(in_x, js10, 0.0f32);
    let je01 = select(in_y, js01, 0.0f32);
    let je11 = select(in_y, js11, 0.0f32);
    let je02 = (select(in_x, 0.0f32, mx_rz * js00)) + (select(in_y, 0.0f32, my_rz * js01)) + js02;
    let je12 = (select(in_x, 0.0f32, mx_rz * js10)) + (select(in_y, 0.0f32, my_rz * js11)) + js12;

    // Path 1: v_mean_c += J_eff^T v_mean2d
    let mut v_mx = je00 * v_mean2d_x + je10 * v_mean2d_y;
    let mut v_my = je01 * v_mean2d_x + je11 * v_mean2d_y;
    let mut v_mz = je02 * v_mean2d_x + je12 * v_mean2d_y;

    // v_J_eff = 2 sym(v_cov2d) J_eff cov_c (2x3)
    let tmp = v_cov2d.mul_mat2x3(Mat2x3 {
        c0_x: je00,
        c0_y: je10,
        c1_x: je01,
        c1_y: je11,
        c2_x: je02,
        c2_y: je12,
    });
    let ve_u0 = 2.0f32 * tmp.row0().dot(cov_c.row0());
    let ve_u1 = 2.0f32 * tmp.row0().dot(cov_c.row1());
    let ve_u2 = 2.0f32 * tmp.row0().dot(cov_c.row2());
    let ve_v0 = 2.0f32 * tmp.row1().dot(cov_c.row0());
    let ve_v1 = 2.0f32 * tmp.row1().dot(cov_c.row1());
    let ve_v2 = 2.0f32 * tmp.row1().dot(cov_c.row2());

    // v_J_surr = v_J_eff * S^T
    let vs_u0 = select(in_x, ve_u0, mx_rz * ve_u2);
    let vs_v0 = select(in_x, ve_v0, mx_rz * ve_v2);
    let vs_u1 = select(in_y, ve_u1, my_rz * ve_u2);
    let vs_v1 = select(in_y, ve_v1, my_rz * ve_v2);
    let vs_u2 = ve_u2;
    let vs_v2 = ve_v2;

    // --- Second derivatives of D entries in normalized coords (x, y) ---
    // dD00/dx = 2 x R' + 4 x R' + 2 x² R'_x + 6 p2 = Rx + 4 x R' + 2 x² R'_x + 6 p2
    // dD00/dy = Ry + 2 x² R'_y + 2 p1
    // dD01/dx = 2 y R' + 2 x y R'_x + 2 p1
    // dD01/dy = 2 x R' + 2 x y R'_y + 2 p2
    // dD11/dx = Rx + 2 y² R'_x + 2 p2
    // dD11/dy = Ry + 4 y R' + 2 y² R'_y + 6 p1
    let dd00_dx = rx + 4.0f32 * x * rrp + 2.0f32 * x * x * rpx + 6.0f32 * p2;
    let dd00_dy = ry + 2.0f32 * x * x * rpy + 2.0f32 * p1;
    let dd01_dx = 2.0f32 * y * rrp + 2.0f32 * x * y * rpx + 2.0f32 * p1;
    let dd01_dy = 2.0f32 * x * rrp + 2.0f32 * x * y * rpy + 2.0f32 * p2;
    let dd10_dx = dd01_dx;
    let dd10_dy = dd01_dy;
    let dd11_dx = rx + 2.0f32 * y * y * rpx + 2.0f32 * p2;
    let dd11_dy = ry + 4.0f32 * y * rrp + 2.0f32 * y * y * rpy + 6.0f32 * p1;

    // d D / d (xc, yc, Z) via chain rule: dx/dxc = 1/Z, dx/dZ = -xc/Z²,
    // and similarly for y.
    let dd00_dxc = dd00_dx * inv_z;
    let dd00_dyc = dd00_dy * inv_z;
    let dd00_dz = -(xc * dd00_dx + yc * dd00_dy) * inv_z2;
    let dd01_dxc = dd01_dx * inv_z;
    let dd01_dyc = dd01_dy * inv_z;
    let dd01_dz = -(xc * dd01_dx + yc * dd01_dy) * inv_z2;
    let dd10_dxc = dd10_dx * inv_z;
    let dd10_dyc = dd10_dy * inv_z;
    let dd10_dz = -(xc * dd10_dx + yc * dd10_dy) * inv_z2;
    let dd11_dxc = dd11_dx * inv_z;
    let dd11_dyc = dd11_dy * inv_z;
    let dd11_dz = -(xc * dd11_dx + yc * dd11_dy) * inv_z2;

    // d J_surr / d (xc, yc, Z) (six 2x3 entries; nine numbers per output dim).
    // Recall:
    //   J[0,0] = fx D00 / Z
    //   J[0,1] = fx D01 / Z
    //   J[0,2] = -fx (D00 xc + D01 yc) / Z²
    //   J[1,*] same with fy, D1*.
    let djs00_dxc = fx * dd00_dxc * inv_z;
    let djs00_dyc = fx * dd00_dyc * inv_z;
    let djs00_dz = fx * (dd00_dz * inv_z - d00 * inv_z2);
    let djs01_dxc = fx * dd01_dxc * inv_z;
    let djs01_dyc = fx * dd01_dyc * inv_z;
    let djs01_dz = fx * (dd01_dz * inv_z - d01 * inv_z2);
    let djs10_dxc = fy * dd10_dxc * inv_z;
    let djs10_dyc = fy * dd10_dyc * inv_z;
    let djs10_dz = fy * (dd10_dz * inv_z - d10 * inv_z2);
    let djs11_dxc = fy * dd11_dxc * inv_z;
    let djs11_dyc = fy * dd11_dyc * inv_z;
    let djs11_dz = fy * (dd11_dz * inv_z - d11 * inv_z2);

    let djs02_dxc = -fx * (dd00_dxc * xc + d00 + dd01_dxc * yc) * inv_z2;
    let djs02_dyc = -fx * (dd00_dyc * xc + dd01_dyc * yc + d01) * inv_z2;
    let djs02_dz =
        -fx * ((dd00_dz * xc + dd01_dz * yc) * inv_z2 - 2.0f32 * (d00 * xc + d01 * yc) * inv_z3);
    let djs12_dxc = -fy * (dd10_dxc * xc + d10 + dd11_dxc * yc) * inv_z2;
    let djs12_dyc = -fy * (dd10_dyc * xc + dd11_dyc * yc + d11) * inv_z2;
    let djs12_dz =
        -fy * ((dd10_dz * xc + dd11_dz * yc) * inv_z2 - 2.0f32 * (d10 * xc + d11 * yc) * inv_z3);

    // --- Contractions c_k = sum_{i,j} v_J_surr[i,j] * d J_surr[i,j]/d (xc,yc,Z)_k ---
    // v_J_surr layout: row 0 = (vs_u0, vs_u1, vs_u2); row 1 = (vs_v0, vs_v1, vs_v2)
    // J_surr     layout: row 0 = (js00, js01, js02);  row 1 = (js10, js11, js12)
    let c_xc = vs_u0 * djs00_dxc
        + vs_u1 * djs01_dxc
        + vs_u2 * djs02_dxc
        + vs_v0 * djs10_dxc
        + vs_v1 * djs11_dxc
        + vs_v2 * djs12_dxc;
    let c_yc = vs_u0 * djs00_dyc
        + vs_u1 * djs01_dyc
        + vs_u2 * djs02_dyc
        + vs_v0 * djs10_dyc
        + vs_v1 * djs11_dyc
        + vs_v2 * djs12_dyc;
    let c_z = vs_u0 * djs00_dz
        + vs_u1 * djs01_dz
        + vs_u2 * djs02_dz
        + vs_v0 * djs10_dz
        + vs_v1 * djs11_dz
        + vs_v2 * djs12_dz;

    // Pull contraction back: v_mean_c += S^T (c_xc, c_yc, c_z)
    if in_x {
        v_mx += c_xc;
    }
    if in_y {
        v_my += c_yc;
    }
    v_mz += c_z;
    if !in_x {
        v_mz += mx_rz * c_xc;
    }
    if !in_y {
        v_mz += my_rz * c_yc;
    }

    Vec3A::new(v_mx, v_my, v_mz)
}
