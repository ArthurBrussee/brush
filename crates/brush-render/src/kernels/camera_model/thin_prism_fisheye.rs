use crate::kernels::camera_model::kannala_brandt_4::{
    KannalaBrandt4Params, calculate_project_jacobian_kb4, calculate_projection_vjp_kb4, project_kb4,
};
use crate::kernels::camera_model::pinhole::PinholeParams;
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec2, Vec3A};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// COLMAP `THIN_PRISM_FISHEYE` parameters: Kannala-Brandt 4 radial distortion
/// plus Brown-Conrady tangential `(p1, p2)` plus a 2-coefficient thin-prism
/// term `(sx1, sy1)`. With (u, v) = (x/z, y/z) on the normalized image plane,
/// the model is
///
/// ```text
/// u_d = (theta_d / r) * u + 2 p1 u v + p2 (r² + 2 u²) + sx1 r²
/// v_d = (theta_d / r) * v + 2 p2 u v + p1 (r² + 2 v²) + sy1 r²
/// ```
///
/// where `theta_d` is the KB4 polynomial of `theta = atan(r)`.
#[derive(CubeLaunch, CubeType, Copy, Clone, PartialEq, Debug, Default)]
pub struct ThinPrismFisheyeParams {
    pub kb4: KannalaBrandt4Params,
    pub p1: f32,
    pub p2: f32,
    pub sx1: f32,
    pub sy1: f32,
}

impl ThinPrismFisheyeParams {
    pub fn to_launch_object<R: Runtime>(&self) -> ThinPrismFisheyeParamsLaunch<R> {
        ThinPrismFisheyeParamsLaunch::new(
            self.kb4.to_launch_object(),
            self.p1,
            self.p2,
            self.sx1,
            self.sy1,
        )
    }
}

/// `delta_u_tp = N_u / z²`, `delta_v_tp = N_v / z²` where `N_u`, `N_v` are
/// the order-2 polynomials in `(x, y)` introduced by tangential + thin-prism.
/// Returns `(N_u, N_v, dNu/dx, dNu/dy, dNv/dx, dNv/dy)` — everything the
/// Jacobian and VJP additions need from the un-divided polynomials.
#[cube]
#[allow(clippy::type_complexity)]
fn thin_prism_polys(
    x: f32,
    y: f32,
    params: ThinPrismFisheyeParams,
) -> (f32, f32, f32, f32, f32, f32) {
    let ThinPrismFisheyeParams {
        p1, p2, sx1, sy1, ..
    } = params;

    let x2 = x * x;
    let y2 = y * y;
    let xy = x * y;
    let r2 = x2 + y2;

    let nu = 2.0f32 * p1 * xy + p2 * (3.0f32 * x2 + y2) + sx1 * r2;
    let nv = 2.0f32 * p2 * xy + p1 * (x2 + 3.0f32 * y2) + sy1 * r2;
    let dnu_dx = 2.0f32 * (p1 * y + (3.0f32 * p2 + sx1) * x);
    let dnu_dy = 2.0f32 * (p1 * x + (p2 + sx1) * y);
    let dnv_dx = 2.0f32 * (p2 * y + (p1 + sy1) * x);
    let dnv_dy = 2.0f32 * (p2 * x + (3.0f32 * p1 + sy1) * y);

    (nu, nv, dnu_dx, dnu_dy, dnv_dx, dnv_dy)
}

#[cube]
pub fn project_tpf(
    point: Vec3A,
    pinhole_params: PinholeParams,
    params: ThinPrismFisheyeParams,
) -> (f32, f32) {
    let (u_kb4, v_kb4) = project_kb4(point, pinhole_params, params.kb4);

    let x = point.x();
    let y = point.y();
    let z = point.z();
    let inv_z = 1.0f32 / z;
    let inv_z2 = inv_z * inv_z;

    let (nu, nv, _, _, _, _) = thin_prism_polys(x, y, params);
    let PinholeParams { fx, fy, .. } = pinhole_params;
    (u_kb4 + fx * nu * inv_z2, v_kb4 + fy * nv * inv_z2)
}

#[cube]
pub fn calculate_project_jacobian_tpf(
    point: Vec3A,
    pinhole_params: PinholeParams,
    params: ThinPrismFisheyeParams,
) -> Mat2x3 {
    let kb4_jac = calculate_project_jacobian_kb4(point, pinhole_params, params.kb4);

    let PinholeParams { fx, fy, .. } = pinhole_params;
    let x = point.x();
    let y = point.y();
    let z = point.z();
    let inv_z = 1.0f32 / z;
    let inv_z2 = inv_z * inv_z;
    let inv_z3 = inv_z2 * inv_z;

    let (nu, nv, dnu_dx, dnu_dy, dnv_dx, dnv_dy) = thin_prism_polys(x, y, params);

    // d(delta_u_tp)/dx = dNu/dx / z²,  d(delta_u_tp)/dz = -2 Nu / z³
    let add_du_dx = fx * dnu_dx * inv_z2;
    let add_du_dy = fx * dnu_dy * inv_z2;
    let add_du_dz = -2.0f32 * fx * nu * inv_z3;
    let add_dv_dx = fy * dnv_dx * inv_z2;
    let add_dv_dy = fy * dnv_dy * inv_z2;
    let add_dv_dz = -2.0f32 * fy * nv * inv_z3;

    Mat2x3 {
        c0: Vec2::new(kb4_jac.c0.x() + add_du_dx, kb4_jac.c0.y() + add_dv_dx),
        c1: Vec2::new(kb4_jac.c1.x() + add_du_dy, kb4_jac.c1.y() + add_dv_dy),
        c2: Vec2::new(kb4_jac.c2.x() + add_du_dz, kb4_jac.c2.y() + add_dv_dz),
    }
}

/// VJP for ThinPrismFisheye. Delegates to the KB4 VJP for the radial part
/// (path 1 + KB4 second-derivatives in path 2) and adds the tangential +
/// thin-prism second-derivative contractions on top. `project_jacobian` is
/// the FULL Jacobian (already includes the additions), which is what KB4 VJP
/// needs for path 1 / `vj_*` to be correct.
#[allow(clippy::too_many_arguments)]
#[cube]
pub fn calculate_projection_vjp_tpf(
    project_jacobian: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d: Vec2,
    params: ThinPrismFisheyeParams,
) -> Vec3A {
    let kb4_grad = calculate_projection_vjp_kb4(
        project_jacobian,
        mean_c,
        cov_c,
        u,
        v_cov2d,
        v_mean2d,
        params.kb4,
    );

    let PinholeParams { fx, fy, .. } = u.pinhole_params;
    let ThinPrismFisheyeParams {
        p1, p2, sx1, sy1, ..
    } = params;

    let mx = mean_c.x();
    let my = mean_c.y();
    let mz = mean_c.z();
    let inv_z = 1.0f32 / mz;
    let inv_z2 = inv_z * inv_z;
    let inv_z3 = inv_z2 * inv_z;
    let inv_z4 = inv_z2 * inv_z2;

    let (nu, nv, dnu_dx, dnu_dy, dnv_dx, dnv_dy) = thin_prism_polys(mx, my, params);

    // Second derivatives of N_u, N_v w.r.t. (x, y). N is order 2 in (x, y),
    // so the spatial Hessian is constant in (x, y).
    let d2nu_dxx = 6.0f32 * p2 + 2.0f32 * sx1;
    let d2nu_dyy = 2.0f32 * p2 + 2.0f32 * sx1;
    let d2nu_dxy = 2.0f32 * p1;
    let d2nv_dxx = 2.0f32 * p1 + 2.0f32 * sy1;
    let d2nv_dyy = 6.0f32 * p1 + 2.0f32 * sy1;
    let d2nv_dxy = 2.0f32 * p2;

    // Hessian of `delta_u_tp = N_u / z²` (and same shape for delta_v_tp).
    // (x, y) block: d² / z².  z-mixed: chain through d/dz (N/z²) = -2N/z³.
    // zz: 6 N / z⁴.
    let h_u_00 = d2nu_dxx * inv_z2;
    let h_u_01 = d2nu_dxy * inv_z2;
    let h_u_11 = d2nu_dyy * inv_z2;
    let h_u_02 = -2.0f32 * dnu_dx * inv_z3;
    let h_u_12 = -2.0f32 * dnu_dy * inv_z3;
    let h_u_22 = 6.0f32 * nu * inv_z4;
    let h_v_00 = d2nv_dxx * inv_z2;
    let h_v_01 = d2nv_dxy * inv_z2;
    let h_v_11 = d2nv_dyy * inv_z2;
    let h_v_02 = -2.0f32 * dnv_dx * inv_z3;
    let h_v_12 = -2.0f32 * dnv_dy * inv_z3;
    let h_v_22 = 6.0f32 * nv * inv_z4;

    // Rebuild vj_u, vj_v from the full Jacobian (matches KB4 VJP layout).
    let tmp = v_cov2d.mul_mat2x3(project_jacobian);
    let vj_u0 = 2.0f32 * tmp.row0().dot(cov_c.row0());
    let vj_u1 = 2.0f32 * tmp.row0().dot(cov_c.row1());
    let vj_u2 = 2.0f32 * tmp.row0().dot(cov_c.row2());
    let vj_v0 = 2.0f32 * tmp.row1().dot(cov_c.row0());
    let vj_v1 = 2.0f32 * tmp.row1().dot(cov_c.row1());
    let vj_v2 = 2.0f32 * tmp.row1().dot(cov_c.row2());

    // v_mean_c[k] += sum_j vj_(i,j) * H_(i)[j, k].  Hessian is symmetric.
    let v_mx = fx * (vj_u0 * h_u_00 + vj_u1 * h_u_01 + vj_u2 * h_u_02)
        + fy * (vj_v0 * h_v_00 + vj_v1 * h_v_01 + vj_v2 * h_v_02);
    let v_my = fx * (vj_u0 * h_u_01 + vj_u1 * h_u_11 + vj_u2 * h_u_12)
        + fy * (vj_v0 * h_v_01 + vj_v1 * h_v_11 + vj_v2 * h_v_12);
    let v_mz = fx * (vj_u0 * h_u_02 + vj_u1 * h_u_12 + vj_u2 * h_u_22)
        + fy * (vj_v0 * h_v_02 + vj_v1 * h_v_12 + vj_v2 * h_v_22);

    Vec3A::new(
        kb4_grad.x() + v_mx,
        kb4_grad.y() + v_my,
        kb4_grad.z() + v_mz,
    )
}
