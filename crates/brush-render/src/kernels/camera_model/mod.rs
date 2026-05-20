pub mod kannala_brandt_4;
pub mod pinhole;
pub mod radial_tangential_8;

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

use crate::kernels::camera_model::CameraModel::{KannalaBrandt4, Pinhole, RadialTangential8};
use crate::kernels::camera_model::kannala_brandt_4::{
    KannalaBrandt4Params, calculate_project_jacobian_kb4, calculate_projection_vjp_kb4, project_kb4,
};
use crate::kernels::camera_model::pinhole::{
    PinholeParams, calculate_project_jacobian_pinhole, calculate_projection_vjp_pinhole,
    project_pinhole,
};
use crate::kernels::camera_model::radial_tangential_8::{
    RadialTangential8Params, calculate_project_jacobian_rt8, calculate_projection_vjp_rt8,
    project_rt8,
};
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec2, Vec3A};

#[derive(Copy, Clone, PartialEq, Debug, Hash, Default)]
pub enum CameraModel {
    #[default]
    Pinhole,
    KannalaBrandt4(KannalaBrandt4Params),
    RadialTangential8(RadialTangential8Params),
}

#[derive(CubeLaunch, CubeType, Debug, Clone, Copy)]
pub struct JacobianClampLimits {
    pub lim_pos_x: f32,
    pub lim_pos_y: f32,
    pub lim_neg_x: f32,
    pub lim_neg_y: f32,
}

#[cube]
pub fn project(
    point: Vec3A,
    pinhole_params: PinholeParams,
    #[comptime] camera_model: CameraModel,
) -> (f32, f32) {
    match camera_model {
        Pinhole => project_pinhole(point, pinhole_params),
        KannalaBrandt4(params) => project_kb4(point, pinhole_params, params),
        RadialTangential8(params) => project_rt8(point, pinhole_params, params),
    }
}

/// Computes the Jacobian of the projection w.r.t. the projected 3d point
#[cube]
pub fn calculate_project_jacobian(
    point: Vec3A,
    jacobian_clamp_limits: JacobianClampLimits,
    pinhole_params: PinholeParams,
    #[comptime] camera_model: CameraModel,
) -> Mat2x3 {
    match camera_model {
        Pinhole => calculate_project_jacobian_pinhole(point, jacobian_clamp_limits, pinhole_params),
        KannalaBrandt4(params) => calculate_project_jacobian_kb4(point, pinhole_params, params),
        RadialTangential8(params) => {
            calculate_project_jacobian_rt8(point, jacobian_clamp_limits, pinhole_params, params)
        }
    }
}

/// VJP of the projection. Returns gradient w.r.t.
/// `mean3d` given grads w.r.t. cov2d (`v_cov2d`) and mean2d (`v_mean2d`).
/// `cov_c` is the 3D covariance in camera space.
#[allow(clippy::too_many_arguments)]
#[cube]
pub fn calculate_projection_vjp(
    projection_jacobian: Mat2x3,
    mean_c: Vec3A,
    cov_c: Sym3,
    u: ProjectUniforms,
    v_cov2d: Sym2,
    v_mean2d: Vec2,
    #[comptime] camera_model: CameraModel,
) -> Vec3A {
    match camera_model {
        Pinhole => calculate_projection_vjp_pinhole(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d,
        ),
        KannalaBrandt4(params) => calculate_projection_vjp_kb4(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d,
            params,
        ),
        RadialTangential8(params) => {
            calculate_projection_vjp_rt8(mean_c, cov_c, u, v_cov2d, v_mean2d, params)
        }
    }
}

impl JacobianClampLimits {
    pub fn to_launch_object<R: Runtime>(&self) -> JacobianClampLimitsLaunch<R> {
        JacobianClampLimitsLaunch::new(
            self.lim_pos_x,
            self.lim_pos_y,
            self.lim_neg_x,
            self.lim_neg_y,
        )
    }
}
