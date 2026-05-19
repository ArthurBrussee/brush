pub mod kannala_brandt_4;
pub mod pinhole;
pub mod radial_tangential_8;

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

use crate::camera::{CameraModelId, KANNALA_BRANDT_4, PINHOLE, RADIAL_TANGENTIAL_8};
use crate::kernels::camera_model::kannala_brandt_4::{
    calculate_project_jacobian_kb4, calculate_projection_vjp_kb4, project_kb4,
};
use crate::kernels::camera_model::pinhole::{
    calculate_project_jacobian_pinhole, calculate_projection_vjp_pinhole, project_pinhole,
};
use crate::kernels::camera_model::radial_tangential_8::{
    calculate_project_jacobian_rt8, calculate_projection_vjp_rt8, project_rt8,
};
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec3A};

#[derive(CubeLaunch, CubeType, Debug, Clone, Copy)]
pub struct CameraParams {
    pub focal_x: f32,
    pub focal_y: f32,
    pub pixel_center_x: f32,
    pub pixel_center_y: f32,
    pub param0: f32,
    pub param1: f32,
    pub param2: f32,
    pub param3: f32,
    pub param4: f32,
    pub param5: f32,
    pub param6: f32,
    pub param7: f32,
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
    camera_params: CameraParams,
    #[comptime] camera_model_id: CameraModelId,
) -> (f32, f32) {
    if comptime![camera_model_id == PINHOLE] {
        project_pinhole(point, camera_params)
    } else if comptime![camera_model_id == KANNALA_BRANDT_4] {
        project_kb4(point, camera_params)
    } else if comptime![camera_model_id == RADIAL_TANGENTIAL_8] {
        project_rt8(point, camera_params)
    } else {
        panic!("not implemented")
    }
}

/// Computes the Jacobian of the projection w.r.t. the projected 3d point
#[cube]
pub fn calculate_project_jacobian(
    point: Vec3A,
    jacobian_clamp_limits: JacobianClampLimits,
    camera_params: CameraParams,
    #[comptime] camera_model_id: CameraModelId,
) -> Mat2x3 {
    if comptime![camera_model_id == PINHOLE] {
        calculate_project_jacobian_pinhole(point, jacobian_clamp_limits, camera_params)
    } else if comptime![camera_model_id == KANNALA_BRANDT_4] {
        calculate_project_jacobian_kb4(point, jacobian_clamp_limits, camera_params)
    } else if comptime![camera_model_id == RADIAL_TANGENTIAL_8] {
        calculate_project_jacobian_rt8(point, jacobian_clamp_limits, camera_params)
    } else {
        panic!("not implemented")
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
    v_mean2d_x: f32,
    v_mean2d_y: f32,
    #[comptime] camera_model_id: CameraModelId,
) -> Vec3A {
    if comptime![camera_model_id == PINHOLE] {
        calculate_projection_vjp_pinhole(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d_x,
            v_mean2d_y,
        )
    } else if comptime![camera_model_id == KANNALA_BRANDT_4] {
        calculate_projection_vjp_kb4(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d_x,
            v_mean2d_y,
        )
    } else if comptime![camera_model_id == RADIAL_TANGENTIAL_8] {
        calculate_projection_vjp_rt8(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d_x,
            v_mean2d_y,
        )
    } else {
        panic!("not implemented")
    }
}

impl CameraParams {
    pub fn to_launch_object<R: Runtime>(&self) -> CameraParamsLaunch<R> {
        CameraParamsLaunch::new(
            self.focal_x,
            self.focal_y,
            self.pixel_center_x,
            self.pixel_center_y,
            self.param0,
            self.param1,
            self.param2,
            self.param3,
            self.param4,
            self.param5,
            self.param6,
            self.param7,
        )
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
