pub mod kannala_brandt_4;
pub mod pinhole;
pub mod radial_tangential_8;
pub mod thin_prism_fisheye;

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

use crate::kernels::camera_model::CameraModel::{
    KannalaBrandt4, Pinhole, RadialTangential8, ThinPrismFisheye,
};
use crate::kernels::camera_model::kannala_brandt_4::{
    KannalaBrandt4Params, calculate_project_jacobian_kb4, calculate_projection_vjp_kb4, project_kb4,
};
use crate::kernels::camera_model::pinhole::{
    calculate_project_jacobian_pinhole, calculate_projection_vjp_pinhole, project_pinhole,
};
use crate::kernels::camera_model::radial_tangential_8::{
    RadialTangential8Params, calculate_project_jacobian_rt8, calculate_projection_vjp_rt8,
    project_rt8,
};
use crate::kernels::camera_model::thin_prism_fisheye::{
    ThinPrismFisheyeParams, calculate_project_jacobian_tpf, calculate_projection_vjp_tpf,
    project_tpf,
};
use crate::kernels::types::ProjectUniforms;
use brush_cube::{Mat2x3, Sym2, Sym3, Vec2, Vec3A};

#[derive(Copy, Clone, PartialEq, Debug, Default)]
pub enum CameraModel {
    #[default]
    Pinhole,
    KannalaBrandt4(KannalaBrandt4Params),
    RadialTangential8(RadialTangential8Params),
    ThinPrismFisheye(ThinPrismFisheyeParams),
}

/// Comptime kernel discriminant for the camera model. The actual k-params
/// are passed at runtime via `ProjectUniforms`, so a kernel only specializes
/// on the model identity, not on the parameter values, preventing blowup if
/// images all have their own calibration.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, Default)]
pub enum CameraKind {
    #[default]
    Pinhole,
    KannalaBrandt4,
    RadialTangential8,
    ThinPrismFisheye,
}

impl CameraModel {
    pub fn kind(&self) -> CameraKind {
        match self {
            Pinhole => CameraKind::Pinhole,
            KannalaBrandt4(_) => CameraKind::KannalaBrandt4,
            RadialTangential8(_) => CameraKind::RadialTangential8,
            ThinPrismFisheye(_) => CameraKind::ThinPrismFisheye,
        }
    }

    pub fn kb4_params(&self) -> KannalaBrandt4Params {
        match self {
            KannalaBrandt4(p) => *p,
            // ThinPrismFisheye carries its own KB4 sub-block; expose it so
            // the FOV helpers can use the radial part directly.
            ThinPrismFisheye(p) => p.kb4,
            _ => KannalaBrandt4Params::default(),
        }
    }

    pub fn rt8_params(&self) -> RadialTangential8Params {
        match self {
            RadialTangential8(p) => *p,
            _ => RadialTangential8Params::default(),
        }
    }

    pub fn tpf_params(&self) -> ThinPrismFisheyeParams {
        match self {
            ThinPrismFisheye(p) => *p,
            _ => ThinPrismFisheyeParams::default(),
        }
    }

    /// Short human-readable name for UI display.
    pub fn label(&self) -> &'static str {
        match self {
            Pinhole => "Pinhole",
            KannalaBrandt4(_) => "Kannala-Brandt 4 (fisheye)",
            RadialTangential8(_) => "Radial-tangential 8 (OpenCV)",
            ThinPrismFisheye(_) => "Thin-prism fisheye",
        }
    }
}

#[derive(CubeLaunch, CubeType, Debug, Clone, Copy)]
pub struct JacobianClampLimits {
    pub lim_pos_x: f32,
    pub lim_pos_y: f32,
    pub lim_neg_x: f32,
    pub lim_neg_y: f32,
}

#[cube]
pub fn project(point: Vec3A, u: ProjectUniforms, #[comptime] kind: CameraKind) -> (f32, f32) {
    match kind {
        CameraKind::Pinhole => project_pinhole(point, u.pinhole_params),
        CameraKind::KannalaBrandt4 => project_kb4(point, u.pinhole_params, u.kb4_params),
        CameraKind::RadialTangential8 => project_rt8(point, u.pinhole_params, u.rt8_params),
        CameraKind::ThinPrismFisheye => project_tpf(point, u.pinhole_params, u.tpf_params),
    }
}

/// Computes the Jacobian of the projection w.r.t. the projected 3d point
#[cube]
pub fn calculate_project_jacobian(
    point: Vec3A,
    u: ProjectUniforms,
    #[comptime] kind: CameraKind,
) -> Mat2x3 {
    match kind {
        CameraKind::Pinhole => {
            calculate_project_jacobian_pinhole(point, u.jacobian_clamp_limits, u.pinhole_params)
        }
        CameraKind::KannalaBrandt4 => {
            calculate_project_jacobian_kb4(point, u.pinhole_params, u.kb4_params)
        }
        CameraKind::RadialTangential8 => calculate_project_jacobian_rt8(
            point,
            u.jacobian_clamp_limits,
            u.pinhole_params,
            u.rt8_params,
        ),
        CameraKind::ThinPrismFisheye => {
            calculate_project_jacobian_tpf(point, u.pinhole_params, u.tpf_params)
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
    #[comptime] kind: CameraKind,
) -> Vec3A {
    match kind {
        CameraKind::Pinhole => calculate_projection_vjp_pinhole(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d,
        ),
        CameraKind::KannalaBrandt4 => calculate_projection_vjp_kb4(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d,
            u.kb4_params,
        ),
        CameraKind::RadialTangential8 => {
            calculate_projection_vjp_rt8(mean_c, cov_c, u, v_cov2d, v_mean2d, u.rt8_params)
        }
        CameraKind::ThinPrismFisheye => calculate_projection_vjp_tpf(
            projection_jacobian,
            mean_c,
            cov_c,
            u,
            v_cov2d,
            v_mean2d,
            u.tpf_params,
        ),
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
