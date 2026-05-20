use crate::kernels::camera_model::CameraModel::{KannalaBrandt4, Pinhole, RadialTangential8};
use crate::kernels::camera_model::kannala_brandt_4::KannalaBrandt4Params;
use crate::kernels::camera_model::pinhole::PinholeParams;
use crate::kernels::camera_model::radial_tangential_8::RadialTangential8Params;
use crate::kernels::camera_model::{CameraModel, JacobianClampLimits};
use glam::Affine3A;
use std::f64::consts::PI;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Camera {
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: glam::Vec2,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub camera_model: CameraModel,
}

impl Camera {
    pub fn new(
        position: glam::Vec3,
        rotation: glam::Quat,
        fov_x: f64,
        fov_y: f64,
        center_uv: glam::Vec2,
    ) -> Self {
        Self {
            fov_x,
            fov_y,
            center_uv,
            position,
            rotation,
            camera_model: Pinhole,
        }
    }

    pub fn new_with_calibration(
        position: glam::Vec3,
        rotation: glam::Quat,
        fov_x: f64,
        fov_y: f64,
        center_uv: glam::Vec2,
        camera_model: CameraModel,
    ) -> Self {
        Self {
            fov_x,
            fov_y,
            center_uv,
            position,
            rotation,
            camera_model,
        }
    }

    /// Check if the camera has valid (non-nan/inf) settings.
    pub fn is_valid(&self) -> bool {
        self.fov_x.is_finite()
            && self.fov_y.is_finite()
            && self.center_uv.is_finite()
            && self.position.is_finite()
            && self.rotation.is_finite()
    }

    pub fn focal(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            fov_to_focal(self.fov_x, img_size.x, &self.camera_model) as f32,
            fov_to_focal(self.fov_y, img_size.y, &self.camera_model) as f32,
        )
    }

    pub fn center(&self, img_size: glam::UVec2) -> glam::Vec2 {
        glam::vec2(
            self.center_uv.x * img_size.x as f32,
            self.center_uv.y * img_size.y as f32,
        )
    }

    pub fn build_pinhole_params(&self, img_size: glam::UVec2) -> PinholeParams {
        let focal = self.focal(img_size);
        let pixel_center = self.center(img_size);

        PinholeParams {
            fx: focal.x,
            fy: focal.y,
            cx: pixel_center.x,
            cy: pixel_center.y,
        }
    }

    pub fn local_to_world(&self) -> Affine3A {
        Affine3A::from_rotation_translation(self.rotation, self.position)
    }

    pub fn world_to_local(&self) -> Affine3A {
        self.local_to_world().inverse()
    }
}

// Converts field of view to focal length
pub fn fov_to_focal(fov: f64, pixels: u32, model: &CameraModel) -> f64 {
    let half_fov = fov / 2.0;
    let r_pix = (pixels as f64) / 2.0;

    // We want focal f such that r_pix = f · projection(half_fov).
    let projected = match model {
        Pinhole => half_fov.tan(),
        KannalaBrandt4(p) => kb4_d(half_fov, p),
        RadialTangential8(p) => {
            let r = half_fov.tan();
            r * rt8_radial(r, p)
        }
    };

    r_pix / projected
}

// Converts focal length to field of view
pub fn focal_to_fov(focal: f64, pixels: u32, model: &CameraModel) -> f64 {
    let r_pix = (pixels as f64) / 2.0;
    let r_norm = r_pix / focal; // distorted normalized radius (= d(θ) for KB4)

    let half_fov = match model {
        Pinhole => r_norm.atan(),
        KannalaBrandt4(p) => kb4_invert_d(r_norm, p),
        RadialTangential8(p) => {
            let r_undist = rt8_undistort_radius(r_norm, p);
            r_undist.atan()
        }
    };

    2.0 * half_fov
}

// KB4 distortion polynomial: d(θ) = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹
#[inline]
fn kb4_d(theta: f64, p: &KannalaBrandt4Params) -> f64 {
    let t2 = theta * theta;
    let t3 = t2 * theta;
    let t5 = t3 * t2;
    let t7 = t5 * t2;
    let t9 = t7 * t2;
    theta + p.k1 as f64 * t3 + p.k2 as f64 * t5 + p.k3 as f64 * t7 + p.k4 as f64 * t9
}

// d'(θ) = 1 + 3k1·θ² + 5k2·θ⁴ + 7k3·θ⁶ + 9k4·θ⁸
#[inline]
fn kb4_dd_dtheta(theta: f64, p: &KannalaBrandt4Params) -> f64 {
    let t2 = theta * theta;
    let t4 = t2 * t2;
    let t6 = t4 * t2;
    let t8 = t6 * t2;
    1.0 + 3.0 * p.k1 as f64 * t2
        + 5.0 * p.k2 as f64 * t4
        + 7.0 * p.k3 as f64 * t6
        + 9.0 * p.k4 as f64 * t8
}

// Solve d(θ) = target for θ ∈ [0, π] via Newton with bisection fallback.
fn kb4_invert_d(target: f64, p: &KannalaBrandt4Params) -> f64 {
    if target <= 0.0 {
        return 0.0;
    }
    let mut theta = target.min(PI - 1e-6);

    // Newton iterations
    for _ in 0..50 {
        let f = kb4_d(theta, p) - target;
        let fp = kb4_dd_dtheta(theta, p);
        if fp.abs() < 1e-12 {
            break;
        }
        let step = f / fp;
        let next = (theta - step).clamp(0.0, PI);
        if (next - theta).abs() < 1e-12 {
            theta = next;
            break;
        }
        theta = next;
    }
    theta
}

// Radial distortion factor for RadTan8: (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶)
#[inline]
fn rt8_radial(r: f64, p: &RadialTangential8Params) -> f64 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let num = 1.0 + p.k1 as f64 * r2 + p.k2 as f64 * r4 + p.k3 as f64 * r6;
    let den = 1.0 + p.k4 as f64 * r2 + p.k5 as f64 * r4 + p.k6 as f64 * r6;
    num / den
}

// Given the *distorted* normalized radius r_d, recover the undistorted r
// such that r · radial(r) = r_d. Fixed-point iteration (standard OpenCV approach).
fn rt8_undistort_radius(r_d: f64, p: &RadialTangential8Params) -> f64 {
    let mut r = r_d;
    for _ in 0..30 {
        let factor = rt8_radial(r, p);
        if factor.abs() < 1e-12 {
            break;
        }
        let r_new = r_d / factor;
        if (r_new - r).abs() < 1e-12 {
            r = r_new;
            break;
        }
        r = r_new;
    }
    r
}

pub fn calculate_jacobian_clamp_limits(
    img_size: glam::UVec2,
    pinhole_params: PinholeParams,
    camera_model: CameraModel,
) -> JacobianClampLimits {
    let PinholeParams { fx, fy, cx, cy } = pinhole_params;

    let mut lim_pos_x = 0.;
    let mut lim_neg_x = 0.;
    let mut lim_pos_y = 0.;
    let mut lim_neg_y = 0.;

    let img_w = img_size.x as f32;
    let img_h = img_size.y as f32;

    match camera_model {
        Pinhole => {
            lim_pos_x = (1.15 * img_w - cx) / fx;
            lim_pos_y = (1.15 * img_h - cy) / fy;
            lim_neg_x = (-0.15 * img_w - cx) / fx;
            lim_neg_y = (-0.15 * img_h - cy) / fy;
        }
        RadialTangential8(_) => {
            // TODO: this works for small distortions,
            // the more sophisticated version would require to invert the RT8 model
            lim_pos_x = (1.3 * img_w - cx) / fx;
            lim_pos_y = (1.3 * img_h - cy) / fy;
            lim_neg_x = (-0.3 * img_w - cx) / fx;
            lim_neg_y = (-0.3 * img_h - cy) / fy;
        }
        KannalaBrandt4(_) => {}
    }

    JacobianClampLimits {
        lim_pos_x,
        lim_pos_y,
        lim_neg_x,
        lim_neg_y,
    }
}
