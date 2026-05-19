use crate::kernels::camera_model::CameraModel::{KannalaBrandt4, Pinhole, RadialTangential8};
use crate::kernels::camera_model::pinhole::PinholeParams;
use crate::kernels::camera_model::{CameraModel, JacobianClampLimits};
use glam::Affine3A;

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

    pub fn new_with_distortion(
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
            fov_to_focal(self.fov_x, img_size.x) as f32,
            fov_to_focal(self.fov_y, img_size.y) as f32,
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
pub fn fov_to_focal(fov_rad: f64, pixels: u32) -> f64 {
    0.5 * (pixels as f64) / (fov_rad * 0.5).tan()
}

// Converts focal length to field of view
pub fn focal_to_fov(focal: f64, pixels: u32) -> f64 {
    2.0 * f64::atan((pixels as f64) / (2.0 * focal))
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
            // TODO find a more sophisticated way to find good Jacobian clamping limits for RT8
            let fov_x = 2.0 * (img_w / (2.0 * fx)).atan();
            let fov_y = 2.0 * (img_h / (2.0 * fy)).atan();
            lim_pos_x = 1.15 * (fov_x / 2.0).tan();
            lim_neg_x = -lim_pos_x;
            lim_pos_y = 1.15 * (fov_y / 2.0).tan();
            lim_neg_y = -lim_pos_y;
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
