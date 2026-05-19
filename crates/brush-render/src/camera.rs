use crate::kernels::camera_model::{CameraParams, JacobianClampLimits};
use glam::Affine3A;

pub type CameraModelId = i32;

pub const PINHOLE: CameraModelId = 0;
pub const KANNALA_BRANDT_4: CameraModelId = 1;
pub const RADIAL_TANGENTIAL_8: CameraModelId = 2;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Camera {
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: glam::Vec2,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub params: [f32; 8],
    pub camera_model_id: CameraModelId,
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
            params: [0.; 8],
            camera_model_id: PINHOLE,
        }
    }

    pub fn new_with_distortion(
        position: glam::Vec3,
        rotation: glam::Quat,
        fov_x: f64,
        fov_y: f64,
        center_uv: glam::Vec2,
        params: [f32; 8],
        camera_model_id: CameraModelId,
    ) -> Self {
        Self {
            fov_x,
            fov_y,
            center_uv,
            position,
            rotation,
            params,
            camera_model_id,
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

    pub fn to_params(&self, img_size: glam::UVec2) -> CameraParams {
        let focal = self.focal(img_size);
        let pixel_center = self.center(img_size);

        CameraParams {
            focal_x: focal.x,
            focal_y: focal.y,
            pixel_center_x: pixel_center.x,
            pixel_center_y: pixel_center.y,
            param0: self.params[0],
            param1: self.params[1],
            param2: self.params[2],
            param3: self.params[3],
            param4: self.params[4],
            param5: self.params[5],
            param6: self.params[6],
            param7: self.params[7],
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
    camera_params: CameraParams,
    camera_model_id: CameraModelId,
) -> JacobianClampLimits {
    let mut lim_pos_x = 0.;
    let mut lim_neg_x = 0.;
    let mut lim_pos_y = 0.;
    let mut lim_neg_y = 0.;

    let img_w = img_size.x as f32;
    let img_h = img_size.y as f32;

    if camera_model_id == PINHOLE {
        lim_pos_x = (1.15 * img_w - camera_params.pixel_center_x) / camera_params.focal_x;
        lim_pos_y = (1.15 * img_h - camera_params.pixel_center_y) / camera_params.focal_y;
        lim_neg_x = (-0.15 * img_w - camera_params.pixel_center_x) / camera_params.focal_x;
        lim_neg_y = (-0.15 * img_h - camera_params.pixel_center_y) / camera_params.focal_y;
    } else if camera_model_id == RADIAL_TANGENTIAL_8 {
        let fov_x = 2.0 * (img_w / (2.0 * camera_params.focal_x)).atan();
        let fov_y = 2.0 * (img_h / (2.0 * camera_params.focal_y)).atan();
        lim_pos_x = 1.15 * (fov_x / 2.0).tan();
        lim_neg_x = -lim_pos_x;
        lim_pos_y = 1.15 * (fov_y / 2.0).tan();
        lim_neg_y = -lim_pos_y;
    }

    JacobianClampLimits {
        lim_pos_x,
        lim_pos_y,
        lim_neg_x,
        lim_neg_y,
    }
}
