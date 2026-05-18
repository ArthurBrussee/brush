use glam::Affine3A;
use crate::kernels::camera_model::CameraParams;

pub const PINHOLE: i32 = 0;
pub const KANNALA_BRANDT_4: i32 = 1;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Camera {
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: glam::Vec2,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub distortion: [f32; 4],
    pub camera_model_id: i32,
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
            distortion: [0.; 4],
            camera_model_id: PINHOLE,
        }
    }

    pub fn new_with_distortion(
        position: glam::Vec3,
        rotation: glam::Quat,
        fov_x: f64,
        fov_y: f64,
        center_uv: glam::Vec2,
        distortion: [f32; 4],
        camera_model_id: i32,
    ) -> Self {
        Self {
            fov_x,
            fov_y,
            center_uv,
            position,
            rotation,
            distortion,
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
            k1: self.distortion[0],
            k2: self.distortion[1],
            k3: self.distortion[2],
            k4: self.distortion[3],
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
