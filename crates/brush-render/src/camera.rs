use glam::Affine3A;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Camera {
    pub fov_x: f64,
    pub fov_y: f64,
    pub center_uv: glam::Vec2,
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
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
