#[derive(Clone, Copy)]
pub struct BoundingBox {
    pub center: glam::Vec3,
    pub extent: glam::Vec3,
}

impl BoundingBox {
    pub fn from_min_max(min: glam::Vec3, max: glam::Vec3) -> Self {
        Self {
            center: (max + min) / 2.0,
            extent: (max - min) / 2.0,
        }
    }

    pub fn min(&self) -> glam::Vec3 {
        self.center - self.extent
    }

    pub fn max(&self) -> glam::Vec3 {
        self.center + self.extent
    }
}
