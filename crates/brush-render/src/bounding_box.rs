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

    pub fn median_size(&self) -> f32 {
        // `total_cmp` is NaN-safe — `partial_cmp(...).unwrap()` used to
        // panic when one extent went NaN mid-training.
        let mut extents = [self.extent.x, self.extent.y, self.extent.z];
        extents.sort_by(|a, b| a.total_cmp(b));
        extents[1] * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_size_with_nan_does_not_panic() {
        let bb = BoundingBox {
            center: glam::Vec3::ZERO,
            extent: glam::Vec3::new(f32::NAN, 2.0, 3.0),
        };
        assert!(bb.median_size().is_finite());
    }

    #[test]
    fn median_size_with_all_nan() {
        let bb = BoundingBox {
            center: glam::Vec3::ZERO,
            extent: glam::Vec3::splat(f32::NAN),
        };
        let _ = bb.median_size();
    }

    #[test]
    fn median_size_normal() {
        let bb = BoundingBox::from_min_max(glam::Vec3::splat(-1.0), glam::Vec3::new(1.0, 3.0, 5.0));
        assert!((bb.median_size() - 4.0).abs() < 1e-6);
    }
}
