//! GOF's per-Gaussian 3D filter (`filter_3D`): the radius below which a
//! Gaussian's contribution would alias against the pixel grid at its
//! closest training camera.
//!
//! For each Gaussian `g` we find the smallest view-space depth across all
//! cameras that actually see it, and set
//!
//! ```text
//! filter_3D_g = min_depth_g / max_focal · √0.2
//! ```
//!
//! Downstream code treats `filter_3D_g` as an isotropic minimum 3D
//! half-extent: `scale_eff² = scale² + filter_3D²` per axis, and applies
//! the opacity compensation `o_eff = o · √(det(scale²) / det(scale_eff²))`
//! so the *integral* of the inflated Gaussian matches the raw one.
//!
//! Mirrors `scene/gaussian_model.py::compute_3D_filter` in the reference.

use brush_render::camera::Camera;
use glam::{UVec2, Vec3, Vec4Swizzles};
use rayon::prelude::*;

/// GOF's `0.2` constant — chosen so the floor is ~√0.2 ≈ 0.45 pixel in
/// projected size at the closest camera, well below the pixel-grid
/// Nyquist limit.
const SQRT_K: f32 = 0.447_213_6; // sqrt(0.2)

/// Frustum margin matches the rest of the pipeline (15% padding).
const FRUSTUM_MARGIN: f32 = 0.15;

/// GOF hardcodes the near-plane filter at 0.2 in `compute_3D_filter`
/// (separate from the configurable near used for seed-point culling).
const FILTER_NEAR: f32 = 0.2;

/// Compute `filter_3D` for every Gaussian in `means`. Returns a
/// `[n_gaussians]` flat array. The two non-trivial choices follow GOF:
/// `max_focal` across all cameras (conservative pixel size); Gaussians
/// invisible from every camera receive the *maximum* finite filter (so
/// they don't fall off into the unbounded "100000" sentinel GOF starts
/// with).
pub fn compute_filter_3d(means: &[f32], cameras: &[Camera], image_sizes: &[UVec2]) -> Vec<f32> {
    assert_eq!(cameras.len(), image_sizes.len());
    let n = means.len() / 3;

    let max_focal = cameras
        .iter()
        .zip(image_sizes.iter())
        .map(|(cam, sz)| {
            let p = cam.build_pinhole_params(*sz);
            p.fx.max(p.fy)
        })
        .fold(0.0_f32, f32::max);

    // Precompute per-camera world-to-view affine + pinhole params + size.
    struct Cam {
        w2c: glam::Mat4,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
    }
    let cams: Vec<Cam> = cameras
        .iter()
        .zip(image_sizes.iter())
        .map(|(cam, sz)| {
            let p = cam.build_pinhole_params(*sz);
            Cam {
                w2c: glam::Mat4::from(cam.world_to_local()),
                fx: p.fx,
                fy: p.fy,
                cx: p.cx,
                cy: p.cy,
                w: sz.x as f32,
                h: sz.y as f32,
            }
        })
        .collect();

    let infinity_marker = f32::INFINITY;
    let dists: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let p = Vec3::new(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
            let mut min_z = infinity_marker;
            for c in &cams {
                let p_cam = (c.w2c * p.extend(1.0)).xyz();
                let z = p_cam.z;
                if !(z > FILTER_NEAR) {
                    continue;
                }
                let px = p_cam.x / z * c.fx + c.cx;
                let py = p_cam.y / z * c.fy + c.cy;
                let m = FRUSTUM_MARGIN;
                if px < -m * c.w || px > (1.0 + m) * c.w || py < -m * c.h || py > (1.0 + m) * c.h {
                    continue;
                }
                if z < min_z {
                    min_z = z;
                }
            }
            min_z
        })
        .collect();

    // Gaussians invisible from every camera: assign the maximum finite
    // distance we did see, matching GOF's `distance[~valid_points] =
    // distance[valid_points].max()`. Skip if no Gaussian was ever visible
    // (degenerate; returns zeros which disables the filter).
    let max_finite = dists
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    if !max_finite.is_finite() {
        return vec![0.0; n];
    }
    if max_focal <= 0.0 {
        return vec![0.0; n];
    }

    dists
        .into_iter()
        .map(|d| {
            let d = if d.is_finite() { d } else { max_finite };
            d / max_focal * SQRT_K
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;

    fn fake_cam(pos: Vec3) -> (Camera, UVec2) {
        let cam = Camera::new(
            pos,
            glam::Quat::IDENTITY,
            std::f64::consts::PI * 0.5,
            std::f64::consts::PI * 0.5,
            glam::Vec2::new(0.5, 0.5),
            CameraModel::default(),
        );
        (cam, UVec2::new(256, 256))
    }

    /// A Gaussian visible from a single nearby camera gets a small
    /// filter; a Gaussian at twice the distance gets twice the filter.
    #[test]
    fn scales_with_camera_distance() {
        let (cam, sz) = fake_cam(Vec3::ZERO);
        let means = vec![
            0.0, 0.0, 1.0, // 1m from cam
            0.0, 0.0, 2.0, // 2m from cam
        ];
        let f3d = compute_filter_3d(&means, &[cam], &[sz]);
        assert!((f3d[1] - 2.0 * f3d[0]).abs() < 1e-6);
        assert!(f3d[0] > 0.0);
    }

    /// Multi-camera: each Gaussian picks its *closest* camera's distance.
    #[test]
    fn picks_closest_camera() {
        let (near_cam, sz) = fake_cam(Vec3::new(0.0, 0.0, -0.5));
        let (far_cam, _) = fake_cam(Vec3::new(0.0, 0.0, -10.0));
        let means = vec![0.0, 0.0, 0.0];
        let f3d_near = compute_filter_3d(&means, &[near_cam], &[sz]);
        let f3d_both = compute_filter_3d(&means, &[near_cam, far_cam], &[sz, sz]);
        // Adding a farther camera shouldn't change the minimum-distance
        // pick. The max-focal stays the same too (both cams identical
        // focal). So filter_3D should be unchanged.
        assert!((f3d_near[0] - f3d_both[0]).abs() < 1e-6);
    }
}
