//! Per-Gaussian seed-point sampling for the tetrahedralization.
//!
//! For each Gaussian we emit the 8 corners of its `±3σ` oriented bounding box
//! plus the Gaussian center — 9 points per Gaussian. Each point's "scale" is
//! the parent Gaussian's max axis length, used downstream by
//! [`crate::filter::filter_mesh`].
//!
//! Points are frustum-culled against the training cameras: a point is kept
//! iff it projects inside the image (with a 15% margin, matching GOF's
//! `get_frustum_mask`) for at least one view, and its depth lies in
//! `(near, far)`.
//!
//! Matches `scene/gaussian_model.py::get_tetra_points` from the reference repo.

use brush_render::camera::Camera;
use glam::{Vec3, Vec4Swizzles};
use rayon::prelude::*;

/// 8 unit-cube corners ∈ `{±1}^3`. Multiplied by the Gaussian's scale below.
const BOX_CORNERS: [[f32; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

/// GOF uses 3σ — the bounding box is a hard cap on Gaussian support since
/// `exp(-0.5 · 3²) ≈ 0.011 ≤ 1/255`.
pub const SIGMA_SCALE: f32 = 3.0;

#[derive(Debug, Clone)]
pub struct TetraPointsConfig {
    pub near: f32,
    /// Far plane of the per-camera seed frustums: the meshed region is the
    /// union of all camera frustums truncated at this distance (metres on
    /// metric scenes). Keeps the mesh to the part of the scene the capture
    /// actually orbits instead of every far-field splat.
    pub far: f32,
    /// Shrink each Gaussian's seed box to its actual support radius
    /// `sigma * sqrt(2 ln(255 * opacity))` (capped at 3 sigma) instead of a
    /// fixed 3 sigma; Gaussians below the 1/255 cutoff emit no seeds at all.
    /// Translucent splats stop seeding crossings in space they barely occupy.
    pub opacity_radius: bool,
    /// Octahedron seeds (the 6 `+-r` axis points, on the support boundary)
    /// instead of GOF's 8 box corners (at `sqrt(3) r`, off it): 22% fewer
    /// seed points for PSNR-neutral quality.
    pub octahedron: bool,
    /// Frustum margin as a fraction of the image size. GOF uses 0 (strict
    /// image bounds) for `get_frustum_mask`, which trims any seed point
    /// whose projection falls outside the rendered image. A small positive
    /// value keeps boundary points that would otherwise create gaps along
    /// the image edges. Default 0 matches GOF.
    pub frustum_margin: f32,
}

impl Default for TetraPointsConfig {
    fn default() -> Self {
        Self {
            // Matches the integrate kernels' NEAR_PLANE: seed points the
            // integration can't see are pure Delaunay load.
            near: 0.2,
            far: 4.0,
            frustum_margin: 0.0,
            opacity_radius: false,
            octahedron: true,
        }
    }
}

pub struct TetraPoints {
    pub points: Vec<Vec3>,
    pub scales: Vec<f32>,
}

/// Build the seed-point set for `means / quats / log_scales`. Input
/// arrays are flat `[N*3]`, `[N*4]` (wxyz), `[N*3]`. `cameras` /
/// `image_sizes` drive frustum culling.
///
/// The caller is expected to pass *baked* scales — i.e. the mip 3D
/// filter floor already folded in via [`Splats::bake_min_scale`]
/// (which `extract_mesh` does up front). No per-Gaussian inflation
/// happens here, which keeps the seed sampler in sync with the
/// integrate kernel that also reads baked transforms.
pub fn build_tetra_points(
    means: &[f32],
    quats_wxyz: &[f32],
    log_scales: &[f32],
    opacities: &[f32],
    cameras: &[Camera],
    image_sizes: &[glam::UVec2],
    cfg: &TetraPointsConfig,
) -> TetraPoints {
    assert_eq!(
        cameras.len(),
        image_sizes.len(),
        "one image size per camera"
    );
    let n = means.len() / 3;
    assert_eq!(quats_wxyz.len(), n * 4, "flat [N*4] wxyz quats");
    assert_eq!(log_scales.len(), n * 3, "flat [N*3] log scales");
    assert_eq!(opacities.len(), n, "one opacity per gaussian");

    // 9 points per Gaussian, parallel over Gaussians. The frustum cull is
    // folded in so we don't materialise a 9× temporary.
    let chunks: Vec<(Vec<Vec3>, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mean = Vec3::new(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
            let q = glam::Quat::from_xyzw(
                quats_wxyz[4 * i + 1],
                quats_wxyz[4 * i + 2],
                quats_wxyz[4 * i + 3],
                quats_wxyz[4 * i],
            )
            .normalize();
            let scale = Vec3::new(
                log_scales[3 * i].exp(),
                log_scales[3 * i + 1].exp(),
                log_scales[3 * i + 2].exp(),
            );
            // Seed radius in sigmas: GOF's fixed 3, or the opacity-aware
            // support radius where contribution drops below the 1/255
            // render cutoff.
            let r_sigma = if cfg.opacity_radius {
                let o = opacities[i].max(0.0);
                if o * 255.0 <= 1.0 {
                    // Never contributes above the cutoff: no seeds.
                    return (Vec::new(), Vec::new());
                }
                (2.0 * (o * 255.0).ln()).sqrt().min(SIGMA_SCALE)
            } else {
                SIGMA_SCALE
            };
            // Stored per-point scale: `r · max_axis_effective`, the half-
            // extent along the widest axis; matches GOF's `vertices_scale`
            // so the filter rule `edge_len > scale_a + scale_b` lines up.
            let s_max = scale.max_element() * r_sigma;

            let mut pts: Vec<Vec3> = Vec::with_capacity(15);
            let mut scs: Vec<f32> = Vec::with_capacity(15);
            let mut push = |world: Vec3| {
                if point_in_any_frustum(world, cameras, image_sizes, cfg) {
                    pts.push(world);
                    scs.push(s_max);
                }
            };
            if cfg.octahedron {
                for axis in 0..3 {
                    let mut d = Vec3::ZERO;
                    d[axis] = scale[axis] * r_sigma;
                    push(mean + q * d);
                    push(mean - q * d);
                }
            } else {
                for c in &BOX_CORNERS {
                    push(mean + q * (Vec3::new(c[0], c[1], c[2]) * scale * r_sigma));
                }
            }
            push(mean);
            (pts, scs)
        })
        .collect();

    let mut points = Vec::with_capacity(n * 9);
    let mut scales = Vec::with_capacity(n * 9);
    for (mut p, mut s) in chunks {
        points.append(&mut p);
        scales.append(&mut s);
    }
    TetraPoints { points, scales }
}

fn point_in_any_frustum(
    p: Vec3,
    cameras: &[Camera],
    image_sizes: &[glam::UVec2],
    cfg: &TetraPointsConfig,
) -> bool {
    for (cam, sz) in cameras.iter().zip(image_sizes.iter()) {
        let w2c = glam::Mat4::from(cam.world_to_local());
        let p_cam = (w2c * p.extend(1.0)).xyz();
        let z = p_cam.z;
        if !(z > cfg.near && z < cfg.far) {
            continue;
        }
        let pinhole = cam.build_pinhole_params(*sz);
        let px = p_cam.x / z * pinhole.fx + pinhole.cx;
        let py = p_cam.y / z * pinhole.fy + pinhole.cy;
        let w = sz.x as f32;
        let h = sz.y as f32;
        let m = cfg.frustum_margin;
        if px >= -m * w && px <= (1.0 + m) * w && py >= -m * h && py <= (1.0 + m) * h {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use brush_render::kernels::camera_model::CameraModel;

    fn fake_cam(pos: Vec3) -> (Camera, glam::UVec2) {
        let cam = Camera::new(
            pos,
            glam::Quat::IDENTITY,
            std::f64::consts::PI * 0.5,
            std::f64::consts::PI * 0.5,
            glam::Vec2::new(0.5, 0.5),
            CameraModel::default(),
        );
        (cam, glam::UVec2::new(256, 256))
    }

    #[test]
    fn emits_expected_seeds_per_gaussian_when_visible() {
        // Within the default 4 m frustum truncation.
        let (cam, sz) = fake_cam(Vec3::new(0.0, 0.0, -3.0));
        let means = vec![0.0, 0.0, 0.0];
        let quats = vec![1.0, 0.0, 0.0, 0.0];
        let log_scales = vec![-2.0; 3];
        // Octahedron default: 6 axis points + center.
        let out = build_tetra_points(
            &means,
            &quats,
            &log_scales,
            &[0.9],
            &[cam],
            &[sz],
            &TetraPointsConfig::default(),
        );
        assert_eq!(out.points.len(), 7, "octahedron: 6 axis points + center");
        let s = (-2.0f32).exp() * SIGMA_SCALE;
        for sc in &out.scales {
            assert!((sc - s).abs() < 1e-6, "stored scale is 3 sigma max axis");
        }
        // Box corners: 8 + center.
        let cfg = TetraPointsConfig {
            octahedron: false,
            ..Default::default()
        };
        let out = build_tetra_points(&means, &quats, &log_scales, &[0.9], &[cam], &[sz], &cfg);
        assert_eq!(out.points.len(), 9, "box: 8 corners + center");
    }

    #[test]
    fn cull_drops_far_points() {
        let (cam, sz) = fake_cam(Vec3::new(0.0, 0.0, 0.0));
        let means = vec![0.0, 0.0, 1.0e10];
        let quats = vec![1.0, 0.0, 0.0, 0.0];
        let log_scales = vec![-2.0; 3];
        let cfg = TetraPointsConfig {
            far: 100.0,
            ..Default::default()
        };
        let out = build_tetra_points(&means, &quats, &log_scales, &[0.9], &[cam], &[sz], &cfg);
        assert!(out.points.is_empty());
    }
}
