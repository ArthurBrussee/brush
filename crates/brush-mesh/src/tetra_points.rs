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
    pub far: f32,
    /// Frustum margin as a fraction of the image size. GOF uses 0 (strict
    /// image bounds) for `get_frustum_mask`, which trims any seed point
    /// whose projection falls outside the rendered image. A small positive
    /// value keeps boundary points that would otherwise create gaps along
    /// the image edges. Default 0 matches GOF.
    pub frustum_margin: f32,
    /// Skip Gaussians whose largest axis is at or above this percentile
    /// of the per-Gaussian max-axis distribution (0..1, default 0.999
    /// = top 0.1%). Such outliers are typically "sky" / billboard splats
    /// brush trains for distant appearance — they're not surfaces, and
    /// their huge scales produce mega-tets that defeat the edge-length
    /// filter downstream. Set to 1.0 to disable.
    pub max_axis_pct: f32,
}

impl Default for TetraPointsConfig {
    fn default() -> Self {
        Self {
            near: 0.02,
            far: 1e6,
            frustum_margin: 0.0,
            max_axis_pct: 0.999,
        }
    }
}

pub struct TetraPoints {
    pub points: Vec<Vec3>,
    pub scales: Vec<f32>,
}

/// Build the seed-point set for `means / quats / log_scales`. `filter_3d`
/// is the per-Gaussian isotropic minimum half-extent (see
/// [`crate::filter_3d`]); pass an array of zeros to disable. Input arrays
/// are flat `[N*3]`, `[N*4]` (wxyz), `[N*3]`, `[N]`. `cameras` /
/// `image_sizes` drive frustum culling.
pub fn build_tetra_points(
    means: &[f32],
    quats_wxyz: &[f32],
    log_scales: &[f32],
    filter_3d: &[f32],
    cameras: &[Camera],
    image_sizes: &[glam::UVec2],
    cfg: &TetraPointsConfig,
) -> TetraPoints {
    assert_eq!(cameras.len(), image_sizes.len());
    let n = means.len() / 3;
    assert_eq!(quats_wxyz.len(), n * 4);
    assert_eq!(log_scales.len(), n * 3);
    assert_eq!(filter_3d.len(), n);

    // Determine the per-Gaussian max-axis cutoff. A brush splat scene
    // typically has a long tail of "billboard" / sky Gaussians whose
    // max-axis is orders of magnitude larger than the median surface
    // splat. They cover huge spatial extents, produce mega-seeds, and
    // hand the edge-length mesh filter a `scale_a + scale_b` budget so
    // large that real spurious mega-tets pass through. Drop them.
    let max_axis_cutoff: f32 = if cfg.max_axis_pct >= 1.0 {
        f32::INFINITY
    } else {
        let mut max_axes: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let sx = log_scales[3 * i].exp();
                let sy = log_scales[3 * i + 1].exp();
                let sz = log_scales[3 * i + 2].exp();
                sx.max(sy).max(sz)
            })
            .collect();
        max_axes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((n as f32 * cfg.max_axis_pct) as usize).min(n.saturating_sub(1));
        let cutoff = max_axes[idx];
        let dropped = n - idx;
        log::info!(
            "max_axis cutoff = {cutoff:.4} (drops top {dropped} of {n} splats, {:.3}%)",
            (dropped as f32 / n as f32) * 100.0
        );
        cutoff
    };

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
            let raw_scale = Vec3::new(
                log_scales[3 * i].exp(),
                log_scales[3 * i + 1].exp(),
                log_scales[3 * i + 2].exp(),
            );
            if raw_scale.max_element() >= max_axis_cutoff {
                return (Vec::new(), Vec::new());
            }
            // Inflated scale: `sqrt(scale² + filter_3D²)` per axis. This
            // matches GOF's `get_scaling_with_3D_filter` — the effective
            // Gaussian for sampling purposes has at least the
            // pixel-floor minimum size along every axis.
            let f3d = filter_3d[i];
            let f3d_sq = f3d * f3d;
            let scale = Vec3::new(
                (raw_scale.x * raw_scale.x + f3d_sq).sqrt(),
                (raw_scale.y * raw_scale.y + f3d_sq).sqrt(),
                (raw_scale.z * raw_scale.z + f3d_sq).sqrt(),
            );
            // Stored per-point scale: `3 · max_axis_effective`. This is
            // the 3σ half-extent of the *filtered* Gaussian along its
            // widest axis, and matches GOF's `vertices_scale` storage so
            // the filter rule `edge_len > scale_a + scale_b` (≈6σ across
            // two Gaussians) lines up.
            let s_max = scale.max_element() * SIGMA_SCALE;

            let mut pts: Vec<Vec3> = Vec::with_capacity(9);
            let mut scs: Vec<f32> = Vec::with_capacity(9);
            for c in &BOX_CORNERS {
                let local = Vec3::new(c[0], c[1], c[2]) * scale * SIGMA_SCALE;
                let world = mean + q * local;
                if point_in_any_frustum(world, cameras, image_sizes, cfg) {
                    pts.push(world);
                    scs.push(s_max);
                }
            }
            if point_in_any_frustum(mean, cameras, image_sizes, cfg) {
                pts.push(mean);
                scs.push(s_max);
            }
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
    fn emits_nine_points_per_gaussian_when_visible() {
        let (cam, sz) = fake_cam(Vec3::new(0.0, 0.0, -5.0));
        let means = vec![0.0, 0.0, 0.0];
        let quats = vec![1.0, 0.0, 0.0, 0.0];
        let log_scales = vec![-2.0; 3];
        let filter = vec![0.0_f32; 1];
        // max_axis_pct default trims the top 0.1% which for N=1 rounds
        // down to the single splat. Disable for this test.
        let cfg = TetraPointsConfig {
            max_axis_pct: 1.0,
            ..Default::default()
        };
        let out = build_tetra_points(&means, &quats, &log_scales, &filter, &[cam], &[sz], &cfg);
        assert_eq!(out.points.len(), 9);
        assert_eq!(out.scales.len(), 9);
        // s_max = max_axis · 3σ (filter_3D = 0 ⇒ effective scale = raw scale)
        let s = (-2.0f32).exp() * SIGMA_SCALE;
        for sc in &out.scales {
            assert!((sc - s).abs() < 1e-6);
        }
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
        let filter = vec![0.0_f32; 1];
        let out = build_tetra_points(&means, &quats, &log_scales, &filter, &[cam], &[sz], &cfg);
        assert!(out.points.is_empty());
    }
}
