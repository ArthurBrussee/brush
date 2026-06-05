use ball_tree::BallTree;
use brush_render::{
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid},
};
use brush_serde::SplatData;
use burn::{config::Config, tensor::Device};
use glam::Vec3;
use rand::{Rng, RngExt};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::trace_span;

#[derive(Config, Debug)]
pub struct RandomSplatsConfig {
    #[config(default = 10000)]
    pub init_count: usize,
}

/// Estimate scene scale from camera positions.
///
/// Uses the average nearest-neighbor distance between cameras,
/// with a minimum of 1.0 (1 meter baseline).
fn estimate_scene_scale(cameras: &[Camera]) -> f32 {
    if cameras.len() < 2 {
        return 1.0;
    }

    let mut total_nn_dist = 0.0f32;
    for (i, cam) in cameras.iter().enumerate() {
        let mut min_dist = f32::INFINITY;
        for (j, other) in cameras.iter().enumerate() {
            if i != j {
                let d = cam.position.distance(other.position);
                if d < min_dist {
                    min_dist = d;
                }
            }
        }
        total_nn_dist += min_dist;
    }

    let avg_nn = total_nn_dist / cameras.len() as f32;
    // Scene depth is roughly a few multiples of the camera spacing.
    // Use 3x the average spacing, with 1m floor.
    (avg_nn * 3.0).max(1.0)
}

/// Sample `num_points` world positions inside the camera frustums, returning
/// `(flattened xyz, scene_scale)`. Depth is bracketed around the scene center
/// as seen from each camera — a minimum standoff out to `d_center +
/// scene_scale` — and sampled uniformly, so splats are spread through the
/// scene volume rather than packed against the lenses. Device-free for testing.
fn sample_frustum_positions(
    cameras: &[Camera],
    scene_scale_override: Option<f32>,
    num_points: usize,
    rng: &mut impl Rng,
) -> (Vec<f32>, f32) {
    // Scene center + radius from the camera rig. For object-centric captures
    // the cameras orbit the scene, so the centroid is ~the scene center and
    // the mean camera-to-centroid distance is ~the scene radius. Fall back to
    // the camera-spacing estimate for degenerate / forward-facing rigs.
    let center =
        cameras.iter().fold(Vec3::ZERO, |acc, c| acc + c.position) / cameras.len().max(1) as f32;
    let mean_radius = cameras
        .iter()
        .map(|c| c.position.distance(center))
        .sum::<f32>()
        / cameras.len().max(1) as f32;
    // The orbit radius is the scene scale for object-centric rigs; only fall
    // back to the camera-spacing estimate when the cameras are ~coincident.
    let scene_scale = scene_scale_override.unwrap_or_else(|| {
        if mean_radius > 1.0 {
            mean_radius
        } else {
            estimate_scene_scale(cameras)
        }
    });

    let min_standoff = scene_scale * 0.2;
    let positions: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            let cam = &cameras[rng.random_range(0..cameras.len())];
            let local_to_world = cam.local_to_world();

            // Random direction within the camera's FOV
            let half_fov_x = (cam.fov_x * 0.5) as f32;
            let half_fov_y = (cam.fov_y * 0.5) as f32;
            let dx = rng.random_range(-half_fov_x..half_fov_x).tan();
            let dy = rng.random_range(-half_fov_y..half_fov_y).tan();

            let d_center = cam.position.distance(center).max(scene_scale);
            let near = (d_center - scene_scale).max(min_standoff);
            let far = d_center + scene_scale;
            let depth = rng.random_range(near..far);

            // Camera looks along +Z in local space (front splats have
            // positive cam-z; negative-z is culled behind the camera).
            let local_point = Vec3::new(dx * depth, dy * depth, depth);
            let world_point = local_to_world.transform_point3(local_point);

            [world_point.x, world_point.y, world_point.z]
        })
        .collect();

    (positions, scene_scale)
}

/// Create initial splats by sampling random points inside camera frustums.
///
/// For each splat, a random camera is chosen, then a random ray direction
/// within its field of view is sampled, and a random depth along that ray
/// is picked.
pub fn create_random_splats(
    config: &RandomSplatsConfig,
    cameras: &[Camera],
    scene_scale_override: Option<f32>,
    rng: &mut impl Rng,
    mode: SplatRenderMode,
    device: &Device,
) -> Splats {
    let num_points = config.init_count;
    let (positions, scene_scale) =
        sample_frustum_positions(cameras, scene_scale_override, num_points, rng);

    // Random colors
    let sh_coeffs: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            [
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
                rng.random_range(0.0..1.0),
            ]
        })
        .collect();

    // Random rotations (normalized quaternions)
    let rotations: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            let x: f32 = rng.random_range(-1.0..1.0);
            let y: f32 = rng.random_range(-1.0..1.0);
            let z: f32 = rng.random_range(-1.0..1.0);
            let w: f32 = rng.random_range(-1.0..1.0);
            let len = (x * x + y * y + z * z + w * w).sqrt().max(1e-6);
            [x / len, y / len, z / len, w / len]
        })
        .collect();

    // Random opacities
    let opacities: Vec<f32> = (0..num_points)
        .map(|_| rng.random_range(inverse_sigmoid(0.1)..inverse_sigmoid(0.25)))
        .collect();

    // Scale based on scene scale and point density
    let default_scale = (scene_scale / (num_points as f32).cbrt()).ln();
    let log_scales: Vec<f32> = vec![default_scale; num_points * 3];

    Splats::from_raw(
        positions, rotations, log_scales, sh_coeffs, opacities, mode, device,
    )
}

pub fn bounds_from_pos(percentile: f32, means: &[f32]) -> BoundingBox {
    let (mut x_vals, mut y_vals, mut z_vals): (Vec<f32>, Vec<f32>, Vec<f32>) = means
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();
    x_vals.retain(|x| x.is_finite());
    y_vals.retain(|y| y.is_finite());
    z_vals.retain(|z| z.is_finite());

    // If any axis is entirely non-finite, fall back to a unit box rather
    // than panicking on the percentile index.
    if x_vals.is_empty() || y_vals.is_empty() || z_vals.is_empty() {
        return BoundingBox::from_min_max(Vec3::splat(-1.0), Vec3::splat(1.0));
    }

    x_vals.sort_by(|a, b| a.total_cmp(b));
    y_vals.sort_by(|a, b| a.total_cmp(b));
    z_vals.sort_by(|a, b| a.total_cmp(b));

    let pick = |vals: &[f32]| -> (f32, f32) {
        let n = vals.len();
        let lo = ((1.0 - percentile) / 2.0 * n as f32) as usize;
        let hi = (n - 1).min(((1.0 + percentile) / 2.0 * n as f32) as usize);
        (vals[lo], vals[hi])
    };

    let (xmin, xmax) = pick(&x_vals);
    let (ymin, ymax) = pick(&y_vals);
    let (zmin, zmax) = pick(&z_vals);
    BoundingBox::from_min_max(Vec3::new(xmin, ymin, zmin), Vec3::new(xmax, ymax, zmax))
}

#[derive(PartialEq, Clone, Copy, Debug)]
struct BallPoint(glam::Vec3A);

impl ball_tree::Point for BallPoint {
    fn distance(&self, other: &Self) -> f64 {
        self.0.distance(other.0) as f64
    }

    fn move_towards(&self, other: &Self, d: f64) -> Self {
        Self(self.0.lerp(other.0, d as f32 / self.0.distance(other.0)))
    }

    fn midpoint(a: &Self, b: &Self) -> Self {
        Self((a.0 + b.0) / 2.0)
    }
}

/// Compute scales using KNN based on point density.
fn compute_knn_scales(pos_data: &[f32]) -> Vec<f32> {
    let _ = trace_span!("compute_knn_scales").entered();

    let n_splats = pos_data.len() / 3;

    if n_splats < 3 {
        return vec![0.0; n_splats * 3];
    }

    let bounding_box = trace_span!("Bounds from pose").in_scope(|| bounds_from_pos(0.75, pos_data));
    let median_size = bounding_box.median_size().max(0.01);

    trace_span!("Splats KNN scale init").in_scope(|| {
        let tree_points: Vec<BallPoint> = pos_data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| BallPoint(glam::Vec3A::new(v[0], v[1], v[2])))
            .collect();

        let empty = vec![(); tree_points.len()];
        let tree = BallTree::new(tree_points.clone(), empty);

        tree_points
            .par_iter()
            .map_with(tree.query(), |query, p| {
                // Get half of the average of 2 nearest distances.
                let mut q = query.nn(p).skip(1);
                let a1 = q.next().unwrap().1 as f32;
                let a2 = q.next().unwrap().1 as f32;
                let dist = (a1 + a2) / 4.0;
                dist.clamp(1e-3, median_size * 0.1).ln()
            })
            .flat_map(|p| [p, p, p])
            .collect()
    })
}

pub fn to_init_splats(data: SplatData, mode: SplatRenderMode, device: &Device) -> Splats {
    let n_splats = data.num_splats();

    // Use KNN for scales if not provided
    let log_scales = data
        .log_scales
        .unwrap_or_else(|| compute_knn_scales(&data.means));

    // Default rotation = identity quaternion [1, 0, 0, 0]
    let rotations = data
        .rotations
        .unwrap_or_else(|| [1.0, 0.0, 0.0, 0.0].repeat(n_splats));

    // Default opacity = inverse_sigmoid(0.5)
    let opacities = data
        .raw_opacities
        .unwrap_or_else(|| vec![inverse_sigmoid(0.5); n_splats]);

    // Default SH coeffs = gray (0.5)
    let sh_coeffs = data.sh_coeffs.unwrap_or_else(|| vec![0.5; n_splats * 3]);

    Splats::from_raw(
        data.means, rotations, log_scales, sh_coeffs, opacities, mode, device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_init_brackets_scene_center() {
        use glam::Quat;
        use rand::SeedableRng;

        // 12 cameras orbiting the origin on a radius-4 ring, all looking in.
        let r = 4.0_f32;
        let cams: Vec<Camera> = (0..12)
            .map(|i| {
                let a = i as f32 / 12.0 * std::f32::consts::TAU;
                let pos = Vec3::new(r * a.cos(), r * a.sin(), 0.0);
                let forward = (-pos).normalize();
                // brush cameras look along +Z in local space.
                let rot = Quat::from_rotation_arc(Vec3::Z, forward);
                Camera::new(
                    pos,
                    rot,
                    0.8,
                    0.8,
                    glam::vec2(0.5, 0.5),
                    brush_render::kernels::camera_model::CameraModel::Pinhole,
                )
            })
            .collect();

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (pos, scene_scale) = sample_frustum_positions(&cams, None, 5000, &mut rng);

        // Scene scale ≈ the orbit radius.
        assert!(
            (scene_scale - r).abs() < 1.5,
            "scene_scale {scene_scale} should be ~{r}"
        );

        // Splats should bracket the scene center, not pile up out at the
        // cameras (the old log-uniform init put the mean near `r`).
        let n = pos.len() / 3;
        let mean_d = pos
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]).length())
            .sum::<f32>()
            / n as f32;
        assert!(
            mean_d < 0.75 * r,
            "mean splat distance from center {mean_d} too large (≥ 0.75·{r}) — init not centered"
        );
        assert!(pos.iter().all(|v| v.is_finite()), "non-finite positions");
    }

    #[test]
    fn bounds_from_pos_all_nan_does_not_panic() {
        let means = vec![f32::NAN; 30];
        let bb = bounds_from_pos(0.8, &means);
        // We expect a finite fallback — no NaN leak, no panic.
        assert!(bb.center.is_finite(), "center: {:?}", bb.center);
        assert!(bb.extent.is_finite(), "extent: {:?}", bb.extent);
    }

    #[test]
    fn bounds_from_pos_empty_does_not_panic() {
        let bb = bounds_from_pos(0.8, &[]);
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
    }

    #[test]
    fn bounds_from_pos_mixed_nan_and_finite() {
        // Half NaN, half finite. The finite half should determine the bounds.
        let mut means = Vec::new();
        for i in 0..100 {
            if i % 2 == 0 {
                means.extend_from_slice(&[f32::NAN, f32::NAN, f32::NAN]);
            } else {
                means.extend_from_slice(&[i as f32, i as f32, i as f32]);
            }
        }
        let bb = bounds_from_pos(0.8, &means);
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
        // Extent should be reasonable (the finite values span 1..99).
        assert!(bb.extent.x > 0.0 && bb.extent.x < 100.0);
    }

    #[test]
    fn bounds_from_pos_one_axis_all_nan() {
        // x and z are OK, y is all NaN — we must not panic indexing into y.
        let mut means = Vec::new();
        for i in 0..50 {
            means.extend_from_slice(&[i as f32, f32::NAN, i as f32]);
        }
        let bb = bounds_from_pos(0.8, &means);
        // y axis collapses to the fallback, other axes should still be
        // reasonable.
        assert!(bb.center.is_finite());
        assert!(bb.extent.is_finite());
    }
}
