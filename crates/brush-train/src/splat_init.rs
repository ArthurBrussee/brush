use ball_tree::BallTree;
use brush_render::{
    bounding_box::BoundingBox,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid},
};
use brush_serde::SplatData;
use burn::{config::Config, prelude::Backend};
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

/// Create initial splats by sampling random points inside camera frustums.
///
/// For each splat, a random camera is chosen, then a random ray direction
/// within its field of view is sampled, and a random depth along that ray
/// is picked.
pub fn create_random_splats<B: Backend>(
    config: &RandomSplatsConfig,
    cameras: &[Camera],
    scene_scale_override: Option<f32>,
    rng: &mut impl Rng,
    mode: SplatRenderMode,
    device: &B::Device,
) -> Splats<B> {
    let num_points = config.init_count;
    let scene_scale = scene_scale_override.unwrap_or_else(|| estimate_scene_scale(cameras));

    let near = scene_scale * 0.05;
    let far = scene_scale;
    let ln_near = near.ln();
    let ln_far = far.ln();

    // Sample points in camera frustums
    let positions: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            let cam = &cameras[rng.random_range(0..cameras.len())];
            let local_to_world = cam.local_to_world();

            // Random direction within the camera's FOV
            let half_fov_x = (cam.fov_x * 0.5) as f32;
            let half_fov_y = (cam.fov_y * 0.5) as f32;
            let dx = rng.random_range(-half_fov_x..half_fov_x).tan();
            let dy = rng.random_range(-half_fov_y..half_fov_y).tan();

            // Log-uniform depth so we don't over-pack near the camera
            let depth = (rng.random_range(ln_near..ln_far)).exp();

            // Camera looks along -Z in local space
            let local_point = Vec3::new(dx * depth, dy * depth, -depth);
            let world_point = local_to_world.transform_point3(local_point);

            [world_point.x, world_point.y, world_point.z]
        })
        .collect();

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
    // Split into x, y, z values
    let (mut x_vals, mut y_vals, mut z_vals): (Vec<f32>, Vec<f32>, Vec<f32>) = means
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect();

    // Filter out NaN and infinite values before sorting
    x_vals.retain(|x| x.is_finite());
    y_vals.retain(|y| y.is_finite());
    z_vals.retain(|z| z.is_finite());

    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    z_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Get upper and lower percentiles.
    let lower_idx = ((1.0 - percentile) / 2.0 * x_vals.len() as f32) as usize;
    let upper_idx =
        (x_vals.len() - 1).min(((1.0 + percentile) / 2.0 * x_vals.len() as f32) as usize);

    BoundingBox::from_min_max(
        Vec3::new(x_vals[lower_idx], y_vals[lower_idx], z_vals[lower_idx]),
        Vec3::new(x_vals[upper_idx], y_vals[upper_idx], z_vals[upper_idx]),
    )
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

pub fn to_init_splats<B: Backend>(
    data: SplatData,
    mode: SplatRenderMode,
    device: &B::Device,
) -> Splats<B> {
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
