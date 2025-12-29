use ball_tree::BallTree;
use brush_render::{
    bounding_box::BoundingBox,
    gaussian_splats::{SplatRenderMode, Splats, inverse_sigmoid},
};
use brush_serde::SplatData;
use burn::{config::Config, prelude::Backend};
use glam::Vec3;
use rand::Rng;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::trace_span;

#[derive(Config, Debug)]
pub struct RandomSplatsConfig {
    #[config(default = 10000)]
    pub init_count: usize,
}

/// Create initial splats from a random configuration within the given bounds.
pub fn create_random_splats<B: Backend>(
    config: &RandomSplatsConfig,
    bounds: BoundingBox,
    rng: &mut impl Rng,
    mode: SplatRenderMode,
    device: &B::Device,
) -> Splats<B> {
    let num_points = config.init_count;

    let min = bounds.min();
    let max = bounds.max();

    // Random positions within bounds
    let positions: Vec<f32> = (0..num_points)
        .flat_map(|_| {
            [
                rng.random_range(min.x..max.x),
                rng.random_range(min.y..max.y),
                rng.random_range(min.z..max.z),
            ]
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

    // Use a reasonable default scale based on bounds
    let avg_extent = (bounds.extent.x + bounds.extent.y + bounds.extent.z) / 3.0;
    let default_scale = (avg_extent / (num_points as f32).cbrt()).ln();
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
