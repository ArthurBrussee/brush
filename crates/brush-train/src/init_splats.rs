use brush_render::{
    bounding_box::BoundingBox,
    gaussian_splats::{Splats, inverse_sigmoid},
};
use burn::{config::Config, prelude::Backend};
use rand::Rng;

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
        positions, rotations, log_scales, sh_coeffs, opacities, device,
    )
}
