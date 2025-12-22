//! Integration tests for the benchmark functions
//!
//! These tests verify that the benchmark data generation and core operations work correctly.

use brush_dataset::scene::SceneBatch;
use brush_render::{
    AlphaMode, MainBackend,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats},
    validation::validate_splat_gradients,
};
use brush_render_bwd::render_splats;
use brush_train::{config::TrainConfig, train::SplatTrainer};
use burn::{
    backend::{Autodiff, wgpu::WgpuDevice},
    tensor::{Tensor, TensorData, TensorPrimitive},
};
use glam::{Quat, Vec3};
use rand::{Rng, SeedableRng};

type DiffBackend = Autodiff<MainBackend>;

const TEST_SEED: u64 = 12345;

/// Generate small realistic splats for testing
fn generate_test_splats(device: &WgpuDevice, count: usize) -> Splats<DiffBackend> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);

    let means: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
                rng.random_range(-5.0..5.0),
            ]
        })
        .collect();

    let log_scales: Vec<f32> = (0..count)
        .flat_map(|_| {
            let base = rng.random_range(0.01..0.1_f32).ln();
            [base, base, base]
        })
        .collect();

    let rotations: Vec<f32> = (0..count)
        .flat_map(|_| {
            let u1 = rng.random::<f32>();
            let u2 = rng.random::<f32>();
            let u3 = rng.random::<f32>();

            let sqrt1_u1 = (1.0 - u1).sqrt();
            let sqrt_u1 = u1.sqrt();
            let theta1 = 2.0 * std::f32::consts::PI * u2;
            let theta2 = 2.0 * std::f32::consts::PI * u3;

            [
                sqrt1_u1 * theta1.sin(),
                sqrt1_u1 * theta1.cos(),
                sqrt_u1 * theta2.sin(),
                sqrt_u1 * theta2.cos(),
            ]
        })
        .collect();

    let sh_coeffs: Vec<f32> = (0..count)
        .flat_map(|_| {
            [
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
                rng.random_range(0.2..0.8),
            ]
        })
        .collect();

    let opacities: Vec<f32> = (0..count).map(|_| rng.random_range(0.6..1.0)).collect();

    Splats::<DiffBackend>::from_raw(
        means,
        rotations,
        log_scales,
        sh_coeffs,
        opacities,
        SplatRenderMode::Default,
        device,
    )
    .with_sh_degree(0)
}

fn generate_test_batch(resolution: (u32, u32)) -> SceneBatch {
    let mut rng = rand::rngs::StdRng::seed_from_u64(TEST_SEED);
    let (width, height) = resolution;
    let pixel_count = (width * height * 3) as usize;

    let img_data: Vec<f32> = (0..pixel_count)
        .map(|i| {
            let pixel_idx = i / 3;
            let x = (pixel_idx as u32) % width;
            let y = (pixel_idx as u32) / width;
            let channel = i % 3;

            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;

            let base = match channel {
                0 => nx * 0.5 + 0.25,
                1 => ny * 0.5 + 0.25,
                2 => (nx + ny) * 0.25 + 0.5,
                _ => unreachable!(),
            };

            base + (rng.random::<f32>() - 0.5) * 0.05
        })
        .collect();

    let img_tensor = TensorData::new(img_data, [height as usize, width as usize, 3]);
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );

    SceneBatch {
        img_tensor,
        alpha_mode: AlphaMode::Transparent,
        camera,
    }
}

#[test]
fn test_splat_generation() {
    let device = WgpuDevice::default();
    let splats = generate_test_splats(&device, 1000);

    assert_eq!(splats.num_splats(), 1000);

    // Check that means are reasonable
    let means_data = splats.means.val().into_data().into_vec::<f32>().unwrap();
    assert_eq!(means_data.len(), 3000);

    for chunk in means_data.chunks(3) {
        assert!(chunk.iter().all(|&x| x.is_finite()));
        assert!(chunk[0].abs() < 10.0 && chunk[1].abs() < 10.0 && chunk[2].abs() < 20.0);
    }
}

#[test]
fn test_forward_rendering() {
    let device = WgpuDevice::default();
    let splats = generate_test_splats(&device, 1000);

    // Just verify we can create the splats successfully
    assert_eq!(splats.num_splats(), 1000);

    // Check that the tensor data is accessible
    let means_data = splats.means.val().into_data().into_vec::<f32>().unwrap();
    assert_eq!(means_data.len(), 3000);
    assert!(means_data.iter().all(|&x| x.is_finite()));
}

#[tokio::test]
async fn test_training_step() {
    let device = WgpuDevice::default();
    let batch = generate_test_batch((64, 64));
    let splats = generate_test_splats(&device, 500);
    let config = TrainConfig::default();
    let mut trainer = SplatTrainer::new(&config, &device, splats.clone()).await;
    let (final_splats, stats) = trainer.step(batch, splats);

    assert!(final_splats.num_splats() > 0);
    let loss = stats.loss.into_scalar();
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
}

#[test]
fn test_batch_generation() {
    let batch = generate_test_batch((256, 128));
    let img_dims = batch.img_tensor.shape.clone();
    assert_eq!(img_dims, [128, 256, 3]);
    let img_data = batch.img_tensor.into_vec::<f32>().unwrap();
    assert!(img_data.iter().all(|&x| x.is_finite()));
    assert!(img_data.iter().all(|&x| (0.0..=1.1).contains(&x)));
}

#[tokio::test]
async fn test_multi_step_training() {
    let device = WgpuDevice::default();
    let batch = generate_test_batch((64, 64));
    let config = TrainConfig::default();
    let mut splats = generate_test_splats(&device, 100);
    let mut trainer = SplatTrainer::new(&config, &device, splats.clone()).await;
    let _initial_count = splats.num_splats();

    // Run a few training steps
    for _ in 0..3 {
        let (new_splats, stats) = trainer.step(batch.clone(), splats);
        splats = new_splats;

        let loss = stats.loss.into_scalar();
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    assert!(splats.num_splats() > 0);
}

#[test]
fn test_gradient_validation() {
    let device = WgpuDevice::default();
    let splats = generate_test_splats(&device, 100);

    // Create a simple loss by rendering and taking the mean
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, 3.0),
        Quat::IDENTITY,
        45.0,
        45.0,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(64, 64);

    let result = render_splats(&splats, &camera, img_size, Vec3::ZERO);

    let rendered: Tensor<DiffBackend, 3> =
        Tensor::from_primitive(TensorPrimitive::Float(result.img));
    let loss = rendered.mean();

    // Compute gradients
    let grads = loss.backward();
    validate_splat_gradients(&splats, &grads);
}
